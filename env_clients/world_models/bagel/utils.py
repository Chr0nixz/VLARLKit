from __future__ import annotations

import os
from typing import Any

import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

from bagel.data.data_utils import add_special_tokens
from bagel.data.transforms import ImageTransform
from bagel.modeling.autoencoder import load_ae
from bagel.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from bagel.modeling.qwen2 import Qwen2Tokenizer


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the actual device used by the BAGEL language embedding."""
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "embed_tokens")
    ):
        return model.language_model.model.embed_tokens.weight.device
    return next(model.parameters()).device


def move_to_device(generation_input: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors inside a BAGEL generation input dict to the model device."""
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            generation_input[key] = value.to(device)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            generation_input[key] = [item.to(device) for item in value]
    return generation_input


def _resolve_gpu_id(gpu_id: int | None) -> int:
    if gpu_id is not None:
        return int(gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("BAGEL world model requires CUDA, but no GPU is available.")
    return int(torch.cuda.current_device())


def load_model(
    load_model_path: str,
    max_mem_per_gpu: str = "40GiB",
    gpu_id: int | None = None,
    offload_folder: str | None = None,
):
    """Load BAGEL model components for a single worker/process."""
    primary_gpu_id = _resolve_gpu_id(gpu_id)
    torch.cuda.set_device(primary_gpu_id)

    llm_config = Qwen2Config.from_json_file(os.path.join(load_model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(load_model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(load_model_path, "ae.safetensors"))
    vae_model = vae_model.to(dtype=torch.bfloat16, device=f"cuda:{primary_gpu_id}").eval()

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config,
            meta=True,
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(load_model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 256, 16)
    vit_transform = ImageTransform(518, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={primary_gpu_id: max_mem_per_gpu},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    for module_name in same_device_modules:
        device_map[module_name] = primary_gpu_id

    if offload_folder is None:
        offload_folder = os.environ.get(
            "BAGEL_OFFLOAD_FOLDER",
            os.path.join(load_model_path, "offload"),
        )
    os.makedirs(offload_folder, exist_ok=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(load_model_path, "model.safetensors"),
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        offload_folder=offload_folder,
        force_hooks=False,
    )
    model = model.eval()

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
