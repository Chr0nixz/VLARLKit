# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.
#
# --------------------------------------------------------------------
# Modifications:
#   Modified by VLARLKit Authors on 2026-06-07.
# --------------------------------------------------------------------

import json
import os

import torch
from omegaconf import DictConfig
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoTokenizer


def _resolve_torch_dtype(precision: str | None) -> torch.dtype:
    if precision is None:
        return torch.bfloat16
    precision = str(precision).lower()
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    if precision in ("fp16", "float16"):
        return torch.float16
    if precision in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported OpenVLA-OFT precision: {precision!r}")


def _register_prismatic_classes():
    from prismatic.extern.hf.configuration_prismatic import (
        OpenVLAConfig as OpenVLAOFTConfig,
    )

    from vlarlkit_openvla_oft.processing_prismatic import (
        MultiInputPrismaticProcessor,
        PrismaticImageProcessor,
    )

    AutoConfig.register("openvla", OpenVLAOFTConfig)
    AutoImageProcessor.register(OpenVLAOFTConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAOFTConfig, MultiInputPrismaticProcessor)
    return OpenVLAOFTConfig, PrismaticImageProcessor, MultiInputPrismaticProcessor


def _load_model_config(cfg: DictConfig):
    OpenVLAOFTConfig, _, _ = _register_prismatic_classes()
    actor_model_config = AutoConfig.from_pretrained(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if not isinstance(actor_model_config, OpenVLAOFTConfig):
        actor_model_config = OpenVLAOFTConfig(**actor_model_config.to_dict())

    dataset_statistics_path = os.path.join(cfg.model_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path) as f:
            norm_stats = getattr(actor_model_config, "norm_stats", {})
            norm_stats.update(json.load(f))
            setattr(actor_model_config, "norm_stats", norm_stats)

    for key, val in cfg.items():
        setattr(actor_model_config, key, val)
    return actor_model_config


def _get_input_processor(cfg: DictConfig):
    OpenVLAOFTConfig, PrismaticImageProcessor, MultiInputPrismaticProcessor = (
        _register_prismatic_classes()
    )
    model_config = OpenVLAOFTConfig.from_pretrained(
        cfg.model_path,
        center_crop=cfg.center_crop,
    )
    image_processor = PrismaticImageProcessor.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    input_processor = MultiInputPrismaticProcessor.from_pretrained(
        cfg.model_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        trust_remote_code=True,
    )
    return model_config, input_processor


LORA_TARGET_MODULES = [
    "proj",
    "qkv",
    "fc1",
    "fc2",
    "q",
    "kv",
    "fc3",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]


def _apply_lora(model, cfg: DictConfig):
    if not bool(cfg.get("is_lora", False)):
        setattr(model, "_is_lora", False)
        return model

    from peft import LoraConfig, PeftModel, get_peft_model

    lora_path = cfg.get("lora_path", None)
    if lora_path is None:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank,
            lora_dropout=0.0,
            target_modules=LORA_TARGET_MODULES,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    if hasattr(model, "value_head"):
        for param in model.value_head.parameters():
            param.requires_grad = True
    setattr(model, "_is_lora", True)
    return model


def get_model(cfg: DictConfig):
    implement_version = cfg.get("implement_version", "rlinf")
    if implement_version != "rlinf":
        raise NotImplementedError(
            "VLARLKit currently supports only the RLinf OpenVLA-OFT implementation."
        )

    from vlarlkit_openvla_oft.openvla_oft_action_model import (
        OpenVLAOFTForRLActionPrediction,
    )

    _register_prismatic_classes()
    torch_dtype = _resolve_torch_dtype(cfg.get("precision", "bf16"))
    actor_model_config = _load_model_config(cfg)

    model = OpenVLAOFTForRLActionPrediction.from_pretrained(
        pretrained_model_name_or_path=cfg.model_path,
        torch_dtype=torch_dtype,
        config=actor_model_config,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        trust_remote_code=cfg.trust_remote_code,
        add_value_head=cfg.add_value_head,
        max_prompt_length=cfg.max_prompt_length,
        low_cpu_mem_usage=cfg.get("low_cpu_mem_usage", True),
        # Match RLinf's OpenVLA-OFT loader: this action model expects the default
        # eager attention path for its action-token prediction layout.
    )
    model.vision_backbone.set_num_images_in_input(cfg.get("num_images_in_input", 1))
    model.to(torch_dtype)

    model_config, input_processor = _get_input_processor(cfg)
    model.setup_config_and_processor(model_config, input_processor)
    model = _apply_lora(model, cfg)
    return model


__all__ = ["get_model"]
