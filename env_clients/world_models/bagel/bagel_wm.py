from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from bagel.data.data_utils import pil_img2rgb
from bagel.modeling.bagel.qwen2_navit import NaiveCache

from .utils import get_model_device, load_model, move_to_device


_logger = logging.getLogger(__name__)

_WORLD_MODEL_PROMPT = """You are now acting as a **world model** that simulates robot manipulation task execution.
Your task is to predict the **next frame of visual observation**, given the following inputs:
- **Multiple current observation images** from the robot's cameras (head camera and wrist camera)
- An **action sequence** describing the manipulation to execute
- Optionally, the **next frame from the head camera** (for predicting wrist camera views)

You will receive images from different camera viewpoints and need to predict the next frame according to the provided action sequence and instruction."""

_REWARD_PROMPT = """You are a vision-language model with advanced reasoning abilities.
Your task is to carefully observe the image and determine whether the task is successfully completed.

### Environment description:
- You are observing a robot workspace with manipulation capabilities
- The environment is from the LIBERO dataset, containing simulated manipulation tasks
- The robot can manipulate objects in the scene
- Common tasks include: picking, placing, arranging objects, etc.

### Task:
Given an image and a task description, determine whether the task has been successfully completed.

### Response format:
- Answer with "Yes." if the task is successfully completed
- Answer with "No." if the task is not yet completed or failed

### Guidelines:
- Carefully examine the state of objects in the scene
- Check if the goal state matches the task description
- Consider the spatial arrangement and object states
- Be precise in your judgment

**Your response must be either "Yes." or "No." without additional explanation.**"""


def _autocast_context(device: torch.device):
    if torch.device(device).type == "cuda":
        return torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    return nullcontext()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _decode_text_output(tokenizer, token_ids: torch.Tensor) -> str:
    output = tokenizer.decode(token_ids[:, 0])
    if "<|im_end|>" in output:
        output = output.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in output:
        output = output.split("<|im_start|>", 1)[1]
    return output.strip()


def _is_yes_answer(text: str) -> bool:
    normalized = text.strip().lower()
    if "</think>" in normalized:
        normalized = normalized.rsplit("</think>", 1)[-1].strip()
    return "yes" in normalized


def _generation_image_shape(image: Image.Image, vae_transform) -> tuple[int, int]:
    resized = vae_transform.resize_transform(pil_img2rgb(image))
    return resized.size[::-1]


def _decode_image_latent(model, vae_model, latent: torch.Tensor, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    latent_h = height // model.latent_downsample
    latent_w = width // model.latent_downsample
    patch_size = model.latent_patch_size
    latent_channel = model.latent_channel

    latent = latent.reshape(1, latent_h, latent_w, patch_size, patch_size, latent_channel)
    latent = latent.to(torch.bfloat16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, latent_channel, latent_h * patch_size, latent_w * patch_size)
    image = vae_model.decode(latent)
    return (
        ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


@torch.no_grad()
def _batch_pred_next_imgs_cfg(
    model,
    vae_model,
    tokenizer,
    new_token_ids,
    vae_transform,
    vit_transform,
    prompt: Sequence[str],
    images: Sequence[Image.Image],
    num_timesteps: int = 50,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    cfg_type: str = "parallel",
    cfg_interval: Sequence[float] = (0.4, 1.0),
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "text_channel",
    timestep_shift: float = 1.0,
    enable_taylorseer: bool = False,
    save_images: bool = False,
    save_dir: str | None = "./generated_images",
    save_prefix: str = "generated",
) -> list[np.ndarray]:
    """Predict next images with BAGEL CFG image generation."""
    assert len(images) == len(prompt)
    batch_size = len(images)

    images = [vae_transform.resize_transform(pil_img2rgb(image)) for image in images]
    h, w = images[0].size[::-1]
    device = get_model_device(model)

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size

    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=prompt,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = move_to_device(generation_input, device)
    with _autocast_context(device):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input, newlens, new_rope = model.prepare_vae_images(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        images=images,
        transforms=vae_transform,
        new_token_ids=new_token_ids,
        timestep=0.0,
    )
    generation_input = move_to_device(generation_input, device)
    with _autocast_context(device):
        past_key_values = model.forward_cache_update_vae(
            vae_model,
            past_key_values,
            **generation_input,
        )

    generation_input, newlens, new_rope = model.prepare_vit_images(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        images=images,
        transforms=vit_transform,
        new_token_ids=new_token_ids,
    )
    generation_input = move_to_device(generation_input, device)
    with _autocast_context(device):
        past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(h, w)] * batch_size,
        new_token_ids=new_token_ids,
    )

    cfg_text_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_text_newlens = [0] * batch_size
    cfg_text_new_rope = [0] * batch_size

    generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vae_images(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope,
        images=images,
        transforms=vae_transform,
        new_token_ids=new_token_ids,
        timestep=0.0,
    )
    generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
    with _autocast_context(device):
        cfg_text_past_key_values = model.forward_cache_update_vae(
            vae_model,
            cfg_text_past_key_values,
            **generation_input_cfg_text,
        )

    generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vit_images(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope,
        images=images,
        transforms=vit_transform,
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
    with _autocast_context(device):
        cfg_text_past_key_values = model.forward_cache_update_vit(
            cfg_text_past_key_values,
            **generation_input_cfg_text,
        )

    generation_input_cfg_text = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope,
        image_sizes=[(h, w)] * batch_size,
    )
    generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)

    cfg_img_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0] * batch_size
    cfg_img_new_rope = [0] * batch_size

    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        prompts=prompt,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)
    with _autocast_context(device):
        cfg_img_past_key_values = model.forward_cache_update_text(
            cfg_img_past_key_values,
            **generation_input_cfg_img,
        )

    generation_input_cfg_img = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        image_sizes=[(h, w)] * batch_size,
    )
    generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)

    cfg_text_args = {
        "cfg_text_packed_position_ids": generation_input_cfg_text["cfg_packed_position_ids"],
        "cfg_text_packed_query_indexes": generation_input_cfg_text["cfg_packed_query_indexes"],
        "cfg_text_key_values_lens": generation_input_cfg_text["cfg_key_values_lens"],
        "cfg_text_packed_key_value_indexes": generation_input_cfg_text["cfg_packed_key_value_indexes"],
    }
    cfg_img_args = {
        "cfg_img_packed_position_ids": generation_input_cfg_img["cfg_packed_position_ids"],
        "cfg_img_packed_query_indexes": generation_input_cfg_img["cfg_packed_query_indexes"],
        "cfg_img_key_values_lens": generation_input_cfg_img["cfg_key_values_lens"],
        "cfg_img_packed_key_value_indexes": generation_input_cfg_img["cfg_packed_key_value_indexes"],
    }

    generation_input = move_to_device(generation_input, device)
    with _autocast_context(device):
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            **cfg_text_args,
            **cfg_img_args,
            enable_taylorseer=enable_taylorseer,
        )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16).to(torch.bfloat16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, h // 8, w // 8)
        image = vae_model.decode(latent)
        image = (
            ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        image_list.append(image)

    if save_images and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, image_array in enumerate(image_list):
            save_path = os.path.join(save_dir, f"{save_prefix}_{timestamp}_batch{idx}.png")
            Image.fromarray(image_array).save(save_path)
            _logger.info("Saved generated image to: %s", save_path)

    return image_list


@torch.no_grad()
def _batch_pred_next_imgs_cfg_multi_input(
    model,
    vae_model,
    tokenizer,
    new_token_ids,
    vae_transform,
    vit_transform,
    prompt: str,
    images: Sequence[Sequence[Image.Image]],
    actions: Sequence[str],
    num_timesteps: int = 50,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    cfg_type: str = "parallel",
    cfg_interval: Sequence[float] = (0.4, 1.0),
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "text_channel",
    timestep_shift: float = 1.0,
    enable_taylorseer: bool = False,
    save_images: bool = False,
    save_dir: str | None = "./generated_images",
    save_prefix: str = "generated",
) -> list[np.ndarray]:
    if len(images) != len(actions):
        raise ValueError(f"Expected {len(images)} action prompts, got {len(actions)}.")
    if len(images) == 0:
        return []

    batch_size = len(images)
    num_cond_images = len(images[0])
    if num_cond_images == 0:
        raise ValueError("At least one conditioning image is required.")
    for sample_images in images:
        if len(sample_images) != num_cond_images:
            raise ValueError("All samples must contain the same number of conditioning images.")

    image_shape = _generation_image_shape(images[0][0], vae_transform)
    device = get_model_device(model)

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size

    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = move_to_device(generation_input, device)
    with _autocast_context(device):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    for cond_idx in range(num_cond_images):
        cond_images = [sample_images[cond_idx] for sample_images in images]
        generation_input, newlens, new_rope = model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=cond_images,
            transforms=vae_transform,
            new_token_ids=new_token_ids,
            timestep=0.0,
        )
        generation_input = move_to_device(generation_input, device)
        with _autocast_context(device):
            past_key_values = model.forward_cache_update_vae(
                vae_model,
                past_key_values,
                **generation_input,
            )

        generation_input, newlens, new_rope = model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=cond_images,
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        generation_input = move_to_device(generation_input, device)
        with _autocast_context(device):
            past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=actions,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = move_to_device(generation_input, device)
    with _autocast_context(device):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[image_shape] * batch_size,
        new_token_ids=new_token_ids,
    )
    generation_input = move_to_device(generation_input, device)

    cfg_text_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_text_newlens = [0] * batch_size
    cfg_text_new_rope = [0] * batch_size

    generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope,
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
    with _autocast_context(device):
        cfg_text_past_key_values = model.forward_cache_update_text(
            cfg_text_past_key_values,
            **generation_input_cfg_text,
        )

    for cond_idx in range(num_cond_images):
        cond_images = [sample_images[cond_idx] for sample_images in images]
        generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vae_images(
            curr_kvlens=cfg_text_newlens,
            curr_rope=cfg_text_new_rope,
            images=cond_images,
            transforms=vae_transform,
            new_token_ids=new_token_ids,
            timestep=0.0,
        )
        generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
        with _autocast_context(device):
            cfg_text_past_key_values = model.forward_cache_update_vae(
                vae_model,
                cfg_text_past_key_values,
                **generation_input_cfg_text,
            )

        generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vit_images(
            curr_kvlens=cfg_text_newlens,
            curr_rope=cfg_text_new_rope,
            images=cond_images,
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
        with _autocast_context(device):
            cfg_text_past_key_values = model.forward_cache_update_vit(
                cfg_text_past_key_values,
                **generation_input_cfg_text,
            )

    generation_input_cfg_text = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope,
        image_sizes=[image_shape] * batch_size,
    )
    generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)

    cfg_img_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0] * batch_size
    cfg_img_new_rope = [0] * batch_size

    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)
    with _autocast_context(device):
        cfg_img_past_key_values = model.forward_cache_update_text(
            cfg_img_past_key_values,
            **generation_input_cfg_img,
        )

    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        prompts=actions,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)
    with _autocast_context(device):
        cfg_img_past_key_values = model.forward_cache_update_text(
            cfg_img_past_key_values,
            **generation_input_cfg_img,
        )

    generation_input_cfg_img = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        image_sizes=[image_shape] * batch_size,
    )
    generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)

    cfg_text_args = {
        "cfg_text_packed_position_ids": generation_input_cfg_text["cfg_packed_position_ids"],
        "cfg_text_packed_query_indexes": generation_input_cfg_text["cfg_packed_query_indexes"],
        "cfg_text_key_values_lens": generation_input_cfg_text["cfg_key_values_lens"],
        "cfg_text_packed_key_value_indexes": generation_input_cfg_text["cfg_packed_key_value_indexes"],
    }
    cfg_img_args = {
        "cfg_img_packed_position_ids": generation_input_cfg_img["cfg_packed_position_ids"],
        "cfg_img_packed_query_indexes": generation_input_cfg_img["cfg_packed_query_indexes"],
        "cfg_img_key_values_lens": generation_input_cfg_img["cfg_key_values_lens"],
        "cfg_img_packed_key_value_indexes": generation_input_cfg_img["cfg_packed_key_value_indexes"],
    }

    with _autocast_context(device):
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            **cfg_text_args,
            **cfg_img_args,
            enable_taylorseer=enable_taylorseer,
        )

    image_list = [
        _decode_image_latent(model, vae_model, latent, image_shape)
        for latent in unpacked_latent
    ]

    if save_images and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, image_array in enumerate(image_list):
            save_path = os.path.join(save_dir, f"{save_prefix}_{timestamp}_batch{idx}.png")
            Image.fromarray(image_array).save(save_path)
            _logger.info("Saved generated image to: %s", save_path)

    return image_list


class BagelWM:
    """BAGEL-backed world model for image observation and reward prediction."""

    def __init__(
        self,
        load_model_path: str,
        gpu_id: int | None = None,
        max_mem_per_gpu: str = "40GiB",
        offload_folder: str | None = None,
        edit_cfg_text_scale: float = 4.0,
        edit_cfg_img_scale: float = 2.0,
        edit_cfg_interval: Sequence[float] = (0.4, 1.0),
        edit_timestep_shift: float = 3.0,
        edit_num_timesteps: int = 50,
        edit_cfg_renorm_min: float = 0.0,
        edit_cfg_renorm_type: str = "text_channel",
        save_images: bool = False,
        save_dir: str = "./generated_images",
        save_prefix: str = "generated",
        understand_max_tokens: int = 1000,
        understand_do_sample: bool = False,
        understand_temperature: float = 0.3,
    ):
        self._closed = False

        self._model, self._vae_model, self._tokenizer, self._vae_transform, self._vit_transform, self._new_token_ids = load_model(
            load_model_path=load_model_path,
            max_mem_per_gpu=max_mem_per_gpu,
            gpu_id=gpu_id,
            offload_folder=offload_folder,
        )
        self._device = get_model_device(self._model)
        action_norm_path = os.path.join(load_model_path, "action_normalizer.json")
        self._action_normalizer = self._load_action_normalizer(action_norm_path)

        self._edit_hyper = {
            "cfg_text_scale": edit_cfg_text_scale,
            "cfg_img_scale": edit_cfg_img_scale,
            "cfg_interval": list(edit_cfg_interval),
            "timestep_shift": edit_timestep_shift,
            "num_timesteps": edit_num_timesteps,
            "cfg_renorm_min": edit_cfg_renorm_min,
            "cfg_renorm_type": edit_cfg_renorm_type,
            "save_images": save_images,
            "save_dir": save_dir,
            "save_prefix": save_prefix,
            "enable_taylorseer": True,
        }
        self._understand_hyper = {
            "max_length": understand_max_tokens,
            "do_sample": understand_do_sample,
            "temperature": understand_temperature,
        }

    @staticmethod
    def _load_action_normalizer(action_norm_path: str) -> dict[str, np.ndarray]:
        with open(action_norm_path, "r") as f:
            normalizer = json.load(f)
        min_values = np.asarray(normalizer["min"], dtype=np.float32)
        max_values = np.asarray(normalizer["max"], dtype=np.float32)
        clip_min = np.asarray(normalizer.get("clip_min", min_values), dtype=np.float32)
        clip_max = np.asarray(normalizer.get("clip_max", max_values), dtype=np.float32)
        denom = max_values - min_values
        zero_range = denom == 0
        denom = denom.copy()
        denom[zero_range] = 1.0
        return {
            "min": min_values,
            "max": max_values,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "denom": denom,
            "zero_range": zero_range,
        }

    @staticmethod
    def _prepare_image_batch(images: Any) -> np.ndarray:
        images = _to_numpy(images)
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected image batch [N, H, W, 3], got shape {images.shape}.")
        if images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)
        return images

    @staticmethod
    def _prepare_tasks(tasks: Any, batch_size: int) -> list[str]:
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        elif isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        else:
            tasks = list(tasks)

        if len(tasks) == 1 and batch_size > 1:
            tasks = tasks * batch_size
        if len(tasks) != batch_size:
            raise ValueError(f"Expected {batch_size} tasks, got {len(tasks)}.")
        return [str(task) for task in tasks]

    def _format_action_step(self, action: np.ndarray) -> str:
        normalizer = self._action_normalizer
        if action.shape[-1] != normalizer["min"].shape[0]:
            raise ValueError(
                f"Expected action dim {normalizer['min'].shape[0]}, got {action.shape[-1]}."
            )
        action = np.clip(action, normalizer["clip_min"], normalizer["clip_max"])
        normalized = ((action - normalizer["min"]) / normalizer["denom"] * 256).astype(int)
        normalized = np.clip(normalized, 0, 256)
        normalized[normalizer["zero_range"]] = 128
        return ", ".join(str(int(item)) for item in normalized)

    def _format_action_prompts(self, actions: Any, batch_size: int) -> list[str]:
        actions = _to_numpy(actions).astype(np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        if actions.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} actions, got {actions.shape[0]}.")
        if actions.ndim not in (2, 3):
            raise ValueError(f"Expected actions [N, D] or [N, T, D], got shape {actions.shape}.")

        action_prompts = []
        for batch_idx in range(batch_size):
            step_strings = []
            sample_actions = actions[batch_idx]
            if actions.ndim == 2:
                sample_actions = sample_actions[None]
            for step_idx, step_action in enumerate(sample_actions):
                step_strings.append(f"Step {step_idx}: [{self._format_action_step(step_action)}]")
            action_prompts.append("; ".join(step_strings))
        return action_prompts

    def _predict_next_views(
        self,
        head_images: np.ndarray,
        wrist_images: np.ndarray,
        actions: Any,
        inference_hyper: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        action_prompts = self._format_action_prompts(actions, batch_size=head_images.shape[0])
        pil_head_images = [Image.fromarray(image) for image in head_images]
        pil_wrist_images = [Image.fromarray(image) for image in wrist_images]

        stage1_actions = [
            f"{action}. Predict next head camera view according to the current observation and action."
            for action in action_prompts
        ]
        stage1_input_images = [
            [head_image, wrist_image]
            for head_image, wrist_image in zip(pil_head_images, pil_wrist_images)
        ]

        base_prefix = inference_hyper.get("save_prefix", "generated")
        next_head_images = _batch_pred_next_imgs_cfg_multi_input(
            model=self._model,
            vae_model=self._vae_model,
            tokenizer=self._tokenizer,
            new_token_ids=self._new_token_ids,
            vae_transform=self._vae_transform,
            vit_transform=self._vit_transform,
            prompt=_WORLD_MODEL_PROMPT,
            images=stage1_input_images,
            actions=stage1_actions,
            num_timesteps=inference_hyper["num_timesteps"],
            cfg_text_scale=inference_hyper["cfg_text_scale"],
            cfg_img_scale=inference_hyper["cfg_img_scale"],
            cfg_type=inference_hyper.get("cfg_type", "parallel"),
            cfg_interval=inference_hyper["cfg_interval"],
            cfg_renorm_min=inference_hyper["cfg_renorm_min"],
            cfg_renorm_type=inference_hyper["cfg_renorm_type"],
            timestep_shift=inference_hyper["timestep_shift"],
            enable_taylorseer=inference_hyper.get("enable_taylorseer", True),
            save_images=inference_hyper.get("save_images", False),
            save_dir=inference_hyper.get("save_dir"),
            save_prefix=inference_hyper.get("save_prefix_head", f"{base_prefix}_head"),
        )

        pil_next_head_images = [Image.fromarray(image) for image in next_head_images]
        stage2_input_images = [
            [next_head_image, wrist_image]
            for next_head_image, wrist_image in zip(pil_next_head_images, pil_wrist_images)
        ]
        stage2_actions = [
            "Predict current wrist camera view according to history wrist camera view and current head camera view."
            for _ in pil_wrist_images
        ]
        next_wrist_images = _batch_pred_next_imgs_cfg_multi_input(
            model=self._model,
            vae_model=self._vae_model,
            tokenizer=self._tokenizer,
            new_token_ids=self._new_token_ids,
            vae_transform=self._vae_transform,
            vit_transform=self._vit_transform,
            prompt=_WORLD_MODEL_PROMPT,
            images=stage2_input_images,
            actions=stage2_actions,
            num_timesteps=inference_hyper["num_timesteps"],
            cfg_text_scale=inference_hyper["cfg_text_scale"],
            cfg_img_scale=inference_hyper["cfg_img_scale"],
            cfg_type=inference_hyper.get("cfg_type", "parallel"),
            cfg_interval=inference_hyper["cfg_interval"],
            cfg_renorm_min=inference_hyper["cfg_renorm_min"],
            cfg_renorm_type=inference_hyper["cfg_renorm_type"],
            timestep_shift=inference_hyper["timestep_shift"],
            enable_taylorseer=inference_hyper.get("enable_taylorseer", True),
            save_images=inference_hyper.get("save_images", False),
            save_dir=inference_hyper.get("save_dir"),
            save_prefix=inference_hyper.get("save_prefix_wrist", f"{base_prefix}_wrist"),
        )
        return (
            np.asarray(next_head_images, dtype=np.uint8),
            np.asarray(next_wrist_images, dtype=np.uint8),
        )

    @torch.no_grad()
    def get_observations(
        self,
        observations: dict[str, Any],
        actions: Any,
        image_key: str = "main_images",
        wrist_image_key: str = "wrist_images",
        **kwargs,
    ) -> dict[str, Any]:
        if image_key not in observations:
            raise KeyError(f"Observation does not contain image key '{image_key}'.")
        if wrist_image_key not in observations:
            raise KeyError(f"Observation does not contain wrist image key '{wrist_image_key}'.")

        head_images = self._prepare_image_batch(observations[image_key])
        wrist_images = self._prepare_image_batch(observations[wrist_image_key])
        if head_images.shape[0] != wrist_images.shape[0]:
            raise ValueError(
                f"Expected matching image batch sizes, got {head_images.shape[0]} "
                f"and {wrist_images.shape[0]}."
            )

        inference_hyper = {**self._edit_hyper, **kwargs}
        next_head_images, next_wrist_images = self._predict_next_views(
            head_images=head_images,
            wrist_images=wrist_images,
            actions=actions,
            inference_hyper=inference_hyper,
        )

        next_observations = dict(observations)
        next_observations[image_key] = next_head_images
        next_observations[wrist_image_key] = next_wrist_images
        return next_observations

    @torch.no_grad()
    def get_rewards(
        self,
        observations: dict[str, Any],
        tasks: Sequence[str] | None = None,
        image_key: str = "main_images",
        **kwargs,
    ) -> np.ndarray:
        if image_key not in observations:
            raise KeyError(f"Observation does not contain image key '{image_key}'.")
        if tasks is None:
            tasks = observations.get("task_descriptions")
        if tasks is None:
            raise ValueError("tasks must be provided or present as observations['task_descriptions'].")

        images = self._prepare_image_batch(observations[image_key])
        task_list = self._prepare_tasks(tasks, batch_size=images.shape[0])
        inference_hyper = {**self._understand_hyper, **kwargs}
        return self._predict_success_rewards(images, task_list, **inference_hyper)

    @torch.no_grad()
    def _predict_success_rewards(
        self,
        images: np.ndarray,
        tasks: Sequence[str],
        max_length: int = 1000,
        do_sample: bool = False,
        temperature: float = 0.3,
        max_think_token_n: int | None = None,
        text_temperature: float | None = None,
        num_samples: int | None = None,
    ) -> np.ndarray:
        if max_think_token_n is not None:
            max_length = max_think_token_n
        if text_temperature is not None:
            temperature = text_temperature

        rewards = []
        for image, task in zip(images, tasks):
            pil_image = Image.fromarray(image)
            prompt = (
                f"\n{_REWARD_PROMPT}\nDetermine whether the task: {task} is "
                "successfully completed, answer with Yes or No"
            )
            text = self._generate_understanding_text(
                image=pil_image,
                prompt=prompt,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )
            rewards.append(1.0 if _is_yes_answer(text) else 0.0)
        return np.asarray(rewards, dtype=np.float32)

    @torch.no_grad()
    def _generate_understanding_text(
        self,
        image: Image.Image,
        prompt: str,
        max_length: int,
        do_sample: bool,
        temperature: float,
    ) -> str:
        image = pil_img2rgb(image)
        original_image_size = image.size
        past_key_values = NaiveCache(self._model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        generation_input, newlens, new_rope = self._model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image.resize(original_image_size)],
            transforms=self._vit_transform,
            new_token_ids=self._new_token_ids,
        )
        generation_input = move_to_device(generation_input, self._device)
        with _autocast_context(self._device):
            past_key_values = self._model.forward_cache_update_vit(
                past_key_values,
                **generation_input,
            )

        generation_input, newlens, new_rope = self._model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt],
            tokenizer=self._tokenizer,
            new_token_ids=self._new_token_ids,
        )
        generation_input = move_to_device(generation_input, self._device)
        with _autocast_context(self._device):
            past_key_values = self._model.forward_cache_update_text(
                past_key_values,
                **generation_input,
            )

        generation_input = self._model.prepare_start_tokens(newlens, new_rope, self._new_token_ids)
        generation_input = move_to_device(generation_input, self._device)
        with _autocast_context(self._device):
            token_ids = self._model.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=self._new_token_ids["eos_token_id"],
                **generation_input,
            )
        return _decode_text_output(self._tokenizer, token_ids)

    def close(self) -> dict[str, str]:
        if self._closed:
            return {"status": "already_closed"}
        self._closed = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "closed"}
