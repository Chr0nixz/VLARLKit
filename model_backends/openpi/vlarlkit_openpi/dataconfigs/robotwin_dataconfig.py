# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --------------------------------------------------------------------
# Modifications:
#   Modified by VLARLKit Authors on 2026-06-01.
# --------------------------------------------------------------------

import dataclasses
from typing import ClassVar
import pathlib

import einops
import numpy as np
from openpi import transforms
from openpi.models import model
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


def _joint_flip_mask() -> np.ndarray:
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        temp_val = (horn_radius**2 + linear_position**2 - arm_length**2) / (
            2 * horn_radius * linear_position
        )
        return np.arcsin(np.clip(temp_val, -1.0, 1.0))

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    value = value + 0.5476
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _convert_image(image):
    if image is None:
        return None
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if adapt_to_pi:
        state = _joint_flip_mask() * state
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state.astype(np.float32)


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions.astype(np.float32)


def _encode_actions_inv(
    actions: np.ndarray, *, adapt_to_pi: bool = False
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions.astype(np.float32)


def _decode_robotwin(data: dict, *, adapt_to_pi: bool = False) -> dict:
    if "observation/state" in data:
        images = {
            "cam_high": _convert_image(data["observation/image"]),
        }
        wrist_images = data.get("observation/wrist_image", None)
        if wrist_images is not None:
            wrist_images = np.asarray(wrist_images)
            if wrist_images.ndim == 4:
                images["cam_left_wrist"] = _convert_image(wrist_images[0])
                if wrist_images.shape[0] > 1:
                    images["cam_right_wrist"] = _convert_image(wrist_images[1])
            elif wrist_images.ndim == 3:
                images["cam_left_wrist"] = _convert_image(wrist_images)
        state = data["observation/state"]
    else:
        images = {name: _convert_image(image) for name, image in data["images"].items()}
        state = data["state"]

    data["images"] = images
    data["state"] = _decode_state(state, adapt_to_pi=adapt_to_pi)
    return data


@dataclasses.dataclass(frozen=True)
class RobotwinInputs(transforms.DataTransformFn):
    adapt_to_pi: bool = True

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_robotwin(data, adapt_to_pi=self.adapt_to_pi)
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}"
            )

        base_image = in_images["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images and in_images[source] is not None:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            inputs["actions"] = _encode_actions_inv(
                data["actions"], adapt_to_pi=self.adapt_to_pi
            )
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobotwinOutputs(transforms.DataTransformFn):
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


@dataclasses.dataclass(frozen=True)
class LeRobotRobotwinDataConfig(DataConfigFactory):
    default_prompt: str | None = None
    extra_delta_transform: bool = True
    adapt_to_pi: bool = True

    repack_transforms: transforms.Group = dataclasses.field(
        default_factory=lambda: transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = transforms.Group(
            inputs=[RobotwinInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[RobotwinOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.extra_delta_transform:
            delta_action_mask = np.array(
                [True] * 6 + [False] + [True] * 6 + [False],
                dtype=bool,
            )
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_action_mask)],
                outputs=[transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
