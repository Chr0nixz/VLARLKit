from typing import Any

import numpy as np

from vlarlkit.data.io_struct import RolloutResult


def _video_array(frames: list[np.ndarray]) -> np.ndarray:
    video = np.stack(frames, axis=0)
    if video.shape[-1] in (1, 3):
        video = np.moveaxis(video, -1, 1)

    if np.issubdtype(video.dtype, np.floating):
        if video.max() <= 1.0:
            video = video * 255.0
        video = np.clip(video, 0, 255).astype(np.uint8)
    elif video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    return video


def build_eval_video_log(
    metric_logger: Any,
    rollout_result: RolloutResult,
    success_once: np.ndarray,
    rollout_epochs: int,
    fps: int = 10,
) -> dict[str, Any]:
    num_steps = len(rollout_result.obs) // rollout_epochs
    terminations = np.stack(rollout_result.terminations)
    truncations = np.stack(rollout_result.truncations)
    num_envs = np.asarray(rollout_result.obs[0]["main_images"]).shape[0]
    success_once = np.asarray(success_once).reshape(rollout_epochs, num_envs)

    video_log = {}
    traj_id = 0
    for rollout_idx in range(rollout_epochs):
        start = rollout_idx * num_steps
        end = start + num_steps
        for env_idx in range(num_envs):
            done = terminations[start:end, env_idx] | truncations[start:end, env_idx]
            done_steps = np.flatnonzero(done)
            stop = start + int(done_steps[0]) + 1 if len(done_steps) else end

            frames = [np.asarray(rollout_result.obs[start]["main_images"])[env_idx]]
            frames.extend(
                np.asarray(rollout_result.next_obs[step]["main_images"])[env_idx]
                for step in range(start, stop)
            )
            success = bool(success_once[rollout_idx, env_idx])
            status = "success" if success else "fail"
            task_description = rollout_result.obs[start]["task_descriptions"][env_idx]
            video_log[f"eval/videos/{status}_traj_{traj_id}"] = metric_logger.Video(
                _video_array(frames),
                fps=fps,
                format="mp4",
                caption=f"success={success} | task={task_description}",
            )
            traj_id += 1
    return video_log
