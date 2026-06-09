from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange

from vlarlkit.utils.conversion_utils import to_numpy


@dataclass(kw_only=True)
class RolloutResult:
    """
    Results of multiple-epoch rollouts.
    """

    obs: list[dict[str, Any]] = field(default_factory=list)
    next_obs: list[dict[str, Any]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    terminations: list[np.ndarray] = field(default_factory=list)
    truncations: list[np.ndarray] = field(default_factory=list)
    prev_logprobs: list[np.ndarray] = field(default_factory=list)
    prev_values: list[np.ndarray] = field(default_factory=list)
    forward_inputs: list[dict[str, Any]] = field(default_factory=list)
    next_forward_inputs: list[dict[str, Any]] = field(default_factory=list)

    returns: np.ndarray | None = field(default=None, repr=False)
    advantages: np.ndarray | None = field(default=None, repr=False)
    _loss_filter_mask: np.ndarray | None = field(default=None, repr=False)

    def clear(self) -> None:
        """Clear all list fields and reset returns/advantages for a new rollout."""
        self.obs.clear()
        self.next_obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.prev_logprobs.clear()
        self.prev_values.clear()
        self.forward_inputs.clear()
        self.next_forward_inputs.clear()
        self.returns = None
        self.advantages = None
        self._loss_filter_mask = None

    def append_step(
        self,
        obs: Any,
        next_obs: Any,
        actions: Any,
        rewards: Any,
        terminations: Any,
        truncations: Any,
        prev_logprobs: Optional[Any] = None,
        prev_values: Optional[Any] = None,
        forward_inputs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.terminations.append(terminations)
        self.truncations.append(truncations)
        if prev_logprobs is not None:
            self.prev_logprobs.append(to_numpy(prev_logprobs))
        if prev_values is not None:
            self.prev_values.append(to_numpy(prev_values))
        if forward_inputs is not None:
            self.forward_inputs.append(to_numpy(forward_inputs))

    def build_next_forward_inputs(self):
        """Pure temporal shift: next_fi[t] = fi[t+1]. Requires len(forward_inputs) == T+1."""
        T = len(self.obs)
        assert len(self.forward_inputs) == T + 1, (
            f"Expected {T + 1} forward_inputs, got {len(self.forward_inputs)}"
        )
        self.next_forward_inputs = [self.forward_inputs[t + 1] for t in range(T)]

    def compute_gae_advantages(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> dict[str, Any]:
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Bootstrap is handled upstream by embedding gamma * V(next_obs) into
        the rewards at episode boundaries (see Rollout._get_bootstrap_values).

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter controlling the bias-variance trade-off.
        """
        self._loss_filter_mask = None
        # chunk-level: rewards/values are [T, B].
        # action-level: rewards/values are [T, B, C], one scalar per action step.
        rewards = np.stack(self.rewards).astype(np.float32)
        values = np.stack(self.prev_values).astype(np.float32)
        if values.ndim > rewards.ndim and values.shape[-1] == 1:
            values = values.squeeze(-1)
        # Same shape as rewards: [T, B] for chunk-level or [T, B, C] for action-level.
        terminations = np.stack(self.terminations)
        truncations = np.stack(self.truncations)

        original_shape = rewards.shape
        if rewards.ndim == 3:
            num_steps, num_envs, chunk_size = rewards.shape
            rewards_for_gae = rearrange(rewards, "t b c -> (t c) b")
            values_for_gae = rearrange(values, "t b c -> (t c) b")
            terminations_for_gae = rearrange(terminations, "t b c -> (t c) b")
            truncations_for_gae = rearrange(truncations, "t b c -> (t c) b")
        elif rewards.ndim == 2:
            rewards_for_gae = rewards
            values_for_gae = values
            terminations_for_gae = terminations
            truncations_for_gae = truncations
        else:
            raise ValueError(f"Unsupported rewards shape for GAE: {rewards.shape}")

        num_primitive_steps = rewards_for_gae.shape[0]
        dones = np.logical_or(terminations_for_gae, truncations_for_gae)

        advantages = np.zeros_like(rewards_for_gae)
        last_gae_lam = np.zeros_like(rewards_for_gae[0])

        for t in reversed(range(num_primitive_steps)):
            next_values = (
                np.zeros_like(values_for_gae[0])
                if t == num_primitive_steps - 1
                else values_for_gae[t + 1]
            )
            not_done = (~dones[t]).astype(np.float32)

            delta = rewards_for_gae[t] + gamma * next_values * not_done - values_for_gae[t]
            last_gae_lam = delta + gamma * gae_lambda * not_done * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values_for_gae
        if len(original_shape) == 3:
            self.advantages = rearrange(
                advantages,
                "(t c) b -> t b c",
                t=num_steps,
                b=num_envs,
                c=chunk_size,
            )
            self.returns = rearrange(
                returns,
                "(t c) b -> t b c",
                t=num_steps,
                b=num_envs,
                c=chunk_size,
            )
        else:
            self.advantages = advantages
            self.returns = returns
        return {
            "advantages": self.advantages,
            "returns": self.returns,
        }

    def compute_grpo_advantages(
        self,
        episode_len: int,
        group_size: int,
        filter_rewards: bool = False,
        rewards_lower_bound: float = 0.0,
        rewards_upper_bound: float = 1.0,
    ) -> dict[str, Any]:
        """Compute group-relative advantages from per-episode scores."""
        rewards = np.stack(self.rewards).astype(np.float32)  # (T, B[, C])
        original_shape = rewards.shape
        if rewards.ndim == 3:
            T, n_envs, chunk_size = rewards.shape
            rewards_for_scores = rearrange(rewards, "t b c -> (t c) b")
        elif rewards.ndim == 2:
            T, n_envs = rewards.shape
            chunk_size = None
            rewards_for_scores = rewards
        else:
            raise ValueError(f"Unsupported rewards shape for GRPO: {rewards.shape}")

        assert group_size > 0, f"group_size must be positive, got {group_size}"
        primitive_T = rewards_for_scores.shape[0]
        assert primitive_T % episode_len == 0, (
            f"T ({primitive_T}) must be divisible by episode_len ({episode_len})"
        )
        assert n_envs % group_size == 0, (
            f"n_envs ({n_envs}) must be divisible by group_size ({group_size})"
        )

        num_episodes = primitive_T // episode_len
        num_groups = n_envs // group_size
        loss_mask, _ = self._compute_episode_loss_mask(episode_len)
        loss_mask_for_scores = (
            rearrange(loss_mask, "t b c -> (t c) b")
            if loss_mask.ndim == 3
            else loss_mask
        )
        raw_loss_mask_fraction = float(loss_mask_for_scores.mean())
        rewards_for_scores = rewards_for_scores.reshape(num_episodes, episode_len, n_envs)
        loss_mask_for_scores = loss_mask_for_scores.reshape(
            num_episodes, episode_len, n_envs
        )
        scores = (rewards_for_scores * loss_mask_for_scores.astype(np.float32)).sum(axis=1)

        grouped_scores = scores.reshape(num_episodes, num_groups, group_size)
        group_mean = grouped_scores.mean(axis=-1, keepdims=True)
        if group_size > 1:
            group_std = grouped_scores.std(axis=-1, ddof=1, keepdims=True)
        else:
            group_std = np.zeros_like(group_mean)
        grouped_advantages = (grouped_scores - group_mean) / (group_std + 1e-6)

        step_advantages = grouped_advantages.reshape(num_episodes, n_envs)
        step_advantages = np.broadcast_to(
            step_advantages[:, None, :], rewards_for_scores.shape
        ).copy()
        step_advantages *= loss_mask_for_scores.astype(np.float32)

        self._loss_filter_mask = None
        groups_kept = np.ones((num_episodes, num_groups), dtype=bool)
        filter_mask_fraction = 1.0
        if filter_rewards:
            group_mean = group_mean.squeeze(-1)
            groups_kept = (
                (group_mean >= rewards_lower_bound)
                & (group_mean <= rewards_upper_bound)
            )
            envs_kept = np.repeat(groups_kept, group_size, axis=1)
            loss_filter_mask = np.broadcast_to(
                envs_kept[:, None, :], loss_mask_for_scores.shape
            ).copy()
            if len(original_shape) == 3:
                self._loss_filter_mask = rearrange(
                    loss_filter_mask.reshape(primitive_T, n_envs),
                    "(t c) b -> t b c",
                    t=T,
                    b=n_envs,
                    c=chunk_size,
                )
            else:
                self._loss_filter_mask = loss_filter_mask.reshape(T, n_envs)
            filter_mask_fraction = float(loss_filter_mask.mean())

        self.returns = None
        if len(original_shape) == 3:
            self.advantages = rearrange(
                step_advantages.reshape(primitive_T, n_envs),
                "(t c) b -> t b c",
                t=T,
                b=n_envs,
                c=chunk_size,
            )
        else:
            self.advantages = step_advantages.reshape(T, n_envs)

        sample_keep_fraction = float(groups_kept.mean()) if groups_kept.size else 1.0
        return {
            "advantages": self.advantages,
            "grpo_score_mean": float(scores.mean()),
            "grpo_score_std": float(scores.std()),
            "grpo_group_keep_fraction": sample_keep_fraction,
        }

    def _compute_episode_loss_mask(
        self, episode_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        terminations = np.stack(self.terminations).astype(np.float32)  # (T, B[, C])
        original_shape = terminations.shape
        if terminations.ndim == 3:
            T, n_envs, chunk_size = terminations.shape
            terminations = rearrange(terminations, "t b c -> (t c) b")
        elif terminations.ndim == 2:
            T, n_envs = terminations.shape
            chunk_size = None
        else:
            raise ValueError(f"Unsupported terminations shape: {terminations.shape}")

        primitive_T = terminations.shape[0]
        assert primitive_T % episode_len == 0, (
            f"T ({primitive_T}) must be divisible by episode_len ({episode_len})"
        )
        num_episodes = primitive_T // episode_len
        term = terminations.reshape(num_episodes, episode_len, n_envs)
        cum = np.cumsum(term, axis=1)
        shifted = np.zeros_like(cum)
        shifted[:, 1:] = cum[:, :-1]
        mask = (shifted == 0)  # (num_episodes, episode_len, n_envs)

        valid_counts = mask.sum(axis=1, keepdims=True)
        ratio = (valid_counts / episode_len).astype(np.float32)
        ratio = np.broadcast_to(ratio, mask.shape).copy()
        if len(original_shape) == 3:
            mask = rearrange(
                mask.reshape(primitive_T, n_envs),
                "(t c) b -> t b c",
                t=T,
                b=n_envs,
                c=chunk_size,
            )
            ratio = rearrange(
                ratio.reshape(primitive_T, n_envs),
                "(t c) b -> t b c",
                t=T,
                b=n_envs,
                c=chunk_size,
            )
        else:
            mask = mask.reshape(primitive_T, n_envs)
            ratio = ratio.reshape(primitive_T, n_envs)
        return mask, ratio

    def compute_loss_mask(
        self, episode_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a boolean loss mask and per-step loss_mask_ratio.

        Returns:
            mask: [T*B] for chunk-level or [T*B, C] for action-level. True
                before the first termination per env.
            loss_mask_ratio: same shape as mask. Stores valid_steps / episode_len
                per (episode, env), broadcast to every supervised step.
        """
        mask, ratio = self._compute_episode_loss_mask(episode_len)
        if self._loss_filter_mask is not None:
            mask = mask & self._loss_filter_mask

        mask = rearrange(mask, "t b ... -> (t b) ...").astype(bool)
        ratio = rearrange(ratio, "t b ... -> (t b) ...").copy()
        return mask, ratio

    def norm_adv(self, mean: float, std: float) -> None:
        """Normalize advantages in-place with the given global mean and std."""
        self.advantages = (self.advantages - mean) / std

    def get_batch(
        self,
        compute_loss_masks: bool = False,
        episode_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stack and flatten rollout data into a single batch dict.

        Flattens [T, B, ...] into [T*B, ...]. Action-level per-chunk axes
        remain in the trailing dimensions, e.g. [T, B, C] -> [T*B, C].
        Obs/next_obs nested dicts are flattened to "obs/key", "next_obs/key".

        Returns:
            A dict with flat keys and flattened torch tensors.
        """
        def _stack_and_flatten(name: str, arrays: list[np.ndarray]) -> torch.Tensor:
            try:
                stacked = np.stack(arrays)               # (T, n_envs, ...)
            except ValueError as exc:
                shapes = [np.shape(array) for array in arrays]
                unique_shapes = list(dict.fromkeys(shapes))
                raise ValueError(
                    f"Cannot stack rollout field '{name}' because shapes differ over time: "
                    f"{unique_shapes}"
                ) from exc
            t = torch.from_numpy(stacked)
            return rearrange(t, "t b ... -> (t b) ...")

        batch: dict[str, Any] = {}

        # obs/next_obs → flat keys "obs/key", "next_obs/key"
        for prefix in ("obs", "next_obs"):
            data_list = getattr(self, prefix)
            if not data_list:
                continue
            for k in data_list[0].keys():
                vals = [d[k] for d in data_list]
                if isinstance(vals[0], np.ndarray):
                    batch[f"{prefix}/{k}"] = _stack_and_flatten(f"{prefix}/{k}", vals)

        for name in ("actions", "rewards", "terminations", "truncations",
                      "prev_logprobs", "prev_values"):
            data_list = getattr(self, name)
            if data_list:
                batch[name] = _stack_and_flatten(name, data_list)

        if self.forward_inputs:
            # forward_inputs may have T+1 entries (extra for last next_obs);
            # only include the first T for the batch
            fi_list = self.forward_inputs[:len(self.obs)] if len(self.forward_inputs) > len(self.obs) else self.forward_inputs
            keys = fi_list[0].keys()
            batch["forward_inputs"] = {
                k: _stack_and_flatten(f"forward_inputs/{k}", [d[k] for d in fi_list])
                for k in keys
            }

        if self.next_forward_inputs:
            keys = self.next_forward_inputs[0].keys()
            batch["next_forward_inputs"] = {
                k: _stack_and_flatten(
                    f"next_forward_inputs/{k}",
                    [d[k] for d in self.next_forward_inputs],
                )
                for k in keys
            }

        if compute_loss_masks:
            assert episode_len is not None, "episode_len is required when compute_loss_masks=True"
            mask, ratio = self.compute_loss_mask(episode_len=episode_len)
            batch["loss_mask"] = torch.from_numpy(mask.astype(np.float32))
            batch["loss_mask_ratio"] = torch.from_numpy(ratio)

        if self.returns is not None:
            batch["returns"] = torch.from_numpy(
                rearrange(self.returns, "t b ... -> (t b) ...")
            )
        if self.advantages is not None:
            batch["advantages"] = torch.from_numpy(
                rearrange(self.advantages, "t b ... -> (t b) ...")
            )

        return batch
