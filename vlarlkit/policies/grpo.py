import contextlib
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from vlarlkit.models.base import BaseModel
from vlarlkit.utils.conversion_utils import to_device
from vlarlkit.utils.fsdp_utils import clip_grad_norm_, wrap_model_with_fsdp
from vlarlkit.utils.logging import get_logger
from vlarlkit.policies.loss_utils import (
    align_to_shape,
    expand_to_shape,
    masked_mean,
    reduce_entropy,
    reduce_logprobs,
)

logger = get_logger("vlarlkit.policy")


class GRPOPolicy:
    """
    Critic-free GRPO policy with FSDP for single-machine multi-GPU training.
    """

    def __init__(self, cfg: DictConfig, model: BaseModel, rank: int) -> None:
        self._cfg = cfg
        self._algo_cfg = cfg.algorithm
        self._optim_cfg = cfg.training.optim
        self._rank = rank
        self._device = torch.device(f"cuda:{self._rank}")
        self._fsdp_cfg = getattr(cfg.training, "fsdp_config", None) or {}

        self._model = wrap_model_with_fsdp(model, self._fsdp_cfg, rank)

        self._setup_optimizer()
        self._setup_lr_scheduler()

        self._clip_grad = float(self._optim_cfg.get("clip_grad", 0.0))
        self._logprob_type = str(self._algo_cfg.get("logprob_type", "chunk_level"))
        self._entropy_type = str(self._algo_cfg.get("entropy_type", "chunk_level"))
        self._action_dim = int(cfg.model.action_dim)
        self._global_step = 0

    def _setup_optimizer(self) -> None:
        lr = float(self._optim_cfg.get("lr"))
        beta1 = float(self._optim_cfg.get("adam_beta1", 0.9))
        beta2 = float(self._optim_cfg.get("adam_beta2", 0.999))
        eps = float(self._optim_cfg.get("adam_eps", 1e-8))
        weight_decay = float(self._optim_cfg.get("weight_decay", 0.0))

        params = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

        n_optim = sum(p.numel() for p in params)
        if self._rank == 0:
            logger.info(f"GRPO optimizable params: {n_optim:,}")

    def _setup_lr_scheduler(self) -> None:
        sched_type = self._optim_cfg.get("lr_scheduler", "constant")
        total_steps = int(self._optim_cfg.get("total_training_steps", 1000))
        min_lr_rate = float(self._optim_cfg.get("min_lr_rate", 0.1))
        if sched_type == "cosine":

            def lr_lambda(step: int) -> float:
                if step >= total_steps:
                    return min_lr_rate
                import math

                return min_lr_rate + 0.5 * (1 - min_lr_rate) * (
                    1 + math.cos(step / total_steps * math.pi)
                )

            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda
            )
        elif sched_type == "constant":
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lambda step: 1.0
            )
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    def state_dict(self) -> dict:
        return {
            "global_step": self._global_step,
            "lr_scheduler": self._lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._global_step = state["global_step"]
        self.set_global_step(self._global_step)
        self._lr_scheduler.load_state_dict(state["lr_scheduler"])

    def get_model(self) -> torch.nn.Module:
        return self._model

    def set_global_step(self, step: int) -> None:
        self._global_step = step
        inner = self._model.module if hasattr(self._model, "module") else self._model
        if hasattr(inner, "set_global_step"):
            inner.set_global_step(step)

    def _slice_batch(self, batch: dict, indices: torch.Tensor) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                out[k] = {
                    kk: vv[indices] if torch.is_tensor(vv) else vv
                    for kk, vv in v.items()
                }
            elif torch.is_tensor(v):
                out[k] = v[indices]
            else:
                out[k] = v
        return out

    def _sampling_kwargs(self) -> dict[str, Any]:
        params = self._algo_cfg.get("sampling_params", {})
        kwargs = {key: params[key] for key in params} if params else {}
        temperature = kwargs.pop("temperature_train", None)
        kwargs.pop("temperature_eval", None)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        return kwargs

    def run_update(self, batch: dict[str, Any]) -> dict[str, float]:
        self._model.train()

        update_epochs = int(self._algo_cfg.get("update_epochs", 1))
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_mini_bs = self._algo_cfg.get("global_mini_batch_size")
        micro_batch_size = int(self._algo_cfg.get("micro_batch_size"))
        gradient_accumulation_steps = int(global_mini_bs) // (
            micro_batch_size * world_size
        )
        norm_loss_by_traj_len = self._algo_cfg.get("norm_loss_by_traj_len", False)

        clip_ratio_high = float(self._algo_cfg.get("clip_ratio_high", 0.2))
        clip_ratio_low = float(self._algo_cfg.get("clip_ratio_low", 0.2))
        clip_ratio_c = float(self._algo_cfg.get("clip_ratio_c", 0.0))
        entropy_bonus = float(self._algo_cfg.get("entropy_bonus", 0.0))

        advantages = batch["advantages"]
        prev_logprobs = batch["prev_logprobs"]
        forward_inputs = batch["forward_inputs"]
        loss_mask = batch["loss_mask"]
        loss_mask_ratio = batch["loss_mask_ratio"]

        n_samples = advantages.shape[0]

        total_policy_loss = 0.0
        total_entropy = 0.0
        total_ratio = 0.0
        total_clip_fraction = 0.0
        total_dual_clip_fraction = 0.0
        num_minibatches = 0

        for _ in range(update_epochs):
            perm = torch.randperm(n_samples)
            self._optimizer.zero_grad()
            accum_step = 0
            accum_valid_count = 0.0

            for start in range(0, n_samples, micro_batch_size):
                end = min(start + micro_batch_size, n_samples)
                mb_inds = perm[start:end]

                mb_advantages = advantages[mb_inds].to(self._device)
                mb_prev_logprobs = prev_logprobs[mb_inds].to(self._device)
                mb_forward_inputs = to_device(
                    self._slice_batch(forward_inputs, mb_inds), self._device
                )
                mb_raw_mask = loss_mask[mb_inds].to(self._device)
                mb_raw_ratio = loss_mask_ratio[mb_inds].to(self._device)

                out = self._model(
                    forward_inputs=mb_forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=True,
                    compute_values=False,
                    **self._sampling_kwargs(),
                )
                loss_inputs = reduce_logprobs(
                    logprobs=out["logprobs"],
                    old_logprobs=mb_prev_logprobs,
                    advantages=mb_advantages,
                    loss_mask=mb_raw_mask,
                    loss_mask_ratio=mb_raw_ratio,
                    logprob_type=self._logprob_type,
                    action_dim=self._action_dim,
                )
                logprobs = loss_inputs["logprobs"]
                mb_prev_logprobs = loss_inputs["old_logprobs"]
                mb_advantages = loss_inputs["advantages"]
                mb_policy_mask = loss_inputs["loss_mask"]
                mb_policy_ratio = loss_inputs["loss_mask_ratio"]
                ratio_mask = torch.broadcast_to(
                    expand_to_shape(mb_policy_mask, logprobs.shape), logprobs.shape
                )
                mb_valid_count = ratio_mask.sum().item()
                accum_valid_count += mb_valid_count
                entropy = reduce_entropy(
                    out.get("entropy"),
                    entropy_type=self._entropy_type,
                    action_dim=self._action_dim,
                )
                if entropy is None:
                    entropy = torch.zeros_like(logprobs)

                ratio = torch.exp(logprobs - mb_prev_logprobs)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
                )
                per_sample_policy_loss = torch.max(policy_loss1, policy_loss2)
                clip_mask = (policy_loss1 < policy_loss2).detach()

                if clip_ratio_c > 1.0:
                    dual_clip_bound = clip_ratio_c * torch.abs(mb_advantages)
                    dual_clip_mask = (
                        dual_clip_bound.detach() < per_sample_policy_loss.detach()
                    )
                    per_sample_policy_loss = torch.min(
                        per_sample_policy_loss, dual_clip_bound
                    )
                else:
                    dual_clip_mask = torch.zeros_like(clip_mask)

                entropy_mask = align_to_shape(mb_raw_mask, entropy.shape, reduce="max")
                entropy_ratio = align_to_shape(mb_raw_ratio, entropy.shape, reduce="mean")
                policy_loss = masked_mean(
                    per_sample_policy_loss,
                    mb_policy_mask,
                    mb_policy_ratio,
                    norm_loss_by_traj_len=norm_loss_by_traj_len,
                )
                entropy_val = masked_mean(
                    entropy,
                    entropy_mask,
                    entropy_ratio,
                    norm_loss_by_traj_len=norm_loss_by_traj_len,
                )

                loss = policy_loss
                if entropy_bonus != 0:
                    loss = loss - entropy_bonus * entropy_val

                accum_step += 1
                should_sync = (
                    accum_step == gradient_accumulation_steps
                ) or (end >= n_samples)
                sync_context = (
                    contextlib.nullcontext() if should_sync else self._model.no_sync()
                )
                with sync_context:
                    (loss / gradient_accumulation_steps).backward()

                if should_sync:
                    valid_count = torch.tensor(
                        accum_valid_count,
                        device=self._device,
                    )
                    if dist.is_initialized():
                        dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)

                    if valid_count.item() > 0:
                        if self._clip_grad > 0:
                            clip_grad_norm_(self._model, self._clip_grad)
                        self._optimizer.step()
                    self._optimizer.zero_grad()
                    accum_step = 0
                    accum_valid_count = 0.0

                if mb_valid_count > 0:
                    total_policy_loss += policy_loss.detach().item()
                    total_entropy += entropy_val.detach().item()

                    total_ratio += (
                        (ratio.detach() * ratio_mask).sum().item() / mb_valid_count
                    )
                    total_clip_fraction += (
                        (clip_mask.float() * ratio_mask).sum().item() / mb_valid_count
                    )
                    total_dual_clip_fraction += (
                        (dual_clip_mask.float() * ratio_mask).sum().item()
                        / mb_valid_count
                    )
                    num_minibatches += 1

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self._global_step += 1
        self.set_global_step(self._global_step)

        n = max(1, num_minibatches)
        return {
            "policy_loss": total_policy_loss / n,
            "entropy": total_entropy / n,
            "ratio": total_ratio / n,
            "clip_fraction": total_clip_fraction / n,
            "dual_clip_fraction": total_dual_clip_fraction / n,
        }
