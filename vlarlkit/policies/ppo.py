import contextlib
import torch
import torch.distributed as dist

from omegaconf import DictConfig
from typing import Any

from vlarlkit.models.base import BaseModel
from vlarlkit.utils.conversion_utils import to_device
from vlarlkit.utils.fsdp_utils import clip_grad_norm_, wrap_model_with_fsdp
from vlarlkit.utils.logging import get_logger
from vlarlkit.policies.loss_utils import (
    align_to_shape,
    expand_to_shape,
    masked_count,
    masked_mean,
    reduce_entropy,
    reduce_logprobs,
)

logger = get_logger("vlarlkit.policy")


class PPOPolicy:
    """
    PPO policy with FSDP for single-machine multi-GPU training.
    """

    def __init__(self, cfg: DictConfig, model: BaseModel, rank: int) -> None:
        self.cfg = cfg
        self._algo_cfg = cfg.algorithm
        self._optim_cfg = cfg.training.optim
        self.rank = rank
        self.device = torch.device(f"cuda:{self.rank}")
        self.fsdp_cfg = getattr(cfg.training, "fsdp_config", None) or {}

        self.model = wrap_model_with_fsdp(model, self.fsdp_cfg, rank)

        self._setup_optimizer()
        self._setup_lr_scheduler()

        self._clip_grad = float(self._optim_cfg.get("clip_grad", 0.0))
        self._critic_warmup_steps = int(self._optim_cfg.get("critic_warmup_steps", 0))
        self._logprob_type = str(self._algo_cfg.get("logprob_type", "chunk_level"))
        self._entropy_type = str(self._algo_cfg.get("entropy_type", "chunk_level"))
        self._action_dim = int(cfg.model.action_dim)
        self._global_step = 0

    def _setup_optimizer(self) -> None:
        value_lr = self._optim_cfg.get("value_lr")
        lr = float(self._optim_cfg.get("lr"))
        beta1 = float(self._optim_cfg.get("adam_beta1", 0.9))
        beta2 = float(self._optim_cfg.get("adam_beta2", 0.999))
        eps = float(self._optim_cfg.get("adam_eps", 1e-8))
        weight_decay = float(self._optim_cfg.get("weight_decay", 0.0))

        if value_lr is not None:
            value_lr = float(value_lr)
            actor_params = []
            value_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "value_head" in name:
                    value_params.append(param)
                else:
                    actor_params.append(param)

            if value_params:
                param_groups = [
                    {"params": actor_params, "lr": lr},
                    {"params": value_params, "lr": value_lr},
                ]
                self._optimizer = torch.optim.AdamW(
                    param_groups, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
                )
                if self.rank == 0:
                    n_actor = sum(p.numel() for p in actor_params)
                    n_value = sum(p.numel() for p in value_params)
                    logger.info(
                        f"Optimizer: actor params={n_actor:,} (lr={lr}), "
                        f"value params={n_value:,} (lr={value_lr})"
                    )
            else:
                if self.rank == 0:
                    logger.warning("value_lr set but no 'value_head' params found, using single lr")
                params = [p for p in self.model.parameters() if p.requires_grad]
                self._optimizer = torch.optim.AdamW(
                    params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay,
                )
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.AdamW(
                params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay,
            )

        n_optim = sum(p.numel() for group in self._optimizer.param_groups for p in group["params"])
        if self.rank == 0:
            logger.info(f"Optimizable params: {n_optim:,}")

    def _setup_lr_scheduler(self) -> None:
        sched_type = self._optim_cfg.get("lr_scheduler", "constant")
        total_steps = int(self._optim_cfg.get("total_training_steps", 1000))
        min_lr_rate = float(self._optim_cfg.get("min_lr_rate", 0.1))
        if sched_type == "cosine":
            def lr_lambda(step: int) -> float:
                if step >= total_steps:
                    return min_lr_rate
                import math
                return min_lr_rate + 0.5 * (1 - min_lr_rate) * (1 + math.cos(step / total_steps * math.pi))
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
        return self.model

    def set_global_step(self, step: int) -> None:
        self._global_step = step
        inner = self.model.module if hasattr(self.model, "module") else self.model
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
        self.model.train()

        update_epochs = int(self._algo_cfg.get("update_epochs"))
        # Derive per-rank mini-batch size and gradient accumulation
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_mini_bs = self._algo_cfg.get("global_mini_batch_size")
        micro_bs = self._algo_cfg.get("micro_batch_size")
        micro_batch_size = int(micro_bs)
        gradient_accumulation_steps = int(global_mini_bs) // (micro_batch_size * world_size)
        norm_loss_by_traj_len = self._algo_cfg.get("norm_loss_by_traj_len", False)

        clip_ratio_high = float(self._algo_cfg.get("clip_ratio_high", 0.2))
        clip_ratio_low = float(self._algo_cfg.get("clip_ratio_low", 0.2))
        clip_ratio_c = float(self._algo_cfg.get("clip_ratio_c", 0.0))
        value_clip = float(self._algo_cfg.get("value_clip", 0.2))
        huber_delta = float(self._algo_cfg.get("huber_delta", 10.0))
        entropy_bonus = float(self._algo_cfg.get("entropy_bonus", 0.0))
        critic_warmup = self._global_step < self._critic_warmup_steps

        advantages = batch["advantages"]
        prev_logprobs = batch["prev_logprobs"]
        prev_values = batch["prev_values"]
        returns = batch["returns"]
        forward_inputs = batch["forward_inputs"]
        loss_mask = batch["loss_mask"]
        loss_mask_ratio = batch["loss_mask_ratio"]

        N = advantages.shape[0]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_value_mean = 0.0
        total_ratio = 0.0
        total_clip_fraction = 0.0
        total_value_clip_fraction = 0.0
        num_minibatches = 0

        for _ in range(update_epochs):
            perm = torch.randperm(N)
            self._optimizer.zero_grad()
            accum_step = 0

            for start in range(0, N, micro_batch_size):
                end = min(start + micro_batch_size, N)
                mb_inds = perm[start:end]

                mb_advantages = advantages[mb_inds].to(self.device)
                mb_prev_logprobs = prev_logprobs[mb_inds].to(self.device)
                mb_prev_values = prev_values[mb_inds].to(self.device)
                mb_returns = returns[mb_inds].to(self.device)
                mb_forward_inputs = to_device(
                    self._slice_batch(forward_inputs, mb_inds), self.device
                )
                mb_raw_mask = loss_mask[mb_inds].to(self.device)
                mb_raw_ratio = loss_mask_ratio[mb_inds].to(self.device)

                out = self.model(
                    forward_inputs=mb_forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=True,
                    compute_values=True,
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
                values = align_to_shape(out["values"], mb_returns.shape)
                mb_prev_values = align_to_shape(mb_prev_values, mb_returns.shape)
                mb_value_mask = align_to_shape(
                    mb_raw_mask,
                    mb_returns.shape,
                    reduce="max",
                )
                mb_value_ratio = align_to_shape(
                    mb_raw_ratio,
                    mb_returns.shape,
                    reduce="mean",
                )
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
                    per_sample_policy_loss = torch.min(
                        per_sample_policy_loss, dual_clip_bound
                    )

                def _value_loss(residual: torch.Tensor) -> torch.Tensor:
                    if huber_delta > 0:
                        return torch.where(
                            torch.abs(residual) <= huber_delta,
                            0.5 * residual**2,
                            huber_delta * (torch.abs(residual) - 0.5 * huber_delta),
                        )
                    return 0.5 * (residual**2)

                if value_clip > 0:
                    values_clipped = mb_prev_values + torch.clamp(
                        values - mb_prev_values,
                        -value_clip,
                        value_clip,
                    )
                    per_sample_value_loss = torch.max(
                        _value_loss(values - mb_returns),
                        _value_loss(values_clipped - mb_returns),
                    )
                    value_clip_mask = ((values - mb_prev_values).abs() > value_clip).detach()
                else:
                    per_sample_value_loss = _value_loss(values - mb_returns)
                    value_clip_mask = torch.zeros_like(values, dtype=torch.bool)

                entropy_mask = align_to_shape(mb_raw_mask, entropy.shape, reduce="max")
                entropy_ratio = align_to_shape(mb_raw_ratio, entropy.shape, reduce="mean")
                policy_loss = masked_mean(
                    per_sample_policy_loss,
                    mb_policy_mask,
                    mb_policy_ratio,
                    norm_loss_by_traj_len=norm_loss_by_traj_len,
                )
                value_loss = masked_mean(
                    per_sample_value_loss,
                    mb_value_mask,
                    mb_value_ratio,
                    norm_loss_by_traj_len=norm_loss_by_traj_len,
                )
                entropy_val = masked_mean(
                    entropy,
                    entropy_mask,
                    entropy_ratio,
                    norm_loss_by_traj_len=norm_loss_by_traj_len,
                )
                value_mean = masked_mean(
                    values.detach(),
                    mb_value_mask,
                    mb_value_ratio,
                    norm_loss_by_traj_len=norm_loss_by_traj_len,
                )

                if critic_warmup:
                    loss = value_loss
                else:
                    loss = policy_loss + value_loss
                    if entropy_bonus != 0:
                        loss = loss - entropy_bonus * entropy_val

                accum_step += 1
                should_sync = (accum_step == gradient_accumulation_steps) or (end >= N)
                sync_context = contextlib.nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    (loss / gradient_accumulation_steps).backward()

                if should_sync:
                    if self._clip_grad > 0:
                        clip_grad_norm_(self.model, self._clip_grad)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    accum_step = 0

                total_policy_loss += policy_loss.detach().item()
                total_value_loss += value_loss.detach().item()
                total_entropy += entropy_val.detach().item()
                total_value_mean += value_mean.item()
                ratio_count = masked_count(mb_policy_mask, ratio.shape)
                value_count = masked_count(mb_value_mask, value_clip_mask.shape)
                ratio_mask = torch.broadcast_to(
                    expand_to_shape(mb_policy_mask, ratio.shape), ratio.shape
                )
                value_mask = torch.broadcast_to(
                    expand_to_shape(mb_value_mask, value_clip_mask.shape),
                    value_clip_mask.shape,
                )
                total_ratio += (ratio.detach() * ratio_mask).sum().item() / ratio_count.item()
                total_clip_fraction += (
                    clip_mask.float() * ratio_mask
                ).sum().item() / ratio_count.item()
                total_value_clip_fraction += (
                    value_clip_mask.float() * value_mask
                ).sum().item() / value_count.item()
                num_minibatches += 1

        if critic_warmup and self.rank == 0:
            logger.info("Critic warmup step %d/%d (policy loss zeroed)", self._global_step + 1, self._critic_warmup_steps)

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self._global_step += 1
        self.set_global_step(self._global_step)

        n = max(1, num_minibatches)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "value_mean": total_value_mean / n,
            "ratio": total_ratio / n,
            "clip_fraction": total_clip_fraction / n,
            "value_clip_fraction": total_value_clip_fraction / n,
        }
