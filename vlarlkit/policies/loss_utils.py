from __future__ import annotations

import torch


def expand_to_shape(tensor: torch.Tensor | None, target_shape: torch.Size) -> torch.Tensor | None:
    if tensor is None:
        return None
    while tensor.dim() < len(target_shape):
        tensor = tensor.unsqueeze(-1)
    return tensor


def align_to_shape(
    tensor: torch.Tensor | None,
    target_shape: torch.Size,
    reduce: str = "sum",
) -> torch.Tensor | None:
    if tensor is None:
        return None
    while tensor.dim() > len(target_shape):
        if reduce == "max":
            tensor = tensor.max(dim=-1).values
        elif reduce == "mean":
            tensor = tensor.mean(dim=-1)
        else:
            tensor = tensor.sum(dim=-1)
    tensor = expand_to_shape(tensor, target_shape)
    return torch.broadcast_to(tensor, target_shape)


def reduce_logprobs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_mask_ratio: torch.Tensor,
    logprob_type: str,
    action_dim: int,
) -> dict[str, torch.Tensor]:
    batch_size = logprobs.shape[0]

    if logprob_type == "token_level":
        logprobs = logprobs.reshape(batch_size, -1, action_dim)
        old_logprobs = old_logprobs.reshape(batch_size, -1, action_dim)
    elif logprob_type == "action_level":
        logprobs = logprobs.reshape(batch_size, -1, action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(batch_size, -1, action_dim).sum(dim=-1)
    elif logprob_type == "chunk_level":
        logprobs = logprobs.reshape(batch_size, -1, action_dim).sum(dim=(1, 2))
        old_logprobs = old_logprobs.reshape(batch_size, -1, action_dim).sum(dim=(1, 2))
    else:
        raise ValueError(
            f"Unknown logprob_type: {logprob_type!r}. "
            "Expected 'token_level', 'action_level', or 'chunk_level'."
        )

    target_shape = logprobs.shape
    return {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": align_to_shape(advantages, target_shape),
        "loss_mask": align_to_shape(loss_mask, target_shape, reduce="max"),
        "loss_mask_ratio": align_to_shape(loss_mask_ratio, target_shape, reduce="mean"),
    }


def reduce_entropy(
    entropy: torch.Tensor | None,
    entropy_type: str,
    action_dim: int,
) -> torch.Tensor | None:
    if entropy is None:
        return None

    batch_size = entropy.shape[0]
    if entropy_type == "token_level":
        if entropy.dim() == 2 and entropy.shape[1] > 1 and entropy.shape[1] % action_dim == 0:
            entropy = entropy.reshape(batch_size, -1, action_dim)
    elif entropy_type == "action_level":
        if entropy.dim() == 3:
            entropy = entropy.sum(dim=-1)
        elif entropy.dim() == 2 and entropy.shape[1] % action_dim == 0:
            entropy = entropy.reshape(batch_size, -1, action_dim).sum(dim=-1)
    elif entropy_type == "chunk_level":
        if entropy.dim() > 1:
            entropy = entropy.reshape(batch_size, -1).sum(dim=-1)
    else:
        raise ValueError(
            f"Unknown entropy_type: {entropy_type!r}. "
            "Expected 'token_level', 'action_level', or 'chunk_level'."
        )
    return entropy


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    ratio: torch.Tensor | None = None,
    norm_loss_by_traj_len: bool = False,
) -> torch.Tensor:
    mask = expand_to_shape(mask, values.shape).to(dtype=values.dtype)
    mask = torch.broadcast_to(mask, values.shape)
    if norm_loss_by_traj_len:
        assert ratio is not None
        ratio = expand_to_shape(ratio, values.shape).to(dtype=values.dtype)
        ratio = torch.broadcast_to(ratio, values.shape)
        return (values / ratio.clamp(min=1e-6) * mask).mean()

    mask_sum = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / mask_sum


def masked_count(mask: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    mask = expand_to_shape(mask, target_shape).float()
    mask = torch.broadcast_to(mask, target_shape)
    return mask.sum().clamp(min=1.0)
