from typing import Dict, Optional, Tuple

import torch


def reshape_pose(xs: torch.Tensor, x_shape: Optional[Tuple[int, ...]]) -> torch.Tensor:
    if xs.ndim != 3:
        return xs
    if x_shape and len(x_shape) == 3:
        _, height, width = x_shape
        if height == 1 and width == 1:
            return xs.unsqueeze(-1).unsqueeze(-1)
    return xs


def prepare_dynamics_inputs(batch: Dict, config) -> Dict:
    xs = batch["pose"]
    x_shape = getattr(config, "x_shape", None)
    if x_shape is not None:
        xs = reshape_pose(xs, tuple(x_shape))
    xs = xs[:, 1:, ...]

    actions = batch["action"]
    if actions.ndim == 2:
        actions = actions.unsqueeze(-1)
    actions = actions[:, :-1, :]
    if actions.shape[1] != 0:
        actions[:, 0, :] = -100

    padding_mask = batch.get("padding_mask", None)
    if padding_mask is not None:
        padding_mask = padding_mask[:, 1:]
        if padding_mask.shape[1] != 0:
            padding_mask[:, 0] = 0

    actions = actions.clone()
    actions[actions == -100] = 0

    return {
        "xs": xs,
        "conditions": actions.long().squeeze(-1),
        "masks": padding_mask,
    }
