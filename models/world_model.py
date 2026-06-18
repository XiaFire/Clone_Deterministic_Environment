from typing import Dict, Optional

import torch
import torch.nn as nn

from dynamics_inputs import prepare_dynamics_inputs


class WorldModel(nn.Module):
    def __init__(
        self,
        dynamics: nn.Module,
        renderer: nn.Module,
        config,
        context_len: Optional[int] = None,
        sample_method: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.renderer = renderer
        self.config = config
        self.context_len = context_len or getattr(
            config, "n_context_tokens", getattr(config, "max_tokens", 5)
        )
        self.sample_method = sample_method or getattr(config, "sample_method", "3d")

    def rollout_pose(self, xs: torch.Tensor, actions: torch.Tensor, pre_k: int) -> torch.Tensor:
        bsz, seq_len = xs.shape[0], xs.shape[1]
        if pre_k <= 0 or pre_k >= seq_len:
            return xs

        predicted = torch.empty_like(xs)
        predicted[:, :pre_k, ...] = xs[:, :pre_k, ...]
        window = max(1, int(self.context_len))

        for i in range(pre_k, seq_len):
            start = max(0, i - window)
            cond = actions[:, start : i + 1, ...]
            ctx = predicted[:, start:i, ...]
            if self.sample_method == "3d":
                step_pred = self.dynamics.sample_3d(ctx, cond)
            elif self.sample_method == "2d":
                step_pred = self.dynamics.sample(ctx, cond)
            else:
                raise ValueError(f"Unsupported sample_method: {self.sample_method}")
            predicted[:, i, ...] = step_pred[:, -1, ...]

        return predicted

    def decode_obs(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.ndim == 5:
            poses = poses.squeeze(-1).squeeze(-1)
        return self.renderer.decode(poses)

    def forward(self, batch: Dict, pre_k: int = 5) -> Dict:
        inputs = prepare_dynamics_inputs(batch, self.config)
        xs = inputs["xs"]
        actions = inputs["conditions"]

        pred_pose = self.rollout_pose(xs, actions, pre_k=pre_k)
        pred_obs = self.decode_obs(pred_pose)

        output = {
            "pred_pose": pred_pose,
            "pred_obs": pred_obs,
        }
        output["gt_pose"] = xs
        if "image" in batch:
            output["gt_obs"] = batch["image"][:, 1:, ...]
        return output
