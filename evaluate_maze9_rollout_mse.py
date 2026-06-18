import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from dfot.dfot import DFOT
from generate_latent_videos import load_autoencoder_model, reshape_latents
from generate_pose_render_videos import load_render_model


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate maze9 rollout pixel frame-wise MSE from rollout tensors."
    )
    parser.add_argument("--mode", choices=["latent", "pose"], required=True)
    parser.add_argument("--name", required=True, help="Experiment name used in output files.")
    parser.add_argument("--dyna_config", required=True)
    parser.add_argument("--dyna_ckpt", required=True)
    parser.add_argument("--metadata_dir", required=True, help="Precomputed val window directory used for traj/window ids.")
    parser.add_argument("--maze_dir", required=True, help="Raw maze validation directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vae_ckpt", default=None)
    parser.add_argument("--vae_cls", default="grvae")
    parser.add_argument("--render_ckpt", default=None)
    parser.add_argument("--render_cls", default="render")
    parser.add_argument("--window_glob", default="traj*_win0000.pt")
    parser.add_argument("--num_samples", type=int, default=1000, help="-1 means all matching windows.")
    parser.add_argument("--sample_start", type=int, default=0, help="Start offset for strided sample partitioning.")
    parser.add_argument("--sample_stride", type=int, default=1, help="Stride for strided sample partitioning.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", type=str2bool, default=False)
    parser.add_argument("--pre_k", type=int, default=5)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--sample_method", choices=["2d", "3d"], default=None)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--scale", type=float, nargs=2, default=[4.5, 4.5])
    parser.add_argument("--pose_center", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if args.gpu is not None:
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dynamics_model(config_path: str, ckpt_path: str, overrides: Sequence[str], device: torch.device):
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    model = DFOT(config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def select_window_paths(args) -> List[Path]:
    if args.sample_stride < 1:
        raise ValueError("--sample_stride must be >= 1")
    if args.sample_start < 0 or args.sample_start >= args.sample_stride:
        raise ValueError("--sample_start must be in [0, sample_stride)")
    paths = sorted(Path(args.metadata_dir).glob(args.window_glob))
    if not paths:
        raise FileNotFoundError(f"No windows matched {args.window_glob!r} under {args.metadata_dir}")
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(paths)
    if args.num_samples >= 0:
        paths = paths[: args.num_samples]
    if args.sample_stride != 1 or args.sample_start != 0:
        paths = paths[args.sample_start :: args.sample_stride]
    return paths


def process_maze_images(img_selected: np.ndarray, size: int) -> torch.Tensor:
    img_processed = img_selected.astype(np.float32) / 127.5 - 1.0
    if img_processed.shape[-2:] != (size, size):
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_processed))
        img_processed = F.interpolate(
            img_tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).numpy()
    return torch.tensor(img_processed, dtype=torch.float32)


class MazeWindowDataset(Dataset):
    def __init__(
        self,
        window_paths: Sequence[Path],
        maze_dir: str,
        mode: str,
        size: int,
        scale: Sequence[float],
        pose_center: Sequence[float],
    ):
        self.window_paths = list(window_paths)
        self.maze_dir = Path(maze_dir)
        self.mode = mode
        self.size = size
        self.scale_x = float(scale[0])
        self.scale_y = float(scale[1])
        self.pose_center_x = float(pose_center[0])
        self.pose_center_y = float(pose_center[1])
        self.h5_file = h5py.File(self.maze_dir / "info.hdf5", "r")
        self.trajectory_indices = self.h5_file["trajectory_indices"][:]
        self.steps = self.h5_file["steps"]

    def __len__(self):
        return len(self.window_paths)

    def _load_metadata(self, idx: int) -> Tuple[Path, Dict, int, int, int]:
        path = self.window_paths[idx]
        data = torch.load(path, map_location="cpu")
        traj_idx = int(data["traj_idx"].item())
        start_idx = int(data["start_idx"].item())
        end_idx = int(data["end_idx"].item())
        return path, data, traj_idx, start_idx, end_idx

    def _load_pose(self, traj_idx: int, start_idx: int, end_idx: int) -> torch.Tensor:
        global_start = int(self.trajectory_indices[traj_idx]) + start_idx
        global_end = int(self.trajectory_indices[traj_idx]) + end_idx
        info_np = np.array(self.steps[global_start:global_end], dtype=np.float32)
        pose_raw = info_np[:, :3]
        pose = np.zeros((pose_raw.shape[0], 4), dtype=np.float32)
        if pose_raw.shape[0] > 0:
            theta_rad = pose_raw[:, 2] * np.pi / 180.0
            pose[:, 0] = pose_raw[:, 0] / self.scale_x - self.pose_center_x
            pose[:, 1] = pose_raw[:, 1] / self.scale_y - self.pose_center_y
            pose[:, 2] = np.cos(theta_rad)
            pose[:, 3] = np.sin(theta_rad)
        return torch.tensor(pose, dtype=torch.float32)

    def _load_images(self, traj_idx: int, start_idx: int, end_idx: int) -> torch.Tensor:
        image_path = self.maze_dir / "obs" / f"{traj_idx}.npy"
        img_raw = np.load(image_path, mmap_mode="r")
        return process_maze_images(img_raw[start_idx:end_idx], self.size)

    def __getitem__(self, idx: int) -> Dict:
        path, data, traj_idx, start_idx, end_idx = self._load_metadata(idx)
        item = {
            "action": data["action"].float(),
            "padding_mask": data["padding_mask"].bool(),
            "traj_idx": torch.tensor(traj_idx, dtype=torch.long),
            "start_idx": torch.tensor(start_idx, dtype=torch.long),
            "end_idx": torch.tensor(end_idx, dtype=torch.long),
            "dataset_index": torch.tensor(idx, dtype=torch.long),
            "window_path": str(path),
            "image": self._load_images(traj_idx, start_idx, end_idx),
        }
        if self.mode == "latent":
            item["latents"] = data["latents"].float()
        else:
            item["pose"] = self._load_pose(traj_idx, start_idx, end_idx)
        return item

    def close(self):
        self.h5_file.close()


def collate_windows(batch: List[Dict]) -> Dict:
    tensor_batch = {}
    sequence_lengths = [
        value.shape[0]
        for item in batch
        for key, value in item.items()
        if torch.is_tensor(value) and value.dim() > 0 and key not in {"traj_idx", "start_idx", "end_idx", "dataset_index"}
    ]
    max_len = max(sequence_lengths)

    for key in batch[0]:
        if key == "window_path":
            tensor_batch[key] = [item[key] for item in batch]
            continue
        if not all(key in item for item in batch):
            continue
        value = batch[0][key]
        if not torch.is_tensor(value):
            tensor_batch[key] = [item[key] for item in batch]
            continue
        if value.dim() == 0:
            tensor_batch[key] = torch.stack([item[key] for item in batch])
            continue

        padded = []
        for item in batch:
            tensor = item[key]
            pad_len = max_len - tensor.shape[0]
            if pad_len > 0:
                pad_shape = list(tensor.shape)
                pad_shape[0] = pad_len
                pad_value = False if tensor.dtype == torch.bool else 0
                pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            padded.append(tensor)
        tensor_batch[key] = torch.stack(padded)
    return tensor_batch


def prepare_latent_inputs(batch: Dict, config, use_action: bool) -> Dict:
    x_shape = tuple(int(dim) for dim in config.x_shape)
    xs = reshape_latents(batch["latents"], x_shape)[:, 1:, ...]
    actions = batch["action"]
    if actions.ndim == 2:
        actions = actions.unsqueeze(-1)
    actions = actions[:, :-1, :].clone()
    if actions.shape[1] != 0:
        actions[:, 0, :] = -100
    actions[actions == -100] = 0
    masks = batch["padding_mask"][:, 1:].clone().bool()
    if masks.shape[1] != 0:
        masks[:, 0] = False
    if not use_action:
        actions = torch.zeros_like(actions)
    return {"xs": xs, "conditions": actions.long().squeeze(-1), "masks": masks}


def prepare_pose_inputs(batch: Dict, config, use_action: bool) -> Dict:
    xs = batch["pose"][:, 1:, ...]
    actions = batch["action"]
    if actions.ndim == 2:
        actions = actions.unsqueeze(-1)
    actions = actions[:, :-1, :].clone()
    if actions.shape[1] != 0:
        actions[:, 0, :] = -100
    actions[actions == -100] = 0
    masks = batch["padding_mask"][:, 1:].clone().bool()
    if masks.shape[1] != 0:
        masks[:, 0] = False
    if not use_action:
        actions = torch.zeros_like(actions)
    return {"xs": xs, "conditions": actions.long().squeeze(-1), "masks": masks}


@torch.no_grad()
def rollout_trajectory(
    model: torch.nn.Module,
    xs: torch.Tensor,
    actions: torch.Tensor,
    pre_k: int,
    context_len: Optional[int],
    sample_method: Optional[str],
    config,
) -> torch.Tensor:
    predicted = torch.empty_like(xs)
    start_step = min(max(pre_k, 0), xs.shape[1])
    if start_step > 0:
        predicted[:, :start_step, ...] = xs[:, :start_step, ...]
    window = max(
        1,
        int(context_len or getattr(config, "context_len", getattr(config, "max_tokens", 5))),
    )
    mode = sample_method or getattr(config, "sample_method", "3d" if xs.ndim == 5 else "2d")
    for step in range(start_step, xs.shape[1]):
        start = max(0, step - window)
        ctx = predicted[:, start:step, ...]
        cond = actions[:, start : step + 1, ...]
        if mode == "3d":
            step_pred = model.sample_3d(ctx, cond)
        elif mode == "2d":
            step_pred = model.sample(ctx, cond)
        else:
            raise ValueError(f"Unsupported sample_method={mode!r}")
        predicted[:, step, ...] = step_pred[:, -1, ...]
    return predicted


def batch_frame_mse(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = pred.float().clamp(-1.0, 1.0)
    target = target.float().clamp(-1.0, 1.0)
    mse_norm = (pred - target).pow(2).flatten(2).mean(dim=2)
    pred_255 = (pred + 1.0) * 127.5
    target_255 = (target + 1.0) * 127.5
    mse_255 = (pred_255 - target_255).pow(2).flatten(2).mean(dim=2)
    return mse_norm.detach().cpu(), mse_255.detach().cpu()


def append_masked_frames(
    store: List[np.ndarray],
    values: torch.Tensor,
    mask: torch.Tensor,
    pre_k: int,
):
    values_np = values.numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    steps = np.arange(values_np.shape[1])[None, :]
    mask_np = mask_np & (steps >= pre_k)
    for row, row_mask in zip(values_np, mask_np):
        out = np.full(row.shape, np.nan, dtype=np.float32)
        out[row_mask] = row[row_mask]
        store.append(out)


def summarize_metric(name: str, arrays: List[np.ndarray]) -> Dict:
    padded = np.stack(arrays, axis=0).astype(np.float32)
    valid = ~np.isnan(padded)
    per_step_count = valid.sum(axis=0)
    per_step_mean = np.full(padded.shape[1], np.nan, dtype=np.float32)
    per_step_std = np.full(padded.shape[1], np.nan, dtype=np.float32)
    for step, count in enumerate(per_step_count):
        if count <= 0:
            continue
        values = padded[valid[:, step], step]
        per_step_mean[step] = float(np.mean(values))
        per_step_std[step] = float(np.std(values))
    valid_values = padded[~np.isnan(padded)]

    def finite_or_none(values: Iterable[float]) -> List[Optional[float]]:
        return [None if np.isnan(value) else float(value) for value in values]

    return {
        f"{name}_global_mean": float(np.mean(valid_values)),
        f"{name}_global_std": float(np.std(valid_values)),
        f"{name}_global_min": float(np.min(valid_values)),
        f"{name}_global_max": float(np.max(valid_values)),
        f"{name}_per_step_mean": finite_or_none(per_step_mean),
        f"{name}_per_step_std": finite_or_none(per_step_std),
        f"{name}_per_step_count": per_step_count.astype(int).tolist(),
    }


def write_results(
    args,
    metadata_rows: List[Dict],
    mse_norm_rows: List[np.ndarray],
    mse_255_rows: List[np.ndarray],
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mse_norm = np.stack(mse_norm_rows, axis=0).astype(np.float32)
    mse_255 = np.stack(mse_255_rows, axis=0).astype(np.float32)

    sample_mean_norm = np.nanmean(mse_norm, axis=1)
    sample_mean_255 = np.nanmean(mse_255, axis=1)
    for row, mean_norm, mean_255 in zip(metadata_rows, sample_mean_norm, sample_mean_255):
        row["mse_norm_mean"] = float(mean_norm)
        row["mse_255_mean"] = float(mean_255)

    summary = {
        "name": args.name,
        "mode": args.mode,
        "num_samples": len(metadata_rows),
        "pre_k": args.pre_k,
        "context_len": args.context_len,
        "sample_method": args.sample_method,
        "mse_shape": list(mse_norm.shape),
        "args": vars(args),
        **summarize_metric("mse_norm", mse_norm_rows),
        **summarize_metric("mse_255", mse_255_rows),
    }

    json_path = output_dir / f"{args.name}_rollout_pixel_mse_summary.json"
    samples_path = output_dir / f"{args.name}_rollout_pixel_mse_samples.json"
    npz_path = output_dir / f"{args.name}_rollout_pixel_mse.npz"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    samples_path.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        mse_norm=mse_norm,
        mse_255=mse_255,
        sample_mean_norm=sample_mean_norm.astype(np.float32),
        sample_mean_255=sample_mean_255.astype(np.float32),
        traj_idx=np.array([row["traj_idx"] for row in metadata_rows], dtype=np.int64),
        start_idx=np.array([row["start_idx"] for row in metadata_rows], dtype=np.int64),
        end_idx=np.array([row["end_idx"] for row in metadata_rows], dtype=np.int64),
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {samples_path}", flush=True)
    print(f"Saved {npz_path}", flush=True)


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args)

    window_paths = select_window_paths(args)
    dataset = MazeWindowDataset(
        window_paths=window_paths,
        maze_dir=args.maze_dir,
        mode=args.mode,
        size=args.size,
        scale=args.scale,
        pose_center=args.pose_center,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
    )

    dyna_model, dyna_config = load_dynamics_model(
        args.dyna_config,
        args.dyna_ckpt,
        args.overrides,
        device,
    )
    use_action = bool(getattr(dyna_config, "action", True))
    decoder = None
    if args.mode == "latent":
        if not args.vae_ckpt:
            raise ValueError("--vae_ckpt is required for --mode latent")
        decoder = load_autoencoder_model(args.vae_ckpt, args.vae_cls, device)
    else:
        if not args.render_ckpt:
            raise ValueError("--render_ckpt is required for --mode pose")
        decoder = load_render_model(args.render_ckpt, args.render_cls, device)

    metadata_rows: List[Dict] = []
    mse_norm_rows: List[np.ndarray] = []
    mse_255_rows: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=args.name):
            batch = to_device(batch, device)
            if args.mode == "latent":
                inputs = prepare_latent_inputs(batch, dyna_config, use_action)
            else:
                inputs = prepare_pose_inputs(batch, dyna_config, use_action)
            pred_states = rollout_trajectory(
                model=dyna_model,
                xs=inputs["xs"],
                actions=inputs["conditions"],
                pre_k=args.pre_k,
                context_len=args.context_len,
                sample_method=args.sample_method,
                config=dyna_config,
            )
            pred_obs = decoder.decode(pred_states)
            gt_obs = batch["image"][:, 1 : 1 + pred_obs.shape[1], ...]
            masks = inputs["masks"][:, : pred_obs.shape[1]]
            mse_norm, mse_255 = batch_frame_mse(pred_obs, gt_obs)
            append_masked_frames(mse_norm_rows, mse_norm, masks, args.pre_k)
            append_masked_frames(mse_255_rows, mse_255, masks, args.pre_k)

            batch_size = pred_obs.shape[0]
            traj_idx = batch["traj_idx"].detach().cpu().tolist()
            start_idx = batch["start_idx"].detach().cpu().tolist()
            end_idx = batch["end_idx"].detach().cpu().tolist()
            for item_idx in range(batch_size):
                metadata_rows.append(
                    {
                        "traj_idx": int(traj_idx[item_idx]),
                        "start_idx": int(start_idx[item_idx]),
                        "end_idx": int(end_idx[item_idx]),
                    }
                )

    dataset.close()
    write_results(args, metadata_rows, mse_norm_rows, mse_255_rows)


if __name__ == "__main__":
    main()
