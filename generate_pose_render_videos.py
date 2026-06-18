import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence
from contextlib import contextmanager

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dynamics_inputs import prepare_dynamics_inputs


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roll out a pose predictor, render videos, and save N samples."
    )
    parser.add_argument("--dyna_config", type=str, required=True, help="Pose model config")
    parser.add_argument("--dyna_ckpt", type=str, required=True, help="Pose model checkpoint")
    parser.add_argument("--render_ckpt", type=str, required=True, help="Render model checkpoint")
    parser.add_argument("--render_cls", type=str, default="render", help="Render model class")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save videos")
    parser.add_argument("--type", type=str, default="mc", choices=["mc", "maze"])
    parser.add_argument("--num_videos", type=int, default=8, help="How many videos to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--max_length", type=int, default=100, help="Max trajectory length")
    parser.add_argument("--full_trajectory", type=str2bool, default=True, help="Use full trajectories")
    parser.add_argument("--stage", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--size", type=int, default=64, help="Image size")
    parser.add_argument("--scale", type=float, nargs=2, default=[33.0, 20.5], help="Pose scale")
    parser.add_argument(
        "--pose_center",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Pose center",
    )
    parser.add_argument("--load_images", type=str2bool, default=False, help="Load GT images")
    parser.add_argument(
        "--save_compare",
        type=str2bool,
        default=False,
        help="If GT images are available, also save side-by-side comparison videos",
    )
    parser.add_argument("--save_gt", type=str2bool, default=False, help="Also save GT video")
    parser.add_argument("--save_npz", type=str2bool, default=True, help="Save predicted poses")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle dataset before sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--pre_k", type=int, default=5, help="Number of context frames to keep as GT")
    parser.add_argument("--context_len", type=int, default=None, help="Autoregressive context length")
    parser.add_argument(
        "--sample_method",
        type=str,
        default=None,
        choices=["2d", "3d"],
        help="Override sampling method from config",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument(
        "--video_format",
        type=str,
        default="mp4",
        choices=["mp4", "gif"],
        help="Video format to save",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id, e.g. 0")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. data_std=0.005")
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def patched_hf_hub_version(fake_version: str = "0.99.0"):
    import importlib.metadata as importlib_metadata

    real_version = importlib_metadata.version

    def patched_version(package_name: str):
        if package_name == "huggingface-hub":
            return fake_version
        return real_version(package_name)

    importlib_metadata.version = patched_version
    try:
        yield
    finally:
        importlib_metadata.version = real_version


def resolve_device(args) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    if args.gpu is not None:
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(args):
    collate_module = load_module("grwm_data_collate", ROOT / "data" / "collate.py")
    if args.type == "mc":
        dataset_module = load_module("grwm_data_mc", ROOT / "data" / "mc.py")
        dataset_cls = dataset_module.McDataset
    else:
        dataset_module = load_module("grwm_data_maze", ROOT / "data" / "maze.py")
        dataset_cls = dataset_module.MazeDataset

    common_kwargs = dict(
        data_dir=args.data_path,
        max_length=args.max_length,
        scale=args.scale,
        size=args.size,
        full_trajectory=args.full_trajectory,
        stage=args.stage,
        pose_center=args.pose_center,
        load_images=args.load_images or args.save_compare or args.save_gt,
    )
    return dataset_cls(**common_kwargs), collate_module.collate_fn_maze


def load_dynamics_model(config_path: str, ckpt_path: str, overrides, device: torch.device):
    from omegaconf import OmegaConf
    from dfot.dfot import DFOT

    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    model = DFOT(config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, config


def load_render_model(ckpt_path: str, model_cls: str, device: torch.device) -> torch.nn.Module:
    if model_cls != "render":
        raise NotImplementedError(
            f"Only render checkpoints are supported without Lightning imports, got {model_cls!r}."
        )

    with patched_hf_hub_version():
        oracle_module = load_module("grwm_models_oracle", ROOT / "models" / "oracle.py")
        RenderModel = oracle_module.RenderModel

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("hyper_parameters")
        if config is None:
            raise KeyError(f"Checkpoint {ckpt_path} is missing hyper_parameters.")

        model = RenderModel(config)
        raw_state = ckpt.get("state_dict", ckpt)
        model_state = {
            key[len("model.") :]: value
            for key, value in raw_state.items()
            if key.startswith("model.")
        }
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if unexpected_keys:
            print(f"Warning: unexpected render keys ignored: {unexpected_keys[:5]}")
        if missing_keys:
            print(f"Warning: missing render keys: {missing_keys[:5]}")
        return model.to(device).eval()


def rollout_world_model(
    batch: Dict,
    dynamics: torch.nn.Module,
    renderer: torch.nn.Module,
    config,
    pre_k: int,
    context_len,
    sample_method,
) -> Dict:
    inputs = prepare_dynamics_inputs(batch, config)
    xs = inputs["xs"]
    actions = inputs["conditions"]

    pred_pose = torch.empty_like(xs)
    if pre_k <= 0:
        start_step = 0
    else:
        start_step = min(pre_k, xs.shape[1])
        pred_pose[:, :start_step, ...] = xs[:, :start_step, ...]

    window = max(
        1,
        int(
            context_len
            or getattr(config, "n_context_tokens", getattr(config, "max_tokens", 5))
        ),
    )
    mode = sample_method or getattr(config, "sample_method", "3d")

    for step in range(start_step, xs.shape[1]):
        start = max(0, step - window)
        ctx = pred_pose[:, start:step, ...]
        cond = actions[:, start : step + 1, ...]
        if mode == "3d":
            step_pred = dynamics.sample_3d(ctx, cond)
        elif mode == "2d":
            step_pred = dynamics.sample(ctx, cond)
        else:
            raise ValueError(f"Unsupported sample_method: {mode}")
        pred_pose[:, step, ...] = step_pred[:, -1, ...]

    pred_obs = renderer.decode(pred_pose)

    output = {
        "pred_pose": pred_pose,
        "pred_obs": pred_obs,
        "gt_pose": xs,
    }
    if "image" in batch:
        output["gt_obs"] = batch["image"][:, 1:, ...]
    return output


def choose_indices(dataset_size: int, num_videos: int, shuffle: bool, seed: int) -> List[int]:
    indices = list(range(dataset_size))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[: min(num_videos, dataset_size)]


def tensor_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    video = video.detach().cpu().float()
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with shape [T, C, H, W], got {tuple(video.shape)}")
    video = video.clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().to(torch.uint8)
    video = video.permute(0, 2, 3, 1).numpy()
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    return video


def concat_videos_horizontally(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num_frames = min(left.shape[0], right.shape[0])
    return np.concatenate([left[:num_frames], right[:num_frames]], axis=2)


def save_video(frames: np.ndarray, output_path: Path, fps: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.suffix.lower() == ".gif":
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
        else:
            imageio.mimsave(output_path, frames, fps=fps, macro_block_size=None)
        return output_path
    except Exception as exc:
        if output_path.suffix.lower() == ".mp4":
            fallback_path = output_path.with_suffix(".gif")
            print(f"Warning: failed to save {output_path.name} as mp4 ({exc}). Falling back to gif.")
            imageio.mimsave(fallback_path, frames, fps=fps, loop=0)
            return fallback_path
        raise


def trim_batch_tensors(batch: Dict, output: Dict) -> Sequence[Dict]:
    seq_lengths = batch["seq_lengths"].detach().cpu()
    samples = []
    for sample_idx in range(seq_lengths.shape[0]):
        valid_len = max(int(seq_lengths[sample_idx].item()) - 1, 0)
        sample = {
            "pred_pose": output["pred_pose"][sample_idx, :valid_len].detach().cpu(),
            "pred_obs": output["pred_obs"][sample_idx, :valid_len].detach().cpu(),
            "gt_pose": output["gt_pose"][sample_idx, :valid_len].detach().cpu(),
            "valid_len": valid_len,
        }
        if "gt_obs" in output:
            sample["gt_obs"] = output["gt_obs"][sample_idx, :valid_len].detach().cpu()
        samples.append(sample)
    return samples


def save_sample(
    sample: Dict,
    dataset_index: int,
    output_dir: Path,
    video_format: str,
    fps: int,
    save_compare: bool,
    save_gt: bool,
    save_npz: bool,
) -> Dict[str, str]:
    stem = f"sample_{dataset_index:06d}"
    pred_frames = tensor_video_to_uint8(sample["pred_obs"])
    pred_path = save_video(pred_frames, output_dir / f"{stem}_pred.{video_format}", fps)

    saved_paths = {"pred_video": str(pred_path)}
    gt_frames = None
    if (save_compare or save_gt) and "gt_obs" in sample:
        gt_frames = tensor_video_to_uint8(sample["gt_obs"])

    if save_compare and gt_frames is not None:
        compare_frames = concat_videos_horizontally(gt_frames, pred_frames)
        compare_path = save_video(
            compare_frames,
            output_dir / f"{stem}_compare.{video_format}",
            fps,
        )
        saved_paths["compare_video"] = str(compare_path)

    if save_gt and gt_frames is not None:
        gt_path = save_video(gt_frames, output_dir / f"{stem}_gt.{video_format}", fps)
        saved_paths["gt_video"] = str(gt_path)

    if save_npz:
        npz_path = output_dir / f"{stem}_poses.npz"
        np.savez_compressed(
            npz_path,
            pred_pose=sample["pred_pose"].numpy(),
            gt_pose=sample["gt_pose"].numpy(),
        )
        saved_paths["poses"] = str(npz_path)

    return saved_paths


def main():
    args = parse_args()
    device = resolve_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dyna_model, dyna_config = load_dynamics_model(
        args.dyna_config,
        args.dyna_ckpt,
        args.overrides,
        device,
    )
    render_model = load_render_model(args.render_ckpt, args.render_cls, device)

    dataset, collate_fn = build_dataset(args)
    selected_indices = choose_indices(
        dataset_size=len(dataset),
        num_videos=args.num_videos,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "args": vars(args),
        "selected_indices": selected_indices,
        "videos": [],
    }

    saved_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            tensor_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = rollout_world_model(
                batch=tensor_batch,
                dynamics=dyna_model,
                renderer=render_model,
                config=dyna_config,
                pre_k=args.pre_k,
                context_len=args.context_len,
                sample_method=args.sample_method,
            )
            samples = trim_batch_tensors(batch=batch, output=output)

            for local_idx, sample in enumerate(samples):
                dataset_index = selected_indices[saved_count]
                if sample["valid_len"] <= 0:
                    print(f"Skipping dataset index {dataset_index}: no valid frames after trimming.")
                    saved_count += 1
                    continue

                saved_paths = save_sample(
                    sample=sample,
                    dataset_index=dataset_index,
                    output_dir=output_dir,
                    video_format=args.video_format,
                    fps=args.fps,
                    save_compare=args.save_compare,
                    save_gt=args.save_gt,
                    save_npz=args.save_npz,
                )
                manifest["videos"].append(
                    {
                        "dataset_index": dataset_index,
                        "num_frames": sample["valid_len"],
                        **saved_paths,
                    }
                )
                print(
                    f"[{saved_count + 1}/{len(selected_indices)}] saved dataset index "
                    f"{dataset_index} from batch {batch_idx}, item {local_idx}"
                )
                saved_count += 1

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print(f"Finished generating {len(manifest['videos'])} videos in {output_dir}")


if __name__ == "__main__":
    main()
