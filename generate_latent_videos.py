import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from data.collate import collate_fn_precomputed
from data.precomputed import PrecomputedDataset
from dfot.dfot import DFOT
from models import dict_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos by rolling out a latent dynamics model and decoding with an autoencoder."
    )
    parser.add_argument("--dyna_config", type=str, required=True, help="Dynamics model config path")
    parser.add_argument("--dyna_ckpt", type=str, required=True, help="Dynamics model checkpoint (.pth)")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Autoencoder checkpoint (.ckpt)")
    parser.add_argument("--vae_cls", type=str, default="grvae", help="Autoencoder model class")
    parser.add_argument("--data_path", type=str, required=True, help="Precomputed latent dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos")
    parser.add_argument("--num_videos", type=int, default=8, help="How many videos to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle dataset before sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pre_k", type=int, default=15, help="Number of context frames kept as ground truth")
    parser.add_argument(
        "--predict_steps",
        type=int,
        default=None,
        help="How many future frames to predict after the context. Defaults to the rest of the window.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="Autoregressive context length. Defaults to config.context_len or config.max_tokens.",
    )
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
    parser.add_argument("--save_compare", type=str2bool, default=True, help="Save GT vs prediction comparison video")
    parser.add_argument("--save_gt", type=str2bool, default=False, help="Also save decoded GT video")
    parser.add_argument("--save_npz", type=str2bool, default=True, help="Save latent tensors to .npz")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id, e.g. 0")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. x_shape=[4,8,8]")
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    if args.gpu is not None:
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_indices(dataset_size: int, num_videos: int, shuffle: bool, seed: int) -> List[int]:
    indices = list(range(dataset_size))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[: min(num_videos, dataset_size)]


def load_dynamics_model(
    config_path: str,
    ckpt_path: str,
    overrides: Sequence[str],
    device: torch.device,
):
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    model = DFOT(config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, config


def load_autoencoder_model(
    ckpt_path: str,
    model_cls_name: str,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]
    model_cls = dict_model[model_cls_name]
    model = model_cls(config)

    model_state_dict = {
        key[len("model.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: unexpected autoencoder keys ignored: {unexpected_keys[:5]}")
    if missing_keys:
        print(f"Warning: missing autoencoder keys: {missing_keys[:5]}")
    return model.eval().to(device)


def resolve_x_shape(config) -> Tuple[int, int, int]:
    x_shape = getattr(config, "x_shape", None)
    if x_shape is None:
        raise ValueError("Dynamics config is missing x_shape. Pass an override like x_shape=[4,8,8].")
    x_shape = tuple(int(dim) for dim in x_shape)
    if len(x_shape) != 3:
        raise ValueError(f"x_shape must have 3 dims, got {x_shape}")
    return x_shape


def reshape_latents(latents: torch.Tensor, x_shape: Tuple[int, int, int]) -> torch.Tensor:
    if latents.ndim == 5:
        return latents
    if latents.ndim != 3:
        raise ValueError(f"Expected latents with shape [B, T, D] or [B, T, C, H, W], got {tuple(latents.shape)}")
    expected_dim = int(np.prod(x_shape))
    if latents.shape[-1] != expected_dim:
        raise ValueError(
            f"Latent dim {latents.shape[-1]} does not match x_shape product {expected_dim}. "
            f"Set the correct x_shape override."
        )
    return latents.reshape(latents.shape[0], latents.shape[1], *x_shape)


def prepare_latent_inputs(batch: Dict, x_shape: Tuple[int, int, int], use_action: bool) -> Dict:
    xs = reshape_latents(batch["latents"], x_shape)
    xs = xs[:, 1:, ...]

    actions = batch["action"]
    if actions.ndim == 2:
        actions = actions.unsqueeze(-1)
    actions = actions[:, :-1, :]
    actions = actions.clone()
    if actions.shape[1] != 0:
        actions[:, 0, :] = -100
    actions[actions == -100] = 0

    masks = batch.get("padding_mask", None)
    if masks is not None:
        masks = masks[:, 1:].clone().bool()
        if masks.shape[1] != 0:
            masks[:, 0] = False

    if not use_action:
        actions = torch.zeros_like(actions)

    return {
        "xs": xs,
        "conditions": actions.long().squeeze(-1),
        "masks": masks,
    }


@torch.no_grad()
def rollout_latent_trajectory(
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
        int(
            context_len
            or getattr(config, "context_len", getattr(config, "n_context_tokens", getattr(config, "max_tokens", 5)))
        ),
    )
    mode = sample_method or getattr(config, "sample_method", "3d")

    for step in range(start_step, xs.shape[1]):
        start = max(0, step - window)
        ctx = predicted[:, start:step, ...]
        cond = actions[:, start : step + 1, ...]
        if mode == "3d":
            step_pred = model.sample_3d(ctx, cond)
        elif mode == "2d":
            step_pred = model.sample(ctx, cond)
        else:
            raise ValueError(f"Unsupported sample_method: {mode}")
        predicted[:, step, ...] = step_pred[:, -1, ...]

    return predicted


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


def sample_valid_len(batch: Dict, batch_index: int) -> int:
    if "padding_mask" in batch:
        return int(batch["padding_mask"][batch_index].sum().item())
    if "latents" in batch:
        return int(batch["latents"][batch_index].shape[0])
    raise KeyError("Batch is missing both padding_mask and latents.")


def trim_samples(
    batch: Dict,
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    pred_obs: torch.Tensor,
    gt_obs: torch.Tensor,
    pre_k: int,
    predict_steps: Optional[int],
):
    samples = []
    batch_size = pred_latents.shape[0]
    for sample_idx in range(batch_size):
        valid_len = max(sample_valid_len(batch, sample_idx) - 1, 0)
        if predict_steps is not None:
            valid_len = min(valid_len, max(pre_k, 0) + max(predict_steps, 0))

        sample = {
            "pred_latents": pred_latents[sample_idx, :valid_len].detach().cpu(),
            "gt_latents": gt_latents[sample_idx, :valid_len].detach().cpu(),
            "pred_obs": pred_obs[sample_idx, :valid_len].detach().cpu(),
            "gt_obs": gt_obs[sample_idx, :valid_len].detach().cpu(),
            "valid_len": valid_len,
        }
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
    if save_compare:
        gt_frames = tensor_video_to_uint8(sample["gt_obs"])
        compare_frames = concat_videos_horizontally(gt_frames, pred_frames)
        compare_path = save_video(compare_frames, output_dir / f"{stem}_compare.{video_format}", fps)
        saved_paths["compare_video"] = str(compare_path)

    if save_gt:
        gt_frames = tensor_video_to_uint8(sample["gt_obs"])
        gt_path = save_video(gt_frames, output_dir / f"{stem}_gt.{video_format}", fps)
        saved_paths["gt_video"] = str(gt_path)

    if save_npz:
        npz_path = output_dir / f"{stem}_latents.npz"
        np.savez_compressed(
            npz_path,
            pred_latents=sample["pred_latents"].numpy(),
            gt_latents=sample["gt_latents"].numpy(),
        )
        saved_paths["latents"] = str(npz_path)

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
    ae_model = load_autoencoder_model(args.vae_ckpt, args.vae_cls, device)
    x_shape = resolve_x_shape(dyna_config)
    use_action = bool(getattr(dyna_config, "action", True))

    dataset = PrecomputedDataset(args.data_path, max_length=None)
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
        collate_fn=collate_fn_precomputed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "args": vars(args),
        "selected_indices": selected_indices,
        "videos": [],
    }

    saved_count = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            tensor_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            inputs = prepare_latent_inputs(
                batch=tensor_batch,
                x_shape=x_shape,
                use_action=use_action,
            )

            xs = inputs["xs"]
            actions = inputs["conditions"]
            if args.predict_steps is not None:
                keep_len = min(xs.shape[1], max(args.pre_k, 0) + max(args.predict_steps, 0))
                xs = xs[:, :keep_len, ...]
                actions = actions[:, :keep_len, ...]

            pred_latents = rollout_latent_trajectory(
                model=dyna_model,
                xs=xs,
                actions=actions,
                pre_k=args.pre_k,
                context_len=args.context_len,
                sample_method=args.sample_method,
                config=dyna_config,
            )

            pred_obs = ae_model.decode(pred_latents)
            gt_obs = ae_model.decode(xs)
            samples = trim_samples(
                batch=batch,
                pred_latents=pred_latents,
                gt_latents=xs,
                pred_obs=pred_obs,
                gt_obs=gt_obs,
                pre_k=args.pre_k,
                predict_steps=args.predict_steps,
            )

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
