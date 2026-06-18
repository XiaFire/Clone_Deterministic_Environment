import argparse
import gc
import io
import os
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from data import TrajectoryDataModule
from models import dict_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute latent windows by encoding each full trajectory once."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained VAE checkpoint.")
    parser.add_argument("--cls", type=str, required=True, help="Model class name, e.g. grvae.")
    parser.add_argument("--train_dir", type=str, help="Training data directory.")
    parser.add_argument("--val_dir", type=str, help="Validation data directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for saved latent windows.")
    parser.add_argument("--type", type=str, default="mc", help="Dataset type: mc or maze.")
    parser.add_argument("--size", type=int, default=64, help="Image size.")
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Dataset scale. Accepts one or two floats.",
    )
    parser.add_argument("--chunk_size", type=int, default=64, help="Temporal chunk length used for encoding.")
    parser.add_argument(
        "--chunks_per_batch",
        type=int,
        default=1,
        help="How many equal-length chunks to encode together.",
    )
    parser.add_argument(
        "--frames_per_sample",
        type=int,
        default=32,
        help="Length of each saved latent window.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Sliding window stride. Defaults to frames_per_sample.",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device indices to use. Indices are relative to the current "
            "CUDA visibility. Example: --gpu_ids 0,1,2,4"
        ),
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Latent dtype saved to disk.",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="none",
        choices=["none", "fp16", "bf16", "auto"],
        help="Autocast dtype used during encoding on CUDA.",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=0,
        help="Dataset cache size. Set 0 to disable caching.",
    )
    return parser.parse_args()


def parse_gpu_ids(gpu_ids: str | None) -> List[int] | None:
    if gpu_ids is None:
        return None

    parsed: List[int] = []
    for raw_value in gpu_ids.split(","):
        value = raw_value.strip()
        if not value:
            continue
        try:
            gpu_id = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id '{value}' in --gpu_ids.") from exc
        if gpu_id < 0:
            raise ValueError("--gpu_ids only accepts non-negative integers.")
        parsed.append(gpu_id)

    if not parsed:
        raise ValueError("--gpu_ids was provided but no valid GPU indices were found.")
    if len(parsed) != len(set(parsed)):
        raise ValueError("--gpu_ids contains duplicates.")
    return parsed


def resolve_worker_gpu_ids(args) -> List[int]:
    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
    if requested_gpu_ids is not None:
        return requested_gpu_ids
    if args.num_gpus <= 0:
        return []
    return list(range(args.num_gpus))


def validate_worker_gpu_ids(worker_gpu_ids: List[int]):
    if not worker_gpu_ids:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Set --num_gpus 0 to run on CPU or fix the CUDA environment."
        )

    visible_count = torch.cuda.device_count()
    invalid_gpu_ids = [gpu_id for gpu_id in worker_gpu_ids if gpu_id >= visible_count]
    if invalid_gpu_ids:
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise ValueError(
            f"Requested GPU ids {invalid_gpu_ids}, but only {visible_count} CUDA device(s) "
            f"are visible to PyTorch. CUDA_VISIBLE_DEVICES={visible_env}. "
            "Use a smaller --num_gpus or pass valid --gpu_ids."
        )


def normalize_scale(scale_values):
    if len(scale_values) == 1:
        return float(scale_values[0])
    if len(scale_values) == 2:
        return [float(scale_values[0]), float(scale_values[1])]
    raise ValueError("--scale must provide one or two floats.")


def resolve_autocast_dtype(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def build_datamodule(args, rank: int):
    data_config = OmegaConf.create(
        {
            "data": {
                "train_dir": args.train_dir,
                "val_dir": args.val_dir,
                "num_workers": 0,
                "batch_size": 1,
                "scale": normalize_scale(args.scale),
                "size": args.size,
                "max_length": args.frames_per_sample,
                "full_trajectory": True,
                "cache_size": args.cache_size,
                "load_images": True,
            },
            "dataset": args.type,
        }
    )
    sink = io.StringIO()
    context = nullcontext() if rank == 0 else redirect_stdout(sink)
    with context:
        dm = TrajectoryDataModule(config=data_config)
        dm.setup("fit" if args.val_dir else "train")
    return dm


def load_model_only_from_checkpoint(ckpt_path: str, model_cls_name: str, device, rank: int):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]
    model_cls = dict_model[model_cls_name]

    sink = io.StringIO()
    context = nullcontext() if rank == 0 else redirect_stdout(sink)
    with context:
        model = model_cls(config)

    model_state_dict = {
        key[len("model.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0 and (missing_keys or unexpected_keys):
        print(
            f"Model checkpoint load summary: missing={len(missing_keys)} "
            f"unexpected={len(unexpected_keys)}"
        )
    return model.eval().to(device)


def get_source_traj_idx(dataset, dataset_idx: int) -> int:
    if hasattr(dataset, "valid_indices"):
        return int(dataset.valid_indices[dataset_idx])
    if hasattr(dataset, "valid_traj_keys") and hasattr(dataset, "traj_key_to_idx"):
        traj_key = dataset.valid_traj_keys[dataset_idx]
        return int(dataset.traj_key_to_idx[traj_key])
    return int(dataset_idx)


def encode_batch(model, batch_images, device, amp_dtype):
    batch_images = batch_images.to(device, non_blocking=True)
    padding_mask = torch.ones(
        batch_images.shape[0],
        batch_images.shape[1],
        dtype=torch.bool,
        device=device,
    )
    autocast_dtype = resolve_autocast_dtype(device, amp_dtype)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        latents = model.get_latents({"image": batch_images, "padding_mask": padding_mask})
    return latents.detach().cpu()


def encode_full_trajectory(model, images, device, chunk_size, chunks_per_batch, amp_dtype):
    seq_len = int(images.shape[0])
    if seq_len == 0:
        return torch.empty((0,), dtype=torch.float32)

    if seq_len <= chunk_size:
        return encode_batch(model, images.unsqueeze(0), device, amp_dtype).squeeze(0)

    latent_chunks: List[torch.Tensor] = []
    n_full_chunks = seq_len // chunk_size
    full_chunk_limit = n_full_chunks * chunk_size

    if n_full_chunks > 0:
        group_span = max(1, chunks_per_batch) * chunk_size
        for group_start in range(0, full_chunk_limit, group_span):
            group_end = min(group_start + group_span, full_chunk_limit)
            group_images = []
            for start_idx in range(group_start, group_end, chunk_size):
                group_images.append(images[start_idx : start_idx + chunk_size])
            batch_images = torch.stack(group_images, dim=0)
            batch_latents = encode_batch(model, batch_images, device, amp_dtype)
            latent_chunks.extend(batch_latents.unbind(0))

    if full_chunk_limit < seq_len:
        tail_latents = encode_batch(
            model,
            images[full_chunk_limit:].unsqueeze(0),
            device,
            amp_dtype,
        )
        latent_chunks.append(tail_latents.squeeze(0))

    return torch.cat(latent_chunks, dim=0)


def iter_sliding_windows(seq_len: int, frames_per_sample: int, window_stride: int) -> Iterable[Tuple[int, int]]:
    if seq_len <= 0:
        return

    window = min(frames_per_sample, seq_len)
    stride = window if window_stride is None else max(1, window_stride)
    last_start = max(0, seq_len - window)

    for start_idx in range(0, last_start + 1, stride):
        yield start_idx, start_idx + window

    if last_start % stride != 0:
        yield last_start, seq_len


def cast_latents_for_save(latents: torch.Tensor, save_dtype: str) -> torch.Tensor:
    if save_dtype == "float16":
        return latents.to(torch.float16)
    return latents.to(torch.float32)


def maybe_all_exist(output_dir: Path, traj_idx: int, num_windows: int) -> bool:
    done_path = output_dir / f"traj{traj_idx:06d}.done"
    if done_path.exists():
        return True
    for window_idx in range(num_windows):
        if not (output_dir / f"traj{traj_idx:06d}_win{window_idx:04d}.pt").exists():
            return False
    return True


def save_window(output_path: Path, latents, actions, traj_idx, start_idx, end_idx):
    payload = {
        "latents": latents,
        "action": actions[start_idx:end_idx].clone(),
        "padding_mask": torch.ones(end_idx - start_idx, dtype=torch.bool),
        "traj_idx": torch.tensor(traj_idx, dtype=torch.long),
        "start_idx": torch.tensor(start_idx, dtype=torch.long),
        "end_idx": torch.tensor(end_idx, dtype=torch.long),
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)


def process_dataset(model, dataset, output_dir: Path, device, args, rank: int, world_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    step = world_size if world_size > 1 else 1
    indices = range(rank, len(dataset), step)
    iterator = indices
    if rank == 0:
        iterator = tqdm(indices, desc=f"Processing {output_dir.name}", total=len(indices))

    stats = {
        "checked": 0,
        "skipped": 0,
        "encoded": 0,
        "saved": 0,
    }

    for dataset_idx in iterator:
        stats["checked"] += 1
        sample = dataset[dataset_idx]
        images = sample["image"]
        actions = sample["action"]
        seq_len = int(images.shape[0])
        traj_idx = get_source_traj_idx(dataset, dataset_idx)
        windows = list(iter_sliding_windows(seq_len, args.frames_per_sample, args.window_stride))
        if not windows:
            continue

        done_path = output_dir / f"traj{traj_idx:06d}.done"
        if done_path.exists():
            stats["skipped"] += 1
            continue

        if maybe_all_exist(output_dir, traj_idx, len(windows)):
            done_path.write_text("done\n")
            stats["skipped"] += 1
            continue

        stats["encoded"] += 1
        full_latents = encode_full_trajectory(
            model=model,
            images=images,
            device=device,
            chunk_size=args.chunk_size,
            chunks_per_batch=args.chunks_per_batch,
            amp_dtype=args.amp_dtype,
        )
        full_latents = cast_latents_for_save(full_latents, args.save_dtype)

        for window_idx, (start_idx, end_idx) in enumerate(windows):
            output_path = output_dir / f"traj{traj_idx:06d}_win{window_idx:04d}.pt"
            if output_path.exists():
                continue
            save_window(
                output_path=output_path,
                latents=full_latents[start_idx:end_idx].clone(),
                actions=actions,
                traj_idx=traj_idx,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            stats["saved"] += 1

        done_path.write_text("done\n")

    if rank == 0:
        print(
            f"{output_dir.name}: checked={stats['checked']} skipped={stats['skipped']} "
            f"encoded={stats['encoded']} saved={stats['saved']}"
        )


def close_if_needed(obj):
    if obj is None:
        return
    close_fn = getattr(obj, "close", None)
    if callable(close_fn):
        close_fn()


def cleanup_worker(model, dm):
    if dm is not None:
        close_if_needed(getattr(dm, "train_dataset", None))
        close_if_needed(getattr(dm, "val_dataset", None))
    if model is not None:
        del model
    if dm is not None:
        del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_worker(rank, worker_gpu_ids, args):
    world_size = len(worker_gpu_ids) if worker_gpu_ids else 1
    if worker_gpu_ids:
        device_index = worker_gpu_ids[rank]
        device = torch.device(f"cuda:{device_index}")
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise RuntimeError(
                f"Failed to claim CUDA device {device_index} for rank {rank}. "
                f"worker_gpu_ids={worker_gpu_ids}, CUDA_VISIBLE_DEVICES={visible_env}. "
                "This usually means one of the requested GPUs is busy or unavailable. "
                "Exclude that GPU with CUDA_VISIBLE_DEVICES or --gpu_ids."
            ) from exc
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    model = None
    dm = None
    try:
        if rank == 0:
            visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            print(
                f"Using device: {device} | world_size={world_size} "
                f"| worker_gpu_ids={worker_gpu_ids or ['cpu']} "
                f"| CUDA_VISIBLE_DEVICES={visible_env}"
            )
        model = load_model_only_from_checkpoint(args.ckpt, args.cls, device, rank)
        dm = build_datamodule(args, rank)
        
        if args.train_dir:
            if rank == 0:
                print("Processing training data...")
            process_dataset(model, dm.train_dataset, Path(args.output_dir) / "train", device, args, rank, world_size)

        if args.val_dir:
            if rank == 0:
                print("Processing validation data...")
            process_dataset(model, dm.val_dataset, Path(args.output_dir) / "val", device, args, rank, world_size)

        if rank == 0:
            print("Precomputation completed.")
    finally:
        cleanup_worker(model, dm)


def main():
    args = parse_args()
    worker_gpu_ids = resolve_worker_gpu_ids(args)
    validate_worker_gpu_ids(worker_gpu_ids)

    if len(worker_gpu_ids) > 1:
        mp.spawn(run_worker, nprocs=len(worker_gpu_ids), args=(worker_gpu_ids, args))
    else:
        run_worker(0, worker_gpu_ids, args)


if __name__ == "__main__":
    main()
