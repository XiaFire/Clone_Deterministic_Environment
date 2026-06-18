#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate strided rollout MSE part outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--num_parts", type=int, default=None)
    parser.add_argument("--part_names", nargs="*", default=None)
    return parser.parse_args()


def finite_or_none(values: Iterable[float]) -> List[Optional[float]]:
    return [None if np.isnan(value) else float(value) for value in values]


def summarize_metric(name: str, padded: np.ndarray) -> dict:
    padded = padded.astype(np.float32)
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
    valid_values = padded[valid]
    return {
        f"{name}_global_mean": float(np.mean(valid_values)),
        f"{name}_global_std": float(np.std(valid_values)),
        f"{name}_global_min": float(np.min(valid_values)),
        f"{name}_global_max": float(np.max(valid_values)),
        f"{name}_per_step_mean": finite_or_none(per_step_mean),
        f"{name}_per_step_std": finite_or_none(per_step_std),
        f"{name}_per_step_count": per_step_count.astype(int).tolist(),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.part_names:
        part_names = args.part_names
    elif args.num_parts is not None:
        part_names = [f"{args.name}_part{idx}" for idx in range(args.num_parts)]
    else:
        raise ValueError("Provide --part_names or --num_parts")

    summaries = []
    sample_rows = []
    mse_norm_parts = []
    mse_255_parts = []

    for part_name in part_names:
        summary_path = out_dir / f"{part_name}_rollout_pixel_mse_summary.json"
        samples_path = out_dir / f"{part_name}_rollout_pixel_mse_samples.json"
        npz_path = out_dir / f"{part_name}_rollout_pixel_mse.npz"
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        sample_rows.extend(json.loads(samples_path.read_text(encoding="utf-8")))
        with np.load(npz_path) as data:
            mse_norm_parts.append(data["mse_norm"].astype(np.float32))
            mse_255_parts.append(data["mse_255"].astype(np.float32))

    mse_norm = np.concatenate(mse_norm_parts, axis=0)
    mse_255 = np.concatenate(mse_255_parts, axis=0)
    sample_mean_norm = np.nanmean(mse_norm, axis=1).astype(np.float32)
    sample_mean_255 = np.nanmean(mse_255, axis=1).astype(np.float32)

    for row, mean_norm, mean_255 in zip(sample_rows, sample_mean_norm, sample_mean_255):
        row["mse_norm_mean"] = float(mean_norm)
        row["mse_255_mean"] = float(mean_255)

    first = summaries[0]
    summary = {
        "name": args.name,
        "mode": first["mode"],
        "num_samples": int(mse_norm.shape[0]),
        "pre_k": first["pre_k"],
        "context_len": first["context_len"],
        "sample_method": first["sample_method"],
        "mse_shape": list(mse_norm.shape),
        "args": {
            **first.get("args", {}),
            "name": args.name,
            "aggregated_part_names": part_names,
            "aggregated_part_num_samples": [summary["num_samples"] for summary in summaries],
        },
        **summarize_metric("mse_norm", mse_norm),
        **summarize_metric("mse_255", mse_255),
    }

    json_path = out_dir / f"{args.name}_rollout_pixel_mse_summary.json"
    samples_path = out_dir / f"{args.name}_rollout_pixel_mse_samples.json"
    npz_path = out_dir / f"{args.name}_rollout_pixel_mse.npz"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    samples_path.write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        mse_norm=mse_norm,
        mse_255=mse_255,
        sample_mean_norm=sample_mean_norm,
        sample_mean_255=sample_mean_255,
        traj_idx=np.array([row["traj_idx"] for row in sample_rows], dtype=np.int64),
        start_idx=np.array([row["start_idx"] for row in sample_rows], dtype=np.int64),
        end_idx=np.array([row["end_idx"] for row in sample_rows], dtype=np.int64),
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {samples_path}", flush=True)
    print(f"Saved {npz_path}", flush=True)


if __name__ == "__main__":
    main()
