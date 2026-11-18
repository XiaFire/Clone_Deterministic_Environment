"""Per-dimension mean / std of a precomputed latent cache, for the dit2d dynamics config (data_mean / data_std).

Usage: python scripts/compute_latent_stats.py --latent_dir <LATENT_DIR>/train --out_prefix <LATENT_DIR>/stats
Writes <out_prefix>_mean.npy and <out_prefix>_std.npy (shape [D]).
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--max_files", type=int, default=20000)
    args = parser.parse_args()

    files = sorted(Path(args.latent_dir).glob("*.pt"))[: args.max_files]
    if not files:
        raise FileNotFoundError(f"no .pt latent files in {args.latent_dir}")
    total, total_sq, count = 0.0, 0.0, 0
    for f in files:
        item = torch.load(f, map_location="cpu")
        lat = item["latents"].double()
        lat = lat.reshape(lat.shape[0], -1)
        mask = item["padding_mask"].bool() if "padding_mask" in item else torch.ones(lat.shape[0], dtype=torch.bool)
        lat = lat[mask]
        total = total + lat.sum(0)
        total_sq = total_sq + (lat ** 2).sum(0)
        count += lat.shape[0]
    mean = total / count
    std = (total_sq / count - mean ** 2).clamp(min=1e-12).sqrt()
    np.save(f"{args.out_prefix}_mean.npy", mean.float().numpy())
    np.save(f"{args.out_prefix}_std.npy", std.float().numpy())
    print(f"{len(files)} files, {count} frames, dim {mean.numel()}: mean |mu| {mean.abs().mean():.3f}, std {std.mean():.3f}")


if __name__ == "__main__":
    main()
