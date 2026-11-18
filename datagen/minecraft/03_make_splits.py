"""Create train/val directories for McDataset from one rendered MC-DET directory.

Both splits share the rendered frames: each split gets its own info.hdf5 (a subset of the
trajectory groups, original names kept) and an `obs` symlink to the common render folder.
McDataset matches trajectory_XXXXXX to obs/XXXXXX/, so no frames are copied.

Default split: the last 1000 trajectories (trajectory_008135 ... trajectory_009134) are validation.
"""
import argparse
import os

import h5py


def write_subset(src, dst, keys):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        g = fo.create_group("trajectories")
        for k in keys:
            fi.copy(fi["trajectories"][k], g, name=k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/mc.h5", help="Trajectory file that was rendered")
    p.add_argument("--obs", default="datasets/mc/obs", help="Rendered frames (obs/<idx>/<frame>.png)")
    p.add_argument("--out", default="datasets/mc_splits", help="Writes <out>/train and <out>/val")
    p.add_argument("--num_val", type=int, default=1000, help="Last N trajectories go to validation")
    a = p.parse_args()

    with h5py.File(a.h5, "r") as f:
        keys = sorted(f["trajectories"].keys())
    splits = {"train": keys[:-a.num_val], "val": keys[-a.num_val:]}
    obs = os.path.abspath(a.obs)
    for name, ks in splits.items():
        d = os.path.join(a.out, name)
        write_subset(a.h5, os.path.join(d, "info.hdf5"), ks)
        link = os.path.join(d, "obs")
        if not os.path.lexists(link):
            os.symlink(obs, link)
        print(f"{name}: {len(ks)} trajectories ({ks[0]} .. {ks[-1]}) -> {d}")


if __name__ == "__main__":
    main()
