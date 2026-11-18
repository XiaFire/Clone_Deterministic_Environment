# Minecraft-DETERMINISTIC (MC-DET) data generation

This folder regenerates the Minecraft dataset used in *Cloning Deterministic Worlds: The Critical Role of Latent Geometry in Long-Horizon World Models* (CVPR 2026).

Generation happens in two stages:

1. **Trajectories.** `01_generate_trajectories.py` runs an A*-guided policy on the fenced Minecraft map. It writes `(x, y, theta_degrees, action)` for every step.
2. **Observations.** `02_render_blender.py` renders each pose as a first-person 256×256 RGB frame. It uses Blender 4.0.0 (EEVEE) and the map exported to OBJ.

The **exact trajectories used in the paper** are on Hugging Face ([`xiazaishuo/cloning-deterministic-worlds-mc`](https://huggingface.co/datasets/xiazaishuo/cloning-deterministic-worlds-mc)), so stage 1 is optional. Download them into `data/` first:

```bash
mkdir -p data
wget -O data/mc.h5 https://huggingface.co/datasets/xiazaishuo/cloning-deterministic-worlds-mc/resolve/main/mc.h5
```

Run all commands in this README from `datagen/minecraft/`. Rendering is deterministic: re-rendering `data/mc.h5` reproduces the original frames bit for bit (see `VALIDATION.md`).

## Contents

| Path | What it is |
|---|---|
| `01_generate_trajectories.py` | Trajectory generator (noisy A*, 3 discrete actions). |
| `02_render_blender.py` | Blender rendering script. It renders one trajectory as one animation and supports `--idx/--split` for parallel processes. |
| `03_make_splits.py` | Builds train/val directories for `McDataset` from one render folder. Each split gets its own `info.hdf5` and an `obs` symlink. |
| `data/mc.h5` | Downloaded from Hugging Face (see above). The released trajectories: 9,135 trajectories, 5,749,167 steps, 67–1000 steps each (median 613). |
| `assets/maze.npy` | Walkability grid of the map, shape (66, 41), where 1 = walkable. |
| `assets/fence.obj`, `fence.mtl`, `tex/` | Map exported with Mineways 12.09, plus the block textures it uses. |
| `assets/fence.7z` | The Minecraft world save: the map of Gornet & Thomson (2024), enclosed with wooden fences. |
| `LICENSE-map-gornet2024` | MIT license of the original map repository. |
| `SOURCE_MANIFEST.json` | Hashes of the source files before the portability edits. |

## Data format

`data/mc.h5` (it is also the `info.hdf5` of a rendered dataset) has this layout:

```
attrs: generator config (step_size=0.3, delta=15, p_noise=0.0, max_steps=1000, ...)
trajectories/trajectory_000000/data   structured array, fields:
    x, y      float32   continuous position in maze.npy grid units (x = row, y = column)
    theta     float32   heading in degrees
    action    int32     1 = forward 0.3 block, 2 = turn left 15°, 3 = turn right 15°
```

Row `t` stores the pose **after** executing `action[t]`. Rendered frames go to `obs/<trajectory index>/<frame index:04d>.png`. Frame `t` renders row `t`.

## 1. Trajectories (optional)

```bash
pip install -r requirements.txt
python 01_generate_trajectories.py --num_trajectories 10000 --seed 0 \
    --output datasets/mc/info.hdf5 --plot_interval 0
```

The released `data/mc.h5` was generated **without** a fixed seed, so it cannot be regenerated exactly. Use the file itself. `--seed` makes new runs repeatable; we checked that two runs with `--seed 0` give identical files. Trajectories that come out empty are skipped, so the output can hold fewer trajectories than requested. The released file has 9,135.

## 2. Rendering

Download Blender 4.0.0 (Linux x64) from https://download.blender.org/release/Blender4.0/. Its bundled Python already includes `numpy`, `h5py` and `tqdm`. EEVEE needs a GPU with OpenGL; our runs used NVIDIA GPUs headless with `-b`.

Render a single trajectory as a quick check:

```bash
blender-4.0.0-linux-x64/blender -b -P 02_render_blender.py -- \
    --h5 data/mc.h5 --out datasets/mc/obs --limit 1
```

Render the full set with 20 processes:

```bash
for i in $(seq 0 19); do
  blender-4.0.0-linux-x64/blender -b -P 02_render_blender.py -- \
      --h5 data/mc.h5 --out datasets/mc/obs --idx $i --split 20 > render_$i.log 2>&1 &
done
wait
cp data/mc.h5 datasets/mc/info.hdf5
```

If a run is interrupted, re-launch the same command. Directories that are not empty are skipped, so delete the last, partially written trajectory directory of each process before restarting.

**Cost.** We measured about 0.023 s per frame per process (8 EEVEE samples, 256×256), plus about 9 s of scene setup per process. That gives about 37 single-process GPU-hours for all 5.75 M frames. PNG output takes about 104 KB per frame, about **600 GB** in total. Our loader resizes frames on load to the configured `size`. If you train at a lower resolution and disk space is tight, you can downsample right after rendering.

## 3. Train/validation split

```bash
python 03_make_splits.py --h5 data/mc.h5 --obs datasets/mc/obs --out datasets/mc_splits
# datasets/mc_splits/train: trajectory_000000 .. trajectory_008134 (8,135)
# datasets/mc_splits/val:   trajectory_008135 .. trajectory_009134 (1,000)
```

Both splits point to the same rendered frames, so nothing is copied. Use the two directories as `data.train_dir` and `data.val_dir` with `dataset=mc`. The original train/validation split of our MC experiments was not recorded, so this split is the reference split for the release.

## Notes on the original run

- In the original launch loop, `seq 1 20` with `--split 20` skipped split 0, so trajectories 0–456 were never rendered. Interrupted trajectories were also deleted afterwards. The images used in our experiments are therefore a subset of `data/mc.h5`: the trajectories that have an `obs/<idx>/` directory. The loader (`McDataset`) uses exactly those. The loop in this README uses `seq 0 19`.
- The textures in `assets/tex/` are Minecraft block textures exported by Mineways. Minecraft assets are © Mojang/Microsoft; use them for research only.
