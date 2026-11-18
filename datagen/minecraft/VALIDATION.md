# Validation (2026-11-13)

Checks were run on a fresh copy of this folder, on a different machine from the one that produced the original renders.

1. **Bit-exact re-render.** Rendered the first 8 rows of `trajectory_002242` from `data/mc.h5` with `02_render_blender.py` in Blender 4.0.0. All 8 frames match the original `obs/2242/0000–0007.png` exactly: max |difference| = 0 on all 256×256×3 values.
2. **Generator runs and is deterministic with a seed.** Two runs of `01_generate_trajectories.py --num_trajectories 3 --seed 0` produced identical trajectories (lengths 473, 854, 385). The HDF5 attributes match those of `data/mc.h5`.
3. **End to end.** Rendered the first generated trajectory: 473 frames in 20 s, including about 9 s of scene setup.
4. **Hashes.** `data/mc.h5` (Hugging Face `mc.h5`) has SHA-256 `2d8a37d9b0433578c722a03ec5121a7144418173ecfeffbd28408a9a5720d785`.
