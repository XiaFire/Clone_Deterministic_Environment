# Cloning Deterministic Worlds

Official code release for **Cloning Deterministic Worlds: The Critical Role of Latent Geometry in Long-Horizon World Models**.

This repository provides code for training and evaluating two latent-space world model pipelines on the Maze9 dataset:

- **VAE baseline**: train an autoencoder, precompute latents, train a dynamics model in VAE latent space, then evaluate pixel-space rollout error.
- **GRWM**: train a GRWM autoencoder, precompute latents, train a dynamics model in GRWM latent space, then evaluate pixel-space rollout error.

## Dataset

Download the Maze9 data:

- Training set (`9x9_n0.8`): [download](https://huggingface.co/datasets/xiazaishuo/cloning-deterministic-worlds-maze9/tree/main/9x9_n0.8)
- Validation set (`9x9_val`): [download](https://huggingface.co/datasets/xiazaishuo/cloning-deterministic-worlds-maze9/tree/main/9x9_val)

The Maze3x3 dataset will also be released.

After downloading, set the training and validation directories:

```bash
export TRAIN_DIR=/path/to/9x9_n0.8
export VAL_DIR=/path/to/9x9_val
```

Each dataset directory should use the Maze dataset format expected by this repository, including `info.hdf5` and `obs/*.npy`.

### Minecraft (MC-DET)

The MC trajectories are on Hugging Face: [`xiazaishuo/cloning-deterministic-worlds-mc`](https://huggingface.co/datasets/xiazaishuo/cloning-deterministic-worlds-mc). The 256×256 frames are about 600 GB, so they are not hosted. Render them locally with Blender instead; rendering reproduces our frames bit for bit. See [`datagen/minecraft/README.md`](datagen/minecraft/README.md) for rendering and for the train/validation split. The resulting directories load with `dataset=mc` (`data/mc.py`).

## Environment

Create or activate a Python environment, then install the dependencies from this directory:

```bash
pip install -r requirements.txt
```

For non-interactive shells or tmux jobs, you can let the script activate the environment:

```bash
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=grwm
```

Weights & Biases logging is disabled by default. To enable online logging:

```bash
export WANDB_MODE=online
```

## Run Everything

From this directory:

```bash
bash run_maze9_vae_grwm.sh
```

By default, the VAE and GRWM autoencoders are trained in parallel on one GPU each:

```bash
export AE_GPU_VAE=0
export AE_GPU_GRWM=1
```

Precomputation, dynamics training, and evaluation can use multiple GPUs:

```bash
export PRECOMPUTE_GPUS=0,1,2,3,4,5,6,7
export DYNAMICS_GPUS=0,1,2,3,4,5,6,7
export PART_GPUS=0,1,2,3,4,5,6,7
```

## Run by Stage

The pipeline can also be resumed stage by stage:

```bash
bash run_maze9_vae_grwm.sh ae
bash run_maze9_vae_grwm.sh precompute
bash run_maze9_vae_grwm.sh dynamics
bash run_maze9_vae_grwm.sh eval
```

## Visualization Notebook

After training and evaluation, open the notebook to visualize the two autoencoder checkpoints and rollout metrics:

```bash
jupyter lab visualization/maze9_latent_clustering_and_mse.ipynb
```

The notebook shows latent KMeans clusters on the Maze9 training-set `(x, y)` plane and plots VAE vs GRWM rollout MSE per step.

## Common Overrides

Main paths and runtime settings can be changed through environment variables. For example:

```bash
export LATENT_ROOT=/path/to/cache_dir
```

Model and evaluation defaults are defined in the run script and `configs/mc_unet_maze9.yaml`.

## Outputs

The script writes:

- autoencoder checkpoints under `checkpoints/`
- precomputed latents under `LATENT_ROOT`
- dynamics checkpoints under `ckpts/`
- rollout metrics and plots under `outputs/maze9_metrics/`

The final comparison plot is saved as:

```text
outputs/maze9_metrics/maze9_vae_vs_grwm_norm_mse_per_step.pdf
```

## Citation

```bibtex
@InProceedings{Xia_2026_CVPR,
    author    = {Xia, Zaishuo and Lu, Yukuan and Li, Xinyi and Xu, Yifan and Chen, Yubei},
    title     = {Cloning Deterministic Worlds: The Critical Role of Latent Geometry in Long-Horizon World Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {32602-32612}
}
```
