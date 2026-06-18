import argparse
import math
import os
import warnings
from typing import Dict, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

try:
    from lightning.fabric import Fabric
except ImportError:
    from lightning_fabric import Fabric

try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:
    WandbLogger = None

from data import TrajectoryDataModule

warnings.filterwarnings("ignore")


def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        param_count += 1
    return total_norm ** 0.5 if param_count > 0 else 0.0


def sync_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="pose_dfot", help="Run name")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=50_000, help="Number of training steps")
    parser.add_argument("--model", type=str, default="dfot", help="Model type")
    parser.add_argument("--train_path", type=str, required=True, help="Training data directory")
    parser.add_argument("--val_path", type=str, required=True, help="Validation data directory")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ids to use")
    parser.add_argument("--vae_gpu", type=int, default=1, help="Unused legacy arg")
    parser.add_argument("--pre_k", type=int, default=5, help="Number of context steps for rollout validation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--val_freq", type=int, default=-1, help="Validation frequency in epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--precomputed", type=str2bool, default=False, help="Whether to read precomputed latent windows")
    parser.add_argument("--action", type=str2bool, default=True, help="Whether to use action as condition")
    parser.add_argument("--input_key", type=str, default="auto", help="Supported values: auto, pose, latents")
    parser.add_argument("--size", type=int, default=64, help="Image size for optional dataset compatibility")
    parser.add_argument("--type", type=str, default="mc", help="maze or mc")
    parser.add_argument("--use_mask", type=str2bool, default=False, help="Unused legacy arg")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum trajectory length")
    parser.add_argument("--load_images", type=str2bool, default=False, help="Whether to load images in dataset")
    parser.add_argument("--ckpt", type=str, default=None, help="Unused legacy arg")
    parser.add_argument("--cls", type=str, default=None, help="Unused legacy arg")
    parser.add_argument("--val_image_dir", type=str, default=None, help="Unused legacy arg")
    parser.add_argument("--train_image_dir", type=str, default=None, help="Unused legacy arg")
    parser.add_argument("--context_len", type=int, default=1, help="Context length for autoregressive sampling")
    parser.add_argument(
        "--save_steps",
        type=str,
        default="",
        help="Comma-separated global optimizer steps to save as additional checkpoints.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs=2,
        default=[33.0, 20.5],
        help="Scale of the data",
    )
    parser.add_argument(
        "--pose_center",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Pose center of the data",
    )
    parser.add_argument("overrides", nargs="*", help="overrides (key=value)")
    return parser.parse_args()


def parse_save_steps(value: str):
    if not value:
        return set()
    steps = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step <= 0:
            raise ValueError(f"save_steps must be positive, got {step}")
        steps.add(step)
    return steps


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_training_steps:
            decay_steps = num_training_steps - num_warmup_steps
            pct = float(current_step - num_warmup_steps) / float(max(1, decay_steps))
            return (1.0 - pct) * (1.0 - min_lr_ratio) + min_lr_ratio
        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_optimizer(model, config):
    return optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-3),
        betas=config.get("optimizer_beta", [0.9, 0.99]),
    )


def create_lr_scheduler(optimizer, config):
    if "lr_scheduler" not in config:
        return None
    lr_config = config.lr_scheduler
    scheduler_name = lr_config.get("name", "constant_with_warmup")

    if scheduler_name == "constant_with_warmup":
        num_warmup_steps = lr_config.get("num_warmup_steps", 5000)
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    if scheduler_name == "decay_with_warmup":
        num_warmup_steps = lr_config.get("num_warmup_steps", 5000)
        num_training_steps = lr_config.get("num_training_steps", 50_000)
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    warnings.warn(f"Unsupported scheduler: {scheduler_name}")
    return None


def create_model(config):
    from dfot.dfot import DFOT

    if config.model.lower() != "dfot":
        raise ValueError(f"Invalid model type: {config.model}")
    return DFOT(config)


def validate_runtime_config(config) -> None:
    input_key = config.get("input_key", "auto")
    if input_key not in {"auto", "pose", "latents"}:
        raise ValueError(
            f"dyna_efficient.py only supports input_key=auto, pose, or latents, got {input_key!r}."
        )

    if config.get("precomputed", False) and input_key == "pose":
        raise ValueError("Precomputed datasets contain latents, so input_key=pose is invalid.")

    if input_key == "latents":
        x_shape = config.get("x_shape", None)
        if x_shape is None:
            raise ValueError("input_key=latents requires x_shape in the config or overrides.")


def warn_legacy_args(args) -> None:
    if args.ckpt or args.cls:
        print(
            "Warning: --ckpt/--cls are ignored in dyna_efficient.py."
        )
    if WandbLogger is None:
        print("Warning: WandbLogger is unavailable. Training will run without external logging.")


def make_data_module(config):
    data_config = OmegaConf.create(
        {
            "data": {
                "train_dir": config.train_path,
                "val_dir": config.val_path,
                "num_workers": config.get("num_workers", 8),
                "batch_size": config.batch_size,
                "precomputed": config.get("precomputed", False),
                "type": config.type,
                "scale": config.scale,
                "size": config.size,
                "pose_center": config.pose_center,
                "max_length": config.max_length,
                "load_images": config.get("load_images", False),
                "train_image_dir": config.get("train_image_dir", None),
                "val_image_dir": config.get("val_image_dir", None),
                "cache_size": config.get("cache_size", 10000),
                "val_cache_size": config.get("val_cache_size", config.get("cache_size", 1000)),
                "full_trajectory": config.get("full_trajectory", False),
            },
            "dataset": config.type,
        }
    )
    dm = TrajectoryDataModule(config=data_config)
    dm.setup("fit")
    return dm


def _select_input_key(batch: Dict, config) -> str:
    input_key = config.get("input_key", "auto")
    if input_key != "auto":
        return input_key
    if "latents" in batch:
        return "latents"
    if "pose" in batch:
        return "pose"
    raise KeyError("Batch missing both 'latents' and 'pose'; set --input_key explicitly.")


def _reshape_pose(xs: torch.Tensor, config) -> torch.Tensor:
    if xs.ndim != 3:
        return xs
    x_shape = config.get("x_shape", None)
    if x_shape is not None and len(x_shape) == 3 and int(x_shape[1]) == 1 and int(x_shape[2]) == 1:
        return xs.unsqueeze(-1).unsqueeze(-1)
    return xs


def _reshape_latents(xs: torch.Tensor, config) -> torch.Tensor:
    if xs.ndim != 3:
        return xs

    x_shape = tuple(int(dim) for dim in config.x_shape)
    latent_dim = xs.shape[-1]
    expected_dim = int(np.prod(x_shape))
    if latent_dim != expected_dim:
        raise ValueError(
            f"Latent dim {latent_dim} does not match x_shape product {expected_dim}. "
            f"Set x_shape correctly, e.g. x_shape=[4,8,8]."
        )
    return xs.reshape(xs.shape[0], xs.shape[1], *x_shape)


def _prepare_inputs(batch: Dict, config) -> Dict:
    input_key = _select_input_key(batch, config)
    if input_key == "latents":
        if "latents" not in batch:
            raise KeyError("input_key=latents requires batch['latents'].")
        xs = batch["latents"][:, 1:, ...]
        xs = _reshape_latents(xs, config)
    elif input_key == "pose":
        if "pose" not in batch:
            raise KeyError("input_key=pose requires batch['pose'].")
        xs = batch["pose"][:, 1:, ...]
        xs = _reshape_pose(xs, config)
    else:
        raise ValueError(f"Unsupported input_key: {input_key}")

    actions = batch["action"]
    if actions.ndim == 2:
        actions = actions.unsqueeze(-1)
    actions = actions[:, :-1, :]
    if actions.shape[1] != 0:
        actions[:, 0, :] = -100

    padding_mask = batch.get("padding_mask", None)
    if padding_mask is not None:
        padding_mask = padding_mask[:, 1:]
        if padding_mask.shape[1] != 0:
            padding_mask[:, 0] = 0

    actions = actions.clone()
    actions[actions == -100] = 0

    inputs = {
        "xs": xs,
        "conditions": actions.long().squeeze(-1),
        "masks": padding_mask,
    }
    if not config.action:
        inputs["conditions"] = torch.zeros_like(inputs["conditions"])
    return inputs


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target = target.reshape(target.shape[0], target.shape[1], -1)

    if mask is None:
        return torch.mean((pred - target) ** 2)

    mask = mask.bool()
    if not mask.any():
        return pred.new_tensor(0.0)

    mask = mask.unsqueeze(-1).expand_as(pred)
    diff = pred[mask] - target[mask]
    return torch.mean(diff ** 2)


@torch.no_grad()
def rollout_trajectory(model, xs: torch.Tensor, actions: torch.Tensor, config) -> torch.Tensor:
    predicted = torch.empty_like(xs)
    pre_k = int(config.pre_k)
    predicted[:, :pre_k, ...] = xs[:, :pre_k, ...]
    window = max(1, int(config.get("context_len", 1)))
    default_sample_method = "3d" if xs.ndim == 5 else "2d"
    sample_method = config.get("sample_method", default_sample_method)

    for step in range(pre_k, xs.shape[1]):
        start = max(0, step - window)
        ctx = predicted[:, start:step, ...]
        cond = actions[:, start : step + 1, ...]
        if sample_method == "3d":
            step_pred = model.sample_3d(ctx, cond)
        elif sample_method == "2d":
            step_pred = model.sample(ctx, cond)
        else:
            raise ValueError(f"Invalid sample method: {sample_method}")
        predicted[:, step, ...] = step_pred[:, -1, ...]

    return predicted


def train_epoch(
    fabric,
    model,
    optimizer,
    train_loader,
    epoch,
    scheduler=None,
    config=None,
    global_step=0,
    max_steps=None,
    milestone_paths=None,
    saved_milestones=None,
):
    model.train()
    losses = []
    grad_norms = []
    grad_norm_freq = config.logging.get("grad_norm_freq", 100) if config else 100
    milestone_paths = milestone_paths or {}
    saved_milestones = saved_milestones if saved_milestones is not None else set()

    iterator = train_loader
    if fabric.is_global_zero:
        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(iterator):
        if max_steps is not None and global_step >= max_steps:
            break

        batch = fabric.to_device(batch)
        inputs = _prepare_inputs(batch, config)

        optimizer.zero_grad()
        output = model(**inputs)
        loss = output["loss"]

        fabric.backward(loss)
        if batch_idx % grad_norm_freq == 0:
            grad_norm = compute_grad_norm(model)
            grad_norms.append(grad_norm)
            step = epoch * len(train_loader) + batch_idx
            if fabric.is_global_zero:
                fabric.log("train/grad_norm", grad_norm, step=step)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        global_step += 1

        if global_step in milestone_paths and global_step not in saved_milestones:
            fabric.save(milestone_paths[global_step], model.state_dict())
            saved_milestones.add(global_step)
            if fabric.is_global_zero:
                print(f"Saved checkpoint to {milestone_paths[global_step]}")
            barrier()

        loss_value = loss.item()
        losses.append(loss_value)
        if fabric.is_global_zero:
            iterator.set_postfix({"loss": f"{loss_value:.4f}", "step": global_step})

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    device = next(model.parameters()).device
    avg_loss = sync_mean(avg_loss, device)
    avg_grad_norm = sync_mean(avg_grad_norm, device)
    return avg_loss, avg_grad_norm, global_step


@torch.no_grad()
def validate(fabric, model, loader, epoch=0, config=None):
    del epoch
    model.eval()
    metrics = {"val_loss": [], "traj_loss": []}
    device = next(model.parameters()).device

    iterator = loader
    if fabric.is_global_zero:
        iterator = tqdm(loader, desc="Validating")

    for batch_idx, batch in enumerate(iterator):
        if batch_idx >= 1:
            break

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        inputs = _prepare_inputs(batch, config)

        output = model(**inputs)
        metrics["val_loss"].append(output["loss"].item())

        xs = inputs["xs"]
        actions = inputs["conditions"]
        masks = inputs["masks"]
        if xs.shape[1] <= config.pre_k:
            metrics["traj_loss"].append(0.0)
            continue

        predicted = rollout_trajectory(model, xs, actions, config)
        traj_loss = masked_mse(
            predicted[:, config.pre_k :, ...],
            xs[:, config.pre_k :, ...],
            masks[:, config.pre_k :] if masks is not None else None,
        )
        metrics["traj_loss"].append(traj_loss.item())

    reduced = {}
    for key, values in metrics.items():
        local_value = float(np.mean(values)) if values else 0.0
        reduced[key] = sync_mean(local_value, device)
    return reduced


def main():
    args = parse_args()
    save_steps = parse_save_steps(args.save_steps)

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, vars(args))
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    config.action = args.action
    validate_runtime_config(config)

    devices = [int(device_id) for device_id in args.gpu.split(",")]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    fabric_devices = devices if accelerator == "gpu" else 1
    if accelerator == "cpu":
        print("Warning: CUDA is unavailable. Falling back to CPU execution.")

    logger = None
    if WandbLogger is not None:
        logger = WandbLogger(project="GRWM", group=args.model, name=args.name)

    fabric = Fabric(
        accelerator=accelerator,
        devices=fabric_devices,
        loggers=logger,
    )
    fabric.launch()
    if fabric.is_global_zero:
        warn_legacy_args(args)
        for key, value in vars(args).items():
            print(f"{key}: {value}")
    if getattr(fabric, "_loggers", None):
        fabric.logger.log_hyperparams(args)

    model = create_model(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_lr_scheduler(optimizer, config)

    ckpt_path = f"./ckpts/{args.name}_last.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        if fabric.is_global_zero:
            print(f"Loaded checkpoint from {ckpt_path}")

    model, optimizer = fabric.setup(model, optimizer)
    if hasattr(model, "mark_forward_method"):
        model.mark_forward_method("sample_3d")

    if fabric.is_global_zero:
        print(
            f"Model parameters: {sum(param.numel() for param in model.parameters()) / 1024 / 1024:.2f} MB"
        )

    dm = make_data_module(config)
    train_loader = fabric.setup_dataloaders(dm.train_dataloader())
    val_loader = fabric.setup_dataloaders(dm.val_dataloader())

    os.makedirs("./ckpts", exist_ok=True)
    best_traj_loss = float("inf")
    best_ckpt_path = f"./ckpts/{args.name}_best_val_traj_loss.pth"
    milestone_paths = {
        step: f"./ckpts/{args.name}_step{step}.pth"
        for step in save_steps
        if step <= args.steps
    }
    saved_milestones = set()

    epochs = math.ceil(args.steps / len(train_loader))
    global_step = 0
    for epoch in range(epochs):
        train_loss, avg_grad_norm, global_step = train_epoch(
            fabric,
            model,
            optimizer,
            train_loader,
            epoch,
            scheduler=scheduler,
            config=config,
            global_step=global_step,
            max_steps=args.steps,
            milestone_paths=milestone_paths,
            saved_milestones=saved_milestones,
        )

        if fabric.is_global_zero:
            fabric.log("train/loss", train_loss, step=epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        if fabric.is_global_zero:
            fabric.log("train/lr", current_lr, step=epoch)
        if avg_grad_norm > 0 and fabric.is_global_zero:
            fabric.log("train/avg_grad_norm", avg_grad_norm, step=epoch)

        if fabric.is_global_zero:
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.8f}, "
                f"LR: {current_lr:.8f}, Grad Norm: {avg_grad_norm:.6f}, "
                f"Global Step: {global_step}/{args.steps}"
            )

        fabric.save(ckpt_path, model.state_dict())
        if fabric.is_global_zero:
            print(f"Saved checkpoint to {ckpt_path}")
        barrier()

        if global_step >= args.steps:
            break

        if args.val_freq > 0 and epoch % args.val_freq == 0:
            if fabric.is_global_zero:
                print(f"Epoch {epoch + 1}/{epochs} - Starting validation...")
            val_metrics = validate(fabric, model, val_loader, epoch=epoch, config=config)
            is_best = val_metrics["traj_loss"] < best_traj_loss
            if is_best:
                best_traj_loss = val_metrics["traj_loss"]
            if fabric.is_global_zero:
                fabric.log("val/loss", val_metrics["val_loss"], step=epoch)
                fabric.log("val/traj_loss", val_metrics["traj_loss"], step=epoch)
                print(
                    f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_metrics['val_loss']:.8f}, "
                    f"Traj Loss: {val_metrics['traj_loss']:.8f}"
                )
            if is_best:
                fabric.save(best_ckpt_path, model.state_dict())
                if fabric.is_global_zero:
                    print(f"[Save] Best val_traj_loss: {val_metrics['traj_loss']:.8f}")
            barrier()


if __name__ == "__main__":
    main()
