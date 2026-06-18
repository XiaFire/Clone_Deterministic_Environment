import os
from pathlib import Path
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor

from models import ModelFactory, LitModel
from data import TrajectoryDataModule
from pytorch_lightning.utilities.model_summary import ModelSummary
import torch
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = False 
torch.backends.cuda.enable_mem_efficient_sdp(True) 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def load_config():
    """层级化配置加载系统"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", required=True, help="实验配置名称")
    parser.add_argument("-r", "--run", required=True, help="运行名称")
    parser.add_argument("overrides", nargs="*", help="配置覆盖项 (key=value)")
    args = parser.parse_args()
    
    base_config = OmegaConf.load("configs/base.yaml")
    exp_config = OmegaConf.load(f"configs/{args.exp}.yaml")
    config = OmegaConf.merge(base_config, exp_config)

    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, overrides)
    
    config.exp = args.exp
    config.run = args.run
    config.data.train_dir = os.path.expanduser(config.data.train_dir)
    config.data.val_dir = os.path.expanduser(config.data.val_dir)
    config.save_dir = os.path.join(config.save_dir, args.exp, args.run)
    return config

def main():
    config = load_config()
    dm = TrajectoryDataModule(config)
    model = ModelFactory.create_model(config)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=config.save_dir,
            monitor="train/total_loss",
            mode="min",
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False, 
            every_n_train_steps=300, 
            filename="model-{epoch:02d}-{step:04d}--{train/loss:.2f}",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        logger=None if 'debug' in config.run else WandbLogger(
            project=config.wandb.project,
            group=f"{config.model.type}-{config.exp}",
            name=config.run,
            config=OmegaConf.to_container(config, resolve=True)
        ),
        callbacks=callbacks,
        accelerator="auto",
        devices=config.gpus,
        log_every_n_steps=100,
        deterministic=config.training.deterministic,
        enable_model_summary=True,
        enable_progress_bar=True,
        overfit_batches=1 if config.debug else 0,
        # gradient_clip_val=0.5,
        # precision=32,
        precision='16-mixed',
        strategy='ddp_find_unused_parameters_true',
    )
    trainer.strategy.strict_loading = False
    trainer.logger.log_hyperparams(config)
    trainer.logger.log_graph(model)
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    ckpt_path = "last" if (Path(config.save_dir) / "last.ckpt").exists() else None
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
