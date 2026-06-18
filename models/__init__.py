import torch
import pytorch_lightning as pl
import losses
import math
import numpy as np
import wandb
from .grae import GRVAE
from .oracle import RenderModel
from .upsampler import Upsampler

dict_model = {
    'grvae': GRVAE,
    'render': RenderModel,
    'upsampler': Upsampler,
}

class ModelFactory:
    @staticmethod
    def create_model(config):
        """
        根据配置创建合适的模型
        
        自动检测：
        - 如果 loss 是 GAN 类型（包含 'Discriminator'），使用 LitModelGAN
        - 否则使用标准的 LitModel
        """
        if config.model.type in dict_model:
            loss_cls_name = config.loss.cls
            is_gan_loss = 'Discriminator' in loss_cls_name or 'GAN' in loss_cls_name
            
            if is_gan_loss:
                print(f"检测到 GAN Loss ({loss_cls_name})，使用 LitModelGAN")
                return LitModelGAN(dict_model[config.model.type], config)
            else:
                return LitModel(dict_model[config.model.type], config)
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        
class LitModel(pl.LightningModule):
    """统一的Lightning模型包装器"""
    def __init__(self, model_cls, config):
        super().__init__()
        if isinstance(model_cls, str):
            model_cls = dict_model[model_cls]
        self.model = model_cls(config)
        self.config = config
        self.save_hyperparameters(config)
        self.loss_fn = self._get_loss_fn(config)

        training_cfg = getattr(config, "training", {})
        self.train_log_on_step = training_cfg.get("train_log_on_step", True)
        self.train_log_on_epoch = training_cfg.get("train_log_on_epoch", True)
        self.val_log_on_step = training_cfg.get("val_log_on_step", False)
        self.val_log_on_epoch = training_cfg.get("val_log_on_epoch", True)
        self.sync_dist_logging = training_cfg.get("sync_dist_logging", True)
        
        # 图像记录相关配置
        self.log_images_every_n_epochs = getattr(config, 'log_images_every_n_epochs', 1)
        self.max_images_to_log = getattr(config, 'max_images_to_log', 16)
        
    def _get_loss_fn(self, config):
        # 根据配置动态获取损失函数
        loss_cls = getattr(losses, config.loss.cls)
        return loss_cls(**config.loss.kwargs)
    
    def forward(self, batch):
        return self.model(batch)
    
    def _compute_metrics(self, batch):
        # 可扩展的指标计算
        return self.loss_fn(batch)
    
    def _shared_step(self, batch):
        batch = self(batch)
        return self._compute_metrics(batch)
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        metrics = self._shared_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            on_step=self.train_log_on_step,
            on_epoch=self.train_log_on_epoch,
            sync_dist=self.sync_dist_logging,
        )
        return metrics['loss']
    
    def _log_reconstruction_images(self, batch, prefix="val"):
        """记录重建图像到wandb"""
        if self.current_epoch % self.log_images_every_n_epochs != 0:
            return
            
        try:
            # 获取原始图像和重建图像
            if 'image' in batch and 'pred' in batch:
                original_images = batch['image'][0]  # 取第一个样本 [L, C, H, W]
                reconstructed_images = batch['pred'][0]  # [L, C, H, W]
                
                # 限制记录的图像数量
                num_images = min(self.max_images_to_log, original_images.shape[0])
                original_images = original_images[:num_images]
                reconstructed_images = reconstructed_images[:num_images]
                
                # 将图像移动到CPU并转换为numpy
                original_images = original_images.detach().float().cpu()
                reconstructed_images = reconstructed_images.detach().float().cpu()
                
                # 归一化到 [0, 1] 范围（假设输入在 [-1, 1] 范围）
                # if original_images.min() < 0:
                original_images = (original_images + 1) / 2
                reconstructed_images = (reconstructed_images + 1) / 2
                
                # 确保图像在 [0, 1] 范围内
                original_images = torch.clamp(original_images, 0, 1)
                reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
                
                # 转换为numpy
                original_images_np = original_images.numpy()
                reconstructed_images_np = reconstructed_images.numpy()
                
                # 创建wandb图像列表
                wandb_images = []
                for t in range(num_images):
                    # 处理不同的图像格式
                    if original_images_np.shape[1] == 1:
                        # 灰度图像 [1, H, W] -> [H, W]
                        orig_img = original_images_np[t, 0]
                        recon_img = reconstructed_images_np[t, 0]
                    else:
                        # RGB图像 [C, H, W] -> [H, W, C]
                        orig_img = original_images_np[t].transpose(1, 2, 0)
                        recon_img = reconstructed_images_np[t].transpose(1, 2, 0)
                    
                    # 水平拼接原始图像和重建图像
                    combined_img = np.concatenate([orig_img, recon_img], axis=1)
                    wandb_images.append(
                        wandb.Image(
                            combined_img, 
                            caption=f"Frame {t}: Original (left) vs Reconstructed (right)"
                        )
                    )
                if self.logger is not None:
                    self.logger.experiment.log({
                        f"{prefix}/reconstruction_comparison": wandb_images,
                        f"{prefix}/epoch": self.current_epoch
                    })
                    print(f"Successfully logged {len(wandb_images)} reconstruction images to wandb")
                
        except Exception as e:
            # 如果图像记录失败，不影响训练
            print(f"Warning: Failed to log images to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        batch = self(batch)
        metrics = self._compute_metrics(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            on_step=self.val_log_on_step,
            on_epoch=self.val_log_on_epoch,
            sync_dist=self.sync_dist_logging,
        )
        
        # 记录重建图像（只在第一个batch）
        if batch_idx == 0:
            self._log_reconstruction_images(batch, prefix="val")
        
        return metrics['loss']

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.model.eval()
        metrics = self._shared_step(batch)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()})
        
        # 记录重建图像（只在第一个batch）
        if batch_idx == 0:
            self._log_reconstruction_images(batch, prefix="test")
            
        return metrics['loss']
    
    def configure_optimizers(self):
        # 动态配置优化器和学习率调度
        optimizer = getattr(torch.optim, self.config.optimizer.cls)(
            self.parameters(), **self.config.optimizer.kwargs
        )
        
        if hasattr(self.config, 'scheduler'):
            # 支持多种调度器类型
            if self.config.scheduler.cls == "warmup_cosine":
                # Warmup + Cosine Annealing
                warmup_steps = self.config.scheduler.kwargs.get('warmup_steps', 1000)
                max_steps = self.config.scheduler.kwargs.get('max_steps', 50000)
                
                def lr_lambda(step):
                    if step < warmup_steps:
                        # Linear warmup
                        return step / warmup_steps
                    else:
                        # Cosine annealing
                        progress = (step - warmup_steps) / (max_steps - warmup_steps)
                        return 0.5 * (1.0 + math.cos(math.pi * progress))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",  # 每步更新
                        "frequency": 1,
                        "name": "lr"
                    }
                }
            
            elif self.config.scheduler.cls == "warmup_linear":
                # Warmup + Linear Decay
                warmup_steps = self.config.scheduler.kwargs.get('warmup_steps', 1000)
                max_steps = self.config.scheduler.kwargs.get('max_steps', 50000)
                min_lr_ratio = self.config.scheduler.kwargs.get('min_lr_ratio', 0.01)
                
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        progress = (step - warmup_steps) / (max_steps - warmup_steps)
                        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "name": "lr"
                    }
                }
            
            elif self.config.scheduler.cls == "warmup_exponential":
                # Warmup + Exponential Decay
                warmup_steps = self.config.scheduler.kwargs.get('warmup_steps', 1000)
                decay_rate = self.config.scheduler.kwargs.get('decay_rate', 0.95)
                decay_steps = self.config.scheduler.kwargs.get('decay_steps', 1000)
                
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        return decay_rate ** ((step - warmup_steps) // decay_steps)
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "name": "lr"
                    }
                }
            
            else:
                # 使用标准的PyTorch调度器
                scheduler = getattr(torch.optim.lr_scheduler, self.config.scheduler.cls)(
                    optimizer, **self.config.scheduler.kwargs
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": self.config.scheduler.get('interval', 'epoch'),
                        "frequency": self.config.scheduler.get('frequency', 1),
                        "name": "lr"
                    }
                }
        
        return optimizer

class LitModelGAN(pl.LightningModule):
    """
    支持 GAN Loss 的 Lightning 模型包装器
    
    使用场景：当 loss 函数包含判别器时（如 MaskedLPIPSWithDiscriminator）
    
    特点：
    1. 使用手动优化（automatic_optimization=False）
    2. 分别优化生成器和判别器
    3. 支持判别器延迟启动（disc_start）
    """
    
    def __init__(self, model_cls, config):
        super().__init__()
        if isinstance(model_cls, str):
            model_cls = dict_model[model_cls]
        self.model = model_cls(config)
        self.config = config
        self.save_hyperparameters(config)
        self.loss_fn = self._get_loss_fn(config)

        training_cfg = getattr(config, "training", {})
        self.train_log_on_step = training_cfg.get("train_log_on_step", True)
        self.train_log_on_epoch = training_cfg.get("train_log_on_epoch", True)
        self.val_log_on_step = training_cfg.get("val_log_on_step", False)
        self.val_log_on_epoch = training_cfg.get("val_log_on_epoch", True)
        self.sync_dist_logging = training_cfg.get("sync_dist_logging", True)
        self.toggle_optimizers_for_efficiency = training_cfg.get(
            "toggle_optimizers_for_efficiency", False
        )
        
        # 关闭自动优化，手动控制优化过程
        self.automatic_optimization = False
        
        # 图像记录相关配置
        self.log_images_every_n_epochs = getattr(config, 'log_images_every_n_epochs', 1)
        self.max_images_to_log = getattr(config, 'max_images_to_log', 16)
        
    def _get_loss_fn(self, config):
        loss_cls = getattr(losses, config.loss.cls)
        return loss_cls(**config.loss.kwargs)
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """
        GAN 训练步骤：分别更新生成器和判别器
        """
        # 获取两个优化器
        opt_g, opt_d = self.optimizers()
        
        # 前向传播
        self.model.train()
        
        # ==================== 更新生成器 ====================
        if self.toggle_optimizers_for_efficiency:
            self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)
        batch = self(batch)
        
        loss_g, log_g = self.loss_fn(
            batch_data=batch,
            optimizer_idx=0,  # 生成器
            global_step=self.global_step,
            split="train"
        )
        
        # 手动反向传播和更新
        self.manual_backward(loss_g)
        opt_g.step()
        if self.toggle_optimizers_for_efficiency:
            self.untoggle_optimizer(opt_g)
        
        # ==================== 更新判别器 ====================
        if self.toggle_optimizers_for_efficiency:
            self.toggle_optimizer(opt_d)
        opt_d.zero_grad(set_to_none=True)
        
        loss_d, log_d = self.loss_fn(
            batch_data=batch,
            optimizer_idx=1,  # 判别器
            global_step=self.global_step,
            split="train"
        )
        
        # 手动反向传播和更新
        self.manual_backward(loss_d)
        opt_d.step()
        if self.toggle_optimizers_for_efficiency:
            self.untoggle_optimizer(opt_d)
        
        # ==================== 记录日志 ====================
        # 合并日志
        all_logs = {**log_g, **log_d}
        self.log_dict(
            all_logs,
            prog_bar=True,
            on_step=self.train_log_on_step,
            on_epoch=self.train_log_on_epoch,
            sync_dist=self.sync_dist_logging,
        )
        
        # 记录学习率（如果有调度器）
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if isinstance(schedulers, list):
                for i, sch in enumerate(schedulers):
                    if sch is not None:
                        sch.step()
                        self.log(f"lr/optimizer_{i}", sch.get_last_lr()[0])
            else:
                schedulers.step()
                self.log("lr/optimizer_0", schedulers.get_last_lr()[0])

        return loss_g
    
    def _log_reconstruction_images(self, batch, prefix="val"):
        """记录重建图像到wandb"""
        if self.current_epoch % self.log_images_every_n_epochs != 0:
            return
            
        try:
            if 'image' in batch and 'pred' in batch:
                original_images = batch['image'][0]
                reconstructed_images = batch['pred'][0]
                
                num_images = min(self.max_images_to_log, original_images.shape[0])
                original_images = original_images[:num_images]
                reconstructed_images = reconstructed_images[:num_images]
                
                original_images = original_images.detach().float().cpu()
                reconstructed_images = reconstructed_images.detach().float().cpu()
                
                original_images = (original_images + 1) / 2
                reconstructed_images = (reconstructed_images + 1) / 2
                
                original_images = torch.clamp(original_images, 0, 1)
                reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
                
                original_images_np = original_images.numpy()
                reconstructed_images_np = reconstructed_images.numpy()
                
                wandb_images = []
                for t in range(num_images):
                    if original_images_np.shape[1] == 1:
                        orig_img = original_images_np[t, 0]
                        recon_img = reconstructed_images_np[t, 0]
                    else:
                        orig_img = original_images_np[t].transpose(1, 2, 0)
                        recon_img = reconstructed_images_np[t].transpose(1, 2, 0)
                    
                    combined_img = np.concatenate([orig_img, recon_img], axis=1)
                    wandb_images.append(
                        wandb.Image(
                            combined_img, 
                            caption=f"Frame {t}: Original (left) vs Reconstructed (right)"
                        )
                    )
                
                if self.logger is not None:
                    self.logger.experiment.log({
                        f"{prefix}/reconstruction_comparison": wandb_images,
                        f"{prefix}/epoch": self.current_epoch
                    })
                    
        except Exception as e:
            print(f"Warning: Failed to log images: {e}")
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        验证步骤：只计算生成器损失
        """
        self.model.eval()
        batch = self(batch)
        
        # 只计算生成器损失用于验证
        _, log_g = self.loss_fn(
            batch_data=batch,
            optimizer_idx=0,
            global_step=self.global_step,
            split="val"
        )
        
        self.log_dict(
            log_g,
            prog_bar=True,
            on_step=self.val_log_on_step,
            on_epoch=self.val_log_on_epoch,
            sync_dist=self.sync_dist_logging,
        )
        
        # 记录重建图像
        if batch_idx == 0:
            self._log_reconstruction_images(batch, prefix="val")
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        self.model.eval()
        batch = self(batch)
        
        _, log_g = self.loss_fn(
            batch_data=batch,
            optimizer_idx=0,
            global_step=self.global_step,
            split="test"
        )
        
        self.log_dict(log_g)
        
        if batch_idx == 0:
            self._log_reconstruction_images(batch, prefix="test")
    
    def configure_optimizers(self):
        """
        配置两个优化器：一个用于生成器，一个用于判别器
        """
        # 生成器优化器（优化模型参数）
        opt_g = getattr(torch.optim, self.config.optimizer.cls)(
            self.model.parameters(), 
            **self.config.optimizer.kwargs
        )
        
        # 判别器优化器（优化 loss_fn 中的判别器参数）
        disc_lr = self.config.optimizer.kwargs.get('lr', 1e-3)
        if hasattr(self.config.optimizer, 'disc_lr'):
            disc_lr = self.config.optimizer.disc_lr
        
        opt_d = torch.optim.Adam(
            self.loss_fn.discriminator.parameters(),
            lr=disc_lr,
            betas=(0.5, 0.999)  # GAN 常用的 beta 值
        )
        
        # 如果有学习率调度器
        if hasattr(self.config, 'scheduler'):
            # 为生成器创建调度器
            scheduler_g = self._create_scheduler(opt_g)
            
            # 判别器通常使用固定学习率或相同的调度器
            scheduler_d = self._create_scheduler(opt_d)
            
            return [opt_g, opt_d], [scheduler_g, scheduler_d]
        
        return [opt_g, opt_d]
    
    def _create_scheduler(self, optimizer):
        """创建学习率调度器"""
        if self.config.scheduler.cls == "warmup_cosine":
            warmup_steps = self.config.scheduler.kwargs.get('warmup_steps', 1000)
            max_steps = self.config.scheduler.kwargs.get('max_steps', 50000)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (max_steps - warmup_steps)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        elif self.config.scheduler.cls == "warmup_linear":
            warmup_steps = self.config.scheduler.kwargs.get('warmup_steps', 1000)
            max_steps = self.config.scheduler.kwargs.get('max_steps', 50000)
            min_lr_ratio = self.config.scheduler.kwargs.get('min_lr_ratio', 0.01)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (max_steps - warmup_steps)
                    return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        else:
            # 使用标准的 PyTorch 调度器
            return getattr(torch.optim.lr_scheduler, self.config.scheduler.cls)(
                optimizer, **self.config.scheduler.kwargs
            )


__all__ = ["ModelFactory", "LitModel", "LitModelGAN"]
