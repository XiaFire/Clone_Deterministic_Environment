import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from taming.modules.losses.vqperceptual import *  
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.lpips import LPIPS
from torchmetrics import Accuracy
from typing import Dict, Union, Optional
import warnings

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_data):
        return {'loss': batch_data['loss'], 'accuracy': batch_data['accuracy'].item()}

class MSELoss(nn.Module):
    def __init__(self, target_key):
        super().__init__()
        self.target_key = target_key

    def forward(self, batch_data):
        return {'loss': F.mse_loss(batch_data['pred'], batch_data[self.target_key])}

class CrossEntropyLoss(nn.Module):
    def __init__(self, 
                 target_key: str = "targets", 
                 ignore_index: int = -100,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.target_key = target_key
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.accuracy = Accuracy(task='multiclass', num_classes=10_001, ignore_index=ignore_index)

    def forward(self, batch_data: dict) -> torch.Tensor:
        """
        Args:
            batch_data: 包含以下键值的字典
                - 'logits': 模型输出 (B, T, C)
                - [target_key]: 目标标签 (B, T)
        
        Returns:
            计算后的损失值
        """
        logits = batch_data['logits']
        targets = batch_data[self.target_key]
        
        # 维度处理：将序列维度展平
        # logits形状: (B*T, C)
        # targets形状: (B*T)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )
        with torch.no_grad():
            acc = self.accuracy(logits.argmax(-1), targets)
        return {'loss': loss, 'accuracy': acc.item()}

import torch
import torch.nn as nn
class MaskedCrossEntropy(nn.Module):
    def __init__(self, 
                 ignore_index: int = -100, 
                 reduction: str = 'mean',
                 eps: float = 0):
        """
        Args:
            ignore_index: 要忽略的目标索引
            reduction: 'mean' (默认) | 'sum' | 'none'
            eps: 防止除零的小量（仅在reduction='mean'时使用）
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

    def forward(self, batch):
        """
        Args:
        """
        input, target, mask = batch['pred'], batch['target'], batch['padding_mask']
        B, L, D = target.shape
        target = target.reshape(B, L*D)[:, 1:]
        input = input.reshape(B, L*D, -1)[:, :-1, :]
        mask = mask.view(B, L, 1).expand(-1, -1, D).reshape(B, L*D)[:, 1:]
        
        # 处理目标张量
        processed_target = torch.where(
            mask.bool(),
            target,
            torch.full_like(target, self.ignore_index)
        )

        # 展平输入和目标
        flat_input = input.reshape(-1, input.size(-1))  # [B*L*D, C]
        flat_target = processed_target.reshape(-1).long()    # [B*L*D]

        # 计算交叉熵
        ce_loss = F.cross_entropy(
            flat_input,
            flat_target,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        # 应用reduction
        if self.reduction == 'mean':
            valid = (flat_target != self.ignore_index).float().sum()
            loss = ce_loss.sum() / (valid + self.eps)
        elif self.reduction == 'sum':
            loss = ce_loss.sum()
        else:  # 'none'
            loss = ce_loss.view_as(target)

        return dict(loss=loss)

    def extra_repr(self) -> str:
        return f'ignore_index={self.ignore_index}, reduction={self.reduction}'

import torch
import torch.nn as nn
from typing import Dict, Union, Optional
import warnings


class BaseMaskedLoss(nn.Module):
    """
    Base class for masked losses with skip_size support.
    
    This base class provides common functionality for masked loss computation
    including input validation, tensor preparation, and skip_size handling.
    
    Args:
        target_key (str): Key to retrieve target tensor from batch_data
        ad_weight (float): Weight for adversarial loss component. Default: 1.0
        rec_weight (float): Weight for reconstruction loss component. Default: 1.0
        skip_size (int): Number of initial timesteps to skip due to insufficient context. Default: 0
        eps (float): Small epsilon value to prevent division by zero. Default: 1e-8
    """
    
    def __init__(
        self, 
        target_key: str, 
        ad_weight: float = 1.0, 
        rec_weight: float = 1.0, 
        skip_size: int = 0,
        eps: float = 1e-8
    ):
        super().__init__()
        
        # Input validation
        if not isinstance(target_key, str) or not target_key:
            raise ValueError("target_key must be a non-empty string")
        if ad_weight < 0 or rec_weight < 0:
            raise ValueError("Weights must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if skip_size < 0:
            raise ValueError("skip_size must be non-negative")
            
        self.target_key = target_key
        self.ad_weight = float(ad_weight)
        self.rec_weight = float(rec_weight)
        self.skip_size = int(skip_size)
        self.eps = eps
        
        print(f"{self.__class__.__name__} initialized - target_key: '{target_key}', "
              f"ad_weight: {self.ad_weight}, rec_weight: {self.rec_weight}, "
              f"skip_size: {self.skip_size}")
    
    def _validate_inputs(self, batch_data: Dict[str, torch.Tensor]) -> None:
        """Validate input batch data contains required keys and tensors have compatible shapes."""
        required_keys = ['pred', self.target_key, 'padding_mask']
        missing_keys = [key for key in required_keys if key not in batch_data]
        
        if missing_keys:
            raise KeyError(f"Missing required keys in batch_data: {missing_keys}")
        
        pred = batch_data['pred']
        target = batch_data[self.target_key]
        mask = batch_data['padding_mask']
        
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("pred and target must be torch.Tensor")
        
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(f"pred and target must have same batch size and timesteps. "
                           f"Got pred: {pred.shape[:2]}, target: {target.shape[:2]}")
        
        if mask.shape[:2] != pred.shape[:2]:
            raise ValueError(f"mask must have same batch size and timesteps as pred. "
                           f"Got mask: {mask.shape[:2]}, pred: {pred.shape[:2]}")
    
    def _prepare_tensors(self, batch_data: Dict[str, torch.Tensor]) -> tuple:
        """Prepare and reshape tensors for loss computation."""
        pred = batch_data['pred']
        target = batch_data[self.target_key]
        mask = batch_data['padding_mask']
        
        batch_size, timesteps = pred.size(0), pred.size(1)
        
        # Reshape to [batch_size, timesteps, -1]
        pred = pred.reshape(batch_size, timesteps, -1)
        target = target.reshape(batch_size, timesteps, -1)
        
        # Handle mask dimensions
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() > 2:
            # If mask has more than 2 dimensions, take mean over extra dims
            while mask.dim() > 2:
                mask = mask.mean(dim=-1)
        
        # Ensure mask is same dtype as predictions for computation
        mask = mask.to(dtype=pred.dtype, device=pred.device)
        
        return pred, target, mask
    
    def _apply_skip_size(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply skip_size by zeroing out the first skip_size timesteps in the mask.
        
        Args:
            mask: Binary mask [batch_size, timesteps]
            
        Returns:
            Modified mask with first skip_size timesteps set to 0
        """
        if self.skip_size <= 0:
            return mask
            
        # Create a copy of the mask to avoid in-place modification
        modified_mask = mask.clone()
        
        # Zero out the first skip_size timesteps
        if self.skip_size < mask.size(1):
            modified_mask[:, :self.skip_size] = 0
        else:
            # If skip_size >= total timesteps, zero out everything
            warnings.warn(f"skip_size ({self.skip_size}) >= total timesteps ({mask.size(1)}), "
                         "all timesteps will be skipped")
            modified_mask.fill_(0)
            
        return modified_mask

    def _compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked reconstruction loss with skip_size applied.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _compute_reconstruction_loss")
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute masked loss.
        
        Args:
            batch_data (dict): Dictionary containing:
                - 'pred': Predicted values [batch_size, timesteps, ...]
                - target_key: Target values [batch_size, timesteps, ...]
                - 'padding_mask': Binary mask [batch_size, timesteps] or [batch_size, timesteps, 1]
                - 'loss_ad' (optional): Adversarial loss component
                - 'loss_monitor' (optional): Additional monitoring losses
                
        Note:
            The first `skip_size` timesteps will be excluded from loss computation
            regardless of the padding_mask values, as they lack sufficient temporal context
            for reliable autoregressive prediction.
        
        Returns:
            dict: Dictionary containing:
                - 'loss': Total weighted loss
                - 'loss_recons': Reconstruction loss
                - 'loss_ad': Adversarial loss component
                - Additional keys from 'loss_monitor' if present
        """
        # Validate inputs
        self._validate_inputs(batch_data)
        
        # Prepare tensors
        pred, target, mask = self._prepare_tensors(batch_data)
        
        # Compute reconstruction loss (implemented by subclasses)
        recons_loss = self._compute_reconstruction_loss(pred, target, mask)
        
        # Get adversarial loss component
        loss_ad = batch_data.get('loss_ad', torch.tensor(0.0, device=pred.device, dtype=pred.dtype))
        
        # Compute total weighted loss
        total_loss = self.rec_weight * recons_loss + self.ad_weight * loss_ad
        
        # Prepare result dictionary
        result_dict = {
            'loss': total_loss,
            'loss_recons': recons_loss,
            'loss_ad': loss_ad
        }
        
        # Add monitoring losses if present
        if 'loss_monitor' in batch_data:
            monitor_loss = batch_data['loss_monitor']
            if isinstance(monitor_loss, dict):
                # Merge monitoring losses, avoiding key conflicts
                for key, value in monitor_loss.items():
                    if key not in result_dict:
                        result_dict[key] = value
                    else:
                        warnings.warn(f"Key '{key}' conflicts with existing result key, skipping")
            else:
                result_dict['loss_monitor'] = monitor_loss
        
        return result_dict
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'target_key={self.target_key}, ad_weight={self.ad_weight}, '
                f'rec_weight={self.rec_weight}, skip_size={self.skip_size}')


class MaskedMSELoss(BaseMaskedLoss):
    """
    Masked Mean Squared Error Loss with optional adversarial loss component and skip_size support.
    
    This loss function computes MSE loss only on valid (non-masked) timesteps,
    with support for additional adversarial loss weighting. The first `skip_size` timesteps
    are automatically excluded from loss computation to handle insufficient temporal context
    in autoregressive models with limited attention windows.
    
    Args:
        target_key (str): Key to retrieve target tensor from batch_data
        ad_weight (float): Weight for adversarial loss component. Default: 1.0
        rec_weight (float): Weight for reconstruction (MSE) loss component. Default: 1.0
        skip_size (int): Number of initial timesteps to skip due to insufficient context. Default: 0
        eps (float): Small epsilon value to prevent division by zero. Default: 1e-8
    """

    def __init__(
        self,
        target_key: str,
        ad_weight: float = 1.0,
        rec_weight: float = 1.0,
        skip_size: int = 0,
        eps: float = 1e-8,
        dyn_weight: float = 1.0,
        dyn_alpha: float = 10.0,
        dyn_norm: str = "mean",
    ):
        super().__init__(target_key, ad_weight, rec_weight, skip_size, eps)
        if dyn_weight < 0:
            raise ValueError("dyn_weight must be non-negative")
        if dyn_alpha < 0:
            raise ValueError("dyn_alpha must be non-negative")
        if dyn_norm not in ("mean", "max"):
            raise ValueError("dyn_norm must be 'mean' or 'max'")
        self.dyn_weight = float(dyn_weight)
        self.dyn_alpha = float(dyn_alpha)
        self.dyn_norm = dyn_norm

    def _compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked MSE loss with skip_size applied."""
        # Apply skip_size to mask
        effective_mask = self._apply_skip_size(mask)
        
        # Compute squared difference
        diff = (pred - target) ** 2
        
        # Average over feature dimension
        loss_per_timestep = diff.mean(dim=-1)  # [batch_size, timesteps]
        
        # Apply effective mask (which includes skip_size)
        masked_loss = loss_per_timestep * effective_mask
        
        # Compute valid timesteps count
        valid_timesteps = effective_mask.sum()
        
        # Prevent division by zero
        if valid_timesteps < self.eps:
            if self.skip_size > 0:
                warnings.warn(f"No valid timesteps found after applying skip_size={self.skip_size}, "
                             "returning zero loss")
            else:
                warnings.warn("No valid timesteps found in mask, returning zero loss")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Compute final MSE loss
        mse_loss = masked_loss.sum() / (valid_timesteps + self.eps)

        if self.dyn_weight <= 0 or pred.size(1) <= 1:
            return mse_loss

        dyn_loss = self._compute_dynamic_loss(pred, target, effective_mask)
        return mse_loss + self.dyn_weight * dyn_loss

    def _compute_dynamic_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        effective_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Use GT differences only; t >= 2 corresponds to index 1..T-1
        d = torch.abs(target[:, 1:] - target[:, :-1])
        valid_pairs = effective_mask[:, 1:] * effective_mask[:, :-1]

        if valid_pairs.sum() < self.eps:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        valid_pairs_exp = valid_pairs.unsqueeze(-1)

        if self.dyn_norm == "mean":
            denom = valid_pairs.sum(dim=1, keepdim=True) * d.size(-1)
            denom = denom.clamp_min(self.eps)
            mean_d = (d * valid_pairs_exp).sum(dim=(1, 2), keepdim=True) / denom.view(-1, 1, 1)
            d_norm = torch.where(
                valid_pairs_exp.bool(),
                d / mean_d.clamp_min(self.eps),
                torch.zeros_like(d),
            )
        else:  # "max"
            masked_d = d.masked_fill(valid_pairs_exp == 0, 0.0)
            max_d = masked_d.amax(dim=(1, 2), keepdim=True)
            d_norm = torch.where(
                valid_pairs_exp.bool(),
                d / max_d.clamp_min(self.eps),
                torch.zeros_like(d),
            )

        weights = (1.0 + self.dyn_alpha * d_norm).detach()

        dyn_diff = (pred[:, 1:] - target[:, 1:]) ** 2
        dyn_weighted = dyn_diff * weights * valid_pairs_exp
        dyn_loss_per_timestep = dyn_weighted.mean(dim=-1)
        dyn_masked = dyn_loss_per_timestep * valid_pairs

        return dyn_masked.sum() / (valid_pairs.sum() + self.eps)


class MaskedMAELoss(BaseMaskedLoss):
    """
    Masked Mean Absolute Error Loss with optional adversarial loss component and skip_size support.
    
    This loss function computes MAE loss only on valid (non-masked) timesteps,
    with support for additional adversarial loss weighting. The first `skip_size` timesteps
    are automatically excluded from loss computation to handle insufficient temporal context
    in autoregressive models with limited attention windows.
    
    Args:
        target_key (str): Key to retrieve target tensor from batch_data
        ad_weight (float): Weight for adversarial loss component. Default: 1.0
        rec_weight (float): Weight for reconstruction (MAE) loss component. Default: 1.0
        skip_size (int): Number of initial timesteps to skip due to insufficient context. Default: 0
        eps (float): Small epsilon value to prevent division by zero. Default: 1e-8
        
    Example:
        For a transformer with attention window of 10 frames, set skip_size=10 to ensure
        the model is only supervised on timesteps where it has access to sufficient context.
    """
    
    def _compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked MAE loss with skip_size applied."""
        # Apply skip_size to mask
        effective_mask = self._apply_skip_size(mask)
        
        # Compute absolute difference
        diff = torch.abs(pred - target)
        
        # Average over feature dimension
        loss_per_timestep = diff.mean(dim=-1)  # [batch_size, timesteps]
        
        # Apply effective mask (which includes skip_size)
        masked_loss = loss_per_timestep * effective_mask
        
        # Compute valid timesteps count
        valid_timesteps = effective_mask.sum()
        
        # Prevent division by zero
        if valid_timesteps < self.eps:
            if self.skip_size > 0:
                warnings.warn(f"No valid timesteps found after applying skip_size={self.skip_size}, "
                             "returning zero loss")
            else:
                warnings.warn("No valid timesteps found in mask, returning zero loss")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Compute final MAE loss
        mae_loss = masked_loss.sum() / (valid_timesteps + self.eps)
        
        return mae_loss


class WeightedMaskedMSELoss(BaseMaskedLoss):
    """
    Weighted Masked Mean Squared Error Loss with optional adversarial loss component and skip_size support.
    
    This loss function computes MSE loss only on valid (non-masked) timesteps,
    with weighted computation on the last dimension (feature dimension). The weights are applied
    to each feature channel independently. The first `skip_size` timesteps
    are automatically excluded from loss computation to handle insufficient temporal context
    in autoregressive models with limited attention windows.
    
    Args:
        target_key (str): Key to retrieve target tensor from batch_data
        weights (torch.Tensor or list): Weights for the last dimension. Shape should be [feature_dim]
                                       If None, uniform weights (all 1.0) will be used
        ad_weight (float): Weight for adversarial loss component. Default: 1.0
        rec_weight (float): Weight for reconstruction (MSE) loss component. Default: 1.0
        skip_size (int): Number of initial timesteps to skip due to insufficient context. Default: 0
        eps (float): Small epsilon value to prevent division by zero. Default: 1e-8
        
    Example:
        For input shape [B, L, 4], you can provide weights=[1.0, 2.0, 0.5, 1.5] to weight
        each of the 4 feature channels differently.
    """
    
    def __init__(
        self, 
        target_key: str, 
        weights: Union[torch.Tensor, list, None] = None,
        ad_weight: float = 1.0, 
        rec_weight: float = 1.0, 
        skip_size: int = 0,
        eps: float = 1e-8
    ):
        super().__init__(target_key, ad_weight, rec_weight, skip_size, eps)
        
        # Handle weights initialization
        if weights is None:
            # Default to uniform weights, will be set dynamically based on input shape
            self.weights = None
            self.feature_dim = None
        else:
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)
            elif not isinstance(weights, torch.Tensor):
                raise TypeError("weights must be torch.Tensor, list, or None")
            
            self.weights = weights.float()
            self.feature_dim = len(weights)
            
            if torch.any(self.weights < 0):
                raise ValueError("All weights must be non-negative")
        
        print(f"WeightedMaskedMSELoss initialized - target_key: '{target_key}', "
              f"weights: {self.weights}, feature_dim: {self.feature_dim}, "
              f"ad_weight: {self.ad_weight}, rec_weight: {self.rec_weight}, "
              f"skip_size: {self.skip_size}")
    
    def _initialize_weights_if_needed(self, feature_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize uniform weights if not provided."""
        if self.weights is None:
            self.weights = torch.ones(feature_dim, device=device, dtype=dtype)
            self.feature_dim = feature_dim
            print(f"Initialized uniform weights with feature_dim={feature_dim}")
        else:
            # Move weights to correct device and dtype if needed
            if self.weights.device != device or self.weights.dtype != dtype:
                self.weights = self.weights.to(device=device, dtype=dtype)
            
            # Validate feature dimension consistency
            if self.feature_dim != feature_dim:
                raise ValueError(f"Expected feature dimension {self.feature_dim} but got {feature_dim}")
    
    def _compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute weighted masked MSE loss with skip_size applied."""
        # Apply skip_size to mask
        effective_mask = self._apply_skip_size(mask)
        
        # Get feature dimension and initialize weights if needed
        feature_dim = pred.size(-1)
        self._initialize_weights_if_needed(feature_dim, pred.device, pred.dtype)
        
        # Compute squared difference
        diff = (pred - target) ** 2  # [batch_size, timesteps, feature_dim]
        
        # Apply weights to the last dimension
        # Broadcast weights: [feature_dim] -> [1, 1, feature_dim] for broadcasting
        weighted_diff = diff * self.weights.view(1, 1, -1)
        
        # Sum over feature dimension (weighted average)
        loss_per_timestep = weighted_diff.sum(dim=-1) / self.weights.sum()  # [batch_size, timesteps]
        
        # Apply effective mask (which includes skip_size)
        masked_loss = loss_per_timestep * effective_mask
        
        # Compute valid timesteps count
        valid_timesteps = effective_mask.sum()
        
        # Prevent division by zero
        if valid_timesteps < self.eps:
            if self.skip_size > 0:
                warnings.warn(f"No valid timesteps found after applying skip_size={self.skip_size}, "
                             "returning zero loss")
            else:
                warnings.warn("No valid timesteps found in mask, returning zero loss")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Compute final weighted MSE loss
        weighted_mse_loss = masked_loss.sum() / (valid_timesteps + self.eps)
        
        return weighted_mse_loss




class MaskedDoubleMSELoss(nn.Module):
    def __init__(self, ad_weight=0, rec_weight=1, pred_weight=1):
        super().__init__()
        self.ad_weight = ad_weight
        self.rec_weight = rec_weight
        self.pred_weight = pred_weight
        print(f"MaskedDoubleMSELoss initialized with ad_weight: {self.ad_weight}, rec_weight: {self.rec_weight}, pred_weight: {pred_weight}")

    def forward(self, batch_data: dict) -> torch.Tensor:
        pred = batch_data['pred']
        recons = batch_data['recons']
        pred_target = batch_data['targets']
        recons_target = batch_data['input_image']
        mask = batch_data['padding_mask']

        # 保证预测和目标的形状为 [B, T, D]
        bz, timesteps = pred.size(0), pred.size(1)
        pred = pred.reshape(bz, timesteps, -1)
        recons = recons.reshape(bz, timesteps, -1)
        pred_target = pred_target.reshape(bz, timesteps, -1)
        recons_target = recons_target.reshape(bz, timesteps, -1)
        
        # 计算均方差，每个元素的误差平方
        squared_diff = (pred - pred_target) ** 2  # 形状 [B, T, D]
        squared_diff_recons = (recons - recons_target) ** 2  # 形状 [B, T, D]
        
        # 对最后一个维度（feature 维度）取均值，得到每个时间步的误差
        loss_per_timestep = squared_diff.mean(dim=-1)  # 形状 [B, T]
        loss_per_timestep_recons = squared_diff_recons.mean(dim=-1)  # 形状 [B, T]
        # 如果 mask 原始形状为 [B, T]，则无需 squeeze；如果为 [B, T, 1]，则 squeeze 一下
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        
        # 保证 mask 与 loss_per_timestep 的数据类型一致
        mask = mask.to(dtype=loss_per_timestep.dtype)

        # 应用掩码，只考虑有效时间步
        masked_loss = loss_per_timestep * mask  # 形状 [B, T]
        masked_loss_recons = loss_per_timestep_recons * mask  # 形状 [B, T]
        valid_timesteps = mask.sum()

        loss_pred = masked_loss.sum() / valid_timesteps
        loss_recons = masked_loss_recons.sum() / valid_timesteps

        total_loss = (self.pred_weight * loss_pred + loss_recons) * self.rec_weight
        if 'loss_ad' in batch_data:
            total_loss += batch_data['loss_ad'] * self.ad_weight

        result_dict = dict(loss=total_loss, loss_pred=loss_pred, loss_recons=loss_recons, 
                    loss_ad=batch_data.get('loss_ad', total_loss-total_loss))
        if 'loss_monitor' in batch_data:
            if isinstance(batch_data['loss_monitor'], dict):
                result_dict.update(batch_data['loss_monitor'])
            else:
                result_dict['loss_monitor'] = batch_data['loss_monitor']  
        return result_dict


# =====================================================================================
# 辅助函数
# =====================================================================================

def adopt_weight(weight, global_step, threshold=0, value=0.):
    """根据训练步数动态调整权重"""
    if global_step < threshold:
        weight = value
    return weight


def l1(x, y):
    """L1 损失"""
    return torch.abs(x - y)


def l2(x, y):
    """L2 损失"""
    return torch.pow((x - y), 2)


class MaskedLPIPSLoss(nn.Module):
    """
    Masked reconstruction/perceptual autoencoder loss without any GAN discriminator.

    This intentionally accepts the discriminator-related keyword arguments used by
    MaskedLPIPSWithDiscriminator so configs can switch only loss.cls without
    rewriting every loss.kwargs override.
    """

    def __init__(
        self,
        target_key: str,
        disc_start: int = 50000,
        pixelloss_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_weight: float = 0.0,
        disc_factor: float = 0.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_ndf: int = 64,
        disc_loss: str = "hinge",
        pixel_loss: str = "l1",
        use_actnorm: bool = False,
        skip_size: int = 0,
        kl_weight: float = 1.0,
        slow_weight: float = 1.0,
        uniform_weight: float = 1.0,
        dyn_weight: float = 0.0,
        dyn_alpha: float = 10.0,
        dyn_norm: str = "mean",
    ):
        super().__init__()
        del disc_start, disc_weight, disc_factor, disc_num_layers, disc_in_channels
        del disc_ndf, disc_loss, use_actnorm

        assert pixel_loss in ["l1", "l2"], f"pixel_loss must be 'l1' or 'l2', got: {pixel_loss}"
        if dyn_weight < 0:
            raise ValueError("dyn_weight must be non-negative")
        if dyn_alpha < 0:
            raise ValueError("dyn_alpha must be non-negative")
        if dyn_norm not in ("mean", "max"):
            raise ValueError("dyn_norm must be 'mean' or 'max'")

        self.target_key = target_key
        self.skip_size = int(skip_size)
        self.kl_weight = float(kl_weight)
        self.slow_weight = float(slow_weight)
        self.uniform_weight = float(uniform_weight)
        self.dyn_weight = float(dyn_weight)
        self.dyn_alpha = float(dyn_alpha)
        self.dyn_norm = dyn_norm
        self.pixel_weight = float(pixelloss_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.pixel_loss = l1 if pixel_loss == "l1" else l2
        self.perceptual_loss = LPIPS().eval() if self.perceptual_weight > 0 else None

        print(f"{self.__class__.__name__} initialized without GAN discriminator:")
        print(f"  - pixel loss type: {pixel_loss}")
        print(f"  - pixel weight: {self.pixel_weight}, perceptual weight: {self.perceptual_weight}")
        print(f"  - kl weight: {self.kl_weight}, slow weight: {self.slow_weight}, uniform weight: {self.uniform_weight}")
        print(f"  - dyn weight: {self.dyn_weight}, dyn alpha: {self.dyn_alpha}, dyn norm: {self.dyn_norm}")

    def _apply_skip_size(self, mask: torch.Tensor) -> torch.Tensor:
        if self.skip_size <= 0:
            return mask

        modified_mask = mask.clone()
        if self.skip_size < mask.size(1):
            modified_mask[:, : self.skip_size] = 0
        else:
            warnings.warn(
                f"skip_size ({self.skip_size}) >= total timesteps ({mask.size(1)}), "
                "all timesteps will be skipped"
            )
            modified_mask.fill_(0)
        return modified_mask

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = batch_data["pred"]
        target = batch_data[self.target_key]
        mask = batch_data["padding_mask"]

        B, T = mask.shape
        effective_mask = self._apply_skip_size(mask)

        pred_flat = pred.reshape(B * T, *pred.shape[2:])
        target_flat = target.reshape(B * T, *target.shape[2:])
        mask_flat = effective_mask.reshape(B * T)
        valid_indices = mask_flat > 0

        if valid_indices.sum() == 0:
            zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return {
                "loss": zero,
                "total_loss": zero,
                "nll_loss": zero,
                "rec_loss": zero,
                "p_loss": zero,
                "dyn_loss": zero,
                "kl_loss": zero,
                "slow_loss": zero,
                "uniform_loss": zero,
            }

        pred_valid = pred_flat[valid_indices]
        target_valid = target_flat[valid_indices]

        rec_loss = self.pixel_loss(pred_valid.contiguous(), target_valid.contiguous()).mean()

        if self.perceptual_loss is not None:
            p_loss = self.perceptual_loss(pred_valid.contiguous(), target_valid.contiguous()).mean()
        else:
            p_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        dyn_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.dyn_weight > 0 and T > 1:
            dyn_loss = self._compute_dynamic_loss(pred, target, effective_mask)

        kl_loss = batch_data["kl_loss"]
        slow_loss = batch_data["slow_loss"]
        uniform_loss = batch_data["uniform_loss"]

        nll_loss = (
            self.pixel_weight * rec_loss
            + self.perceptual_weight * p_loss
            + self.dyn_weight * dyn_loss
        )
        loss = (
            nll_loss
            + self.kl_weight * kl_loss
            + self.slow_weight * slow_loss
            + self.uniform_weight * uniform_loss
        )

        return {
            "loss": loss,
            "total_loss": loss.detach(),
            "nll_loss": nll_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "p_loss": p_loss.detach(),
            "dyn_loss": dyn_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "slow_loss": slow_loss.detach(),
            "uniform_loss": uniform_loss.detach(),
        }

    def _compute_dynamic_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        effective_mask: torch.Tensor,
    ) -> torch.Tensor:
        d = torch.abs(target[:, 1:] - target[:, :-1])
        valid_pairs = effective_mask[:, 1:] * effective_mask[:, :-1]

        if valid_pairs.sum() < 1e-8:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        valid_pairs_exp = valid_pairs[:, :, None, None, None]
        if self.dyn_norm == "mean":
            denom = valid_pairs.sum(dim=1, keepdim=True) * (d.size(-1) * d.size(-2) * d.size(-3))
            denom = denom.clamp_min(1e-8)
            mean_d = (d * valid_pairs_exp).sum(dim=(1, 2, 3, 4), keepdim=True) / denom.view(-1, 1, 1, 1, 1)
            d_norm = torch.where(
                valid_pairs_exp.bool(),
                d / mean_d.clamp_min(1e-8),
                torch.zeros_like(d),
            )
        else:
            masked_d = d.masked_fill(valid_pairs_exp == 0, 0.0)
            max_d = masked_d.amax(dim=(1, 2, 3, 4), keepdim=True)
            d_norm = torch.where(
                valid_pairs_exp.bool(),
                d / max_d.clamp_min(1e-8),
                torch.zeros_like(d),
            )

        weights = (1.0 + self.dyn_alpha * d_norm).detach()
        dyn_diff = (pred[:, 1:] - target[:, 1:]) ** 2
        dyn_weighted = dyn_diff * weights * valid_pairs_exp
        dyn_loss_per_timestep = dyn_weighted.mean(dim=(2, 3, 4))
        dyn_masked = dyn_loss_per_timestep * valid_pairs
        return dyn_masked.sum() / (valid_pairs.sum() + 1e-8)


# =====================================================================================
# Masked LPIPS with Discriminator Loss - 带掩码支持的 LPIPS 和 GAN 损失
# =====================================================================================

class MaskedLPIPSWithDiscriminator(nn.Module):
    """
    带掩码支持的 LPIPS 和判别器损失函数（使用固定权重策略）。
    
    这个损失函数结合了：
    1. 像素级重建损失 (L1 或 L2)
    2. 感知损失 (LPIPS)
    3. 对抗损失 (GAN Loss with Discriminator)
    
    适用于序列数据（如视频），支持 padding mask。
    
    训练策略：
    - 前期（步数 < disc_start）：只训练重建损失
    - 后期（步数 >= disc_start）：引入判别器，使用固定权重
    
    Args:
        target_key (str): batch_data 中目标数据的键名
        disc_start (int): 开始训练判别器的步数，默认 50000
        pixelloss_weight (float): 像素损失权重，默认 1.0
        perceptual_weight (float): 感知损失（LPIPS）权重，默认 1.0
        disc_weight (float): 判别器损失的固定权重，默认 0.5
        disc_factor (float): 判别器损失因子，默认 1.0
        disc_num_layers (int): 判别器网络层数，默认 3
        disc_in_channels (int): 判别器输入通道数，默认 3
        disc_ndf (int): 判别器特征维度，默认 64
        disc_loss (str): 判别器损失类型，"hinge" 或 "vanilla"，默认 "hinge"
        pixel_loss (str): 像素损失类型，"l1" 或 "l2"，默认 "l1"
        use_actnorm (bool): 判别器是否使用 actnorm，默认 False
        skip_size (int): 跳过前 skip_size 个时间步，默认 0
    """
    
    def __init__(
        self, 
        target_key: str,
        disc_start: int = 50000,
        pixelloss_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_weight: float = 0.5,
        disc_factor: float = 1.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_ndf: int = 64,
        disc_loss: str = "hinge",
        pixel_loss: str = "l1",
        use_actnorm: bool = False,
        skip_size: int = 0,
        kl_weight: float = 1.0,
        slow_weight: float = 1.0,
        uniform_weight: float = 1.0,
        dyn_weight: float = 0.0,
        dyn_alpha: float = 10.0,
        dyn_norm: str = "mean",
    ):
        super().__init__()
        
        assert disc_loss in ["hinge", "vanilla"], f"disc_loss 必须是 'hinge' 或 'vanilla'，得到: {disc_loss}"
        assert pixel_loss in ["l1", "l2"], f"pixel_loss 必须是 'l1' 或 'l2'，得到: {pixel_loss}"
        
        self.target_key = target_key
        self.skip_size = skip_size
        self.kl_weight = kl_weight
        self.slow_weight = slow_weight
        self.uniform_weight = uniform_weight
        if dyn_weight < 0:
            raise ValueError("dyn_weight must be non-negative")
        if dyn_alpha < 0:
            raise ValueError("dyn_alpha must be non-negative")
        if dyn_norm not in ("mean", "max"):
            raise ValueError("dyn_norm must be 'mean' or 'max'")
        self.dyn_weight = float(dyn_weight)
        self.dyn_alpha = float(dyn_alpha)
        self.dyn_norm = dyn_norm
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = disc_weight  # 固定权重
        self.disc_factor = disc_factor
        self.discriminator_iter_start = disc_start
        
        # 像素损失
        self.pixel_loss = l1 if pixel_loss == "l1" else l2
        
        # LPIPS 感知损失
        print(f"{self.__class__.__name__}: 使用 LPIPS 感知损失")
        self.perceptual_loss = LPIPS().eval()
        
        # 判别器
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        ).apply(weights_init)
        
        # 判别器损失函数
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        
        print(f"{self.__class__.__name__} 初始化完成（使用固定权重策略）:")
        print(f"  - 判别器损失类型: {disc_loss}")
        print(f"  - 像素损失类型: {pixel_loss}")
        print(f"  - 判别器开始训练步数: {disc_start}")
        print(f"  - 判别器固定权重: {disc_weight}")
        print(f"  - 跳过的时间步数: {skip_size}")
        print(f"  - 动态损失权重: {self.dyn_weight}, 动态 alpha: {self.dyn_alpha}, 归一化: {self.dyn_norm}")

    def _apply_skip_size(self, mask: torch.Tensor) -> torch.Tensor:
        """应用 skip_size，将前 skip_size 个时间步的 mask 设为 0"""
        if self.skip_size <= 0:
            return mask
            
        modified_mask = mask.clone()
        
        if self.skip_size < mask.size(1):
            modified_mask[:, :self.skip_size] = 0
        else:
            warnings.warn(
                f"skip_size ({self.skip_size}) >= 总时间步数 ({mask.size(1)}), "
                "所有时间步都将被跳过"
            )
            modified_mask.fill_(0)
            
        return modified_mask

    def forward(
        self, 
        batch_data: Dict[str, torch.Tensor], 
        optimizer_idx: int,
        global_step: int, 
        split: str = "train"
    ) -> tuple:
        """
        前向传播（使用固定权重策略）
        
        Args:
            batch_data: 包含以下键的字典:
                - 'pred': 预测结果 [B, T, C, H, W]
                - target_key: 目标数据 [B, T, C, H, W]
                - 'padding_mask': 掩码 [B, T]
            optimizer_idx: 优化器索引 (0: 生成器, 1: 判别器)
            global_step: 全局训练步数
            split: 数据集划分名称 ("train" 或 "val")
            
        Returns:
            loss: 损失值
            log: 日志字典
        """
        pred = batch_data['pred']
        # print(pred.shape)
        target = batch_data[self.target_key]
        mask = batch_data['padding_mask']
        
        # 获取形状信息
        B, T = mask.shape
        
        # 应用 skip_size
        effective_mask = self._apply_skip_size(mask)  # [B, T]
        
        # 将 [B, T, C, H, W] reshape 为 [B*T, C, H, W] 以便使用 LPIPS 和 Discriminator
        pred_flat = pred.reshape(B * T, *pred.shape[2:])
        target_flat = target.reshape(B * T, *target.shape[2:])
        mask_flat = effective_mask.reshape(B * T)  # [B*T]
        
        # 只在有效（非 padding）的位置计算损失
        valid_indices = mask_flat > 0
        
        if valid_indices.sum() == 0:
            # 如果没有有效样本，返回零损失
            zero_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            log = {
                f"{split}/total_loss": zero_loss,
                f"{split}/rec_loss": zero_loss,
                f"{split}/p_loss": zero_loss,
            }
            return zero_loss, log
        
        # 提取有效样本
        pred_valid = pred_flat[valid_indices]
        target_valid = target_flat[valid_indices]
        
        # 计算像素级重建损失
        rec_loss = self.pixel_loss(pred_valid.contiguous(), target_valid.contiguous())
        rec_loss = rec_loss.mean()
        dyn_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.dyn_weight > 0 and T > 1:
            dyn_loss = self._compute_dynamic_loss(pred, target, effective_mask)
        
        # 计算感知损失 (LPIPS)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(pred_valid.contiguous(), target_valid.contiguous())
            p_loss = p_loss.mean()
            nll_loss = rec_loss + self.perceptual_weight * p_loss + self.dyn_weight * dyn_loss
        else:
            p_loss = torch.tensor(0.0, device=pred.device)
            nll_loss = rec_loss + self.dyn_weight * dyn_loss
        
        # ==================== 生成器更新 (optimizer_idx == 0) ====================
        if optimizer_idx == 0:
            # 计算对抗损失
            logits_fake = self.discriminator(pred_valid.contiguous())
            g_loss = -torch.mean(logits_fake)
            scalar_device = pred.device
            scalar_dtype = pred.dtype
            
            # 根据训练步数决定是否使用判别器
            disc_factor = adopt_weight(
                self.disc_factor, 
                global_step, 
                threshold=self.discriminator_iter_start
            )
            
            # 使用固定权重
            d_weight = self.discriminator_weight            

            kl_loss = batch_data['kl_loss']
            slow_loss = batch_data['slow_loss']
            uniform_loss = batch_data['uniform_loss']
            # 总损失 = 重建损失 + 固定权重 * 判别器因子 * 对抗损失
            loss = nll_loss + d_weight * disc_factor * g_loss + self.kl_weight * kl_loss + self.slow_weight * slow_loss + self.uniform_weight * uniform_loss

            
            # 日志
            log = {
                f"{split}/total_loss": loss.clone().detach(),
                f"{split}/nll_loss": nll_loss.detach(),
                f"{split}/rec_loss": rec_loss.detach(),
                f"{split}/p_loss": p_loss.detach() if isinstance(p_loss, torch.Tensor) else p_loss,
                f"{split}/dyn_loss": dyn_loss.detach(),
                f"{split}/d_weight": torch.tensor(
                    d_weight, device=scalar_device, dtype=scalar_dtype
                ),
                f"{split}/disc_factor": torch.tensor(
                    disc_factor, device=scalar_device, dtype=scalar_dtype
                ),
                f"{split}/g_loss": g_loss.detach(),
                f"{split}/kl_loss": kl_loss.detach(),
                f"{split}/slow_loss": slow_loss.detach(),
                f"{split}/uniform_loss": uniform_loss.detach(),
            }
            
            return loss, log
        
        # ==================== 判别器更新 (optimizer_idx == 1) ====================
        if optimizer_idx == 1:
            # 真实样本和假样本的判别
            logits_real = self.discriminator(target_valid.contiguous().detach())
            logits_fake = self.discriminator(pred_valid.contiguous().detach())
            
            # 根据训练步数调整判别器权重
            disc_factor = adopt_weight(
                self.disc_factor, 
                global_step, 
                threshold=self.discriminator_iter_start
            )
            
            # 判别器损失
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            
            # 日志
            log = {
                f"{split}/disc_loss": d_loss.clone().detach(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            
            return d_loss, log

    def _compute_dynamic_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        effective_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Use GT differences only; t >= 2 corresponds to index 1..T-1
        d = torch.abs(target[:, 1:] - target[:, :-1])  # [B, T-1, C, H, W]
        valid_pairs = effective_mask[:, 1:] * effective_mask[:, :-1]  # [B, T-1]

        if valid_pairs.sum() < 1e-8:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        valid_pairs_exp = valid_pairs[:, :, None, None, None]
        if self.dyn_norm == "mean":
            denom = valid_pairs.sum(dim=1, keepdim=True) * (d.size(-1) * d.size(-2) * d.size(-3))
            denom = denom.clamp_min(1e-8)
            mean_d = (d * valid_pairs_exp).sum(dim=(1, 2, 3, 4), keepdim=True) / denom.view(-1, 1, 1, 1, 1)
            d_norm = torch.where(
                valid_pairs_exp.bool(),
                d / mean_d.clamp_min(1e-8),
                torch.zeros_like(d),
            )
        else:  # "max"
            masked_d = d.masked_fill(valid_pairs_exp == 0, 0.0)
            max_d = masked_d.amax(dim=(1, 2, 3, 4), keepdim=True)
            d_norm = torch.where(
                valid_pairs_exp.bool(),
                d / max_d.clamp_min(1e-8),
                torch.zeros_like(d),
            )

        weights = (1.0 + self.dyn_alpha * d_norm).detach()
        dyn_diff = (pred[:, 1:] - target[:, 1:]) ** 2
        dyn_weighted = dyn_diff * weights * valid_pairs_exp
        denom = valid_pairs.sum() * (d.size(-1) * d.size(-2) * d.size(-3))
        return dyn_weighted.sum() / (denom + 1e-8)
