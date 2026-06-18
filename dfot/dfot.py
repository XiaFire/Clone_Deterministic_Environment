import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Union, Sequence
import numpy as np
from functools import partial
from einops import rearrange, reduce, repeat
from torch import Tensor
from typing import Tuple, Union, Sequence
from typing import Optional
from torch.types import _size
from .diffusion import DiscreteDiffusion


from models.transformer import DiTBlock

def bernoulli_tensor(
    size: _size,
    p: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Generate a tensor of the given size,
    where each element is sampled from a Bernoulli distribution with probability `p`.
    """
    return torch.bernoulli(torch.full(size, p, device=device), generator=generator)


class DFOT(nn.Module):
    def __init__(self, cfg: DictConfig, decoder=None):
        super().__init__()
        self.cfg = cfg
        self.x_shape = list(cfg.x_shape)
        self.decoder = decoder
        # Add necessary attributes
        self.max_tokens = cfg.get('max_tokens', 100)
        self.timesteps = cfg.diffusion.timesteps
        self.generator = None  # Will be set during sampling if needed
        self.n_context_tokens = cfg.get('n_context_tokens', 20)
        
        self._build_model()

    @property
    def device(self):
        return next(self.parameters()).device

    def _build_model(self):
        # 1. Diffusion model
        if self.cfg.diffusion.is_continuous:
            raise NotImplementedError(
                "This minimal Maze9 release supports discrete diffusion only."
            )
        diffusion_cls = DiscreteDiffusion

        if self.cfg.compile:
            # NOTE: this compiling is only for training speedup
            if self.cfg.compile == "true_without_ddp_optimizer":
                # NOTE: `cfg.compile` should be set to this value when using `torch.compile` with DDP & Gradient Checkpointing
                # Otherwise, torch.compile will raise an error.
                # Reference: https://github.com/pytorch/pytorch/issues/104674
                # pylint: disable=protected-access
                torch._dynamo.config.optimize_ddp = False
            assert (
                self.cfg.diffusion.is_continuous
            ), "`torch.compile` is only verified for continuous-time diffusion models. To use it for discrete-time models, it should be tested, including # graph breaks"

        self.diffusion_model = torch.compile(
            diffusion_cls(
                cfg=self.cfg.diffusion,
                backbone_cfg=self.cfg.backbone,
                x_shape=self.x_shape,
                max_tokens=self.max_tokens,
                external_cond_dim=self.cfg.get('external_cond_dim', 0),
                decoder=self.decoder,
            ),
            disable=not self.cfg.compile,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    def register_data_mean_std(
        self,
        mean: Union[str, float, Sequence],
        std: Union[str, float, Sequence],
        namespace: str = "data",
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))

    def _get_training_noise_levels(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Generate random noise levels for training."""
        batch_size, n_tokens, *_ = xs.shape

        # random function different for continuous and discrete diffusion
        rand_fn = partial(
            *(
                (torch.rand,)
                if self.cfg.diffusion.is_continuous
                else (torch.randint, 0, self.timesteps)
            ),
            device=xs.device,
            generator=self.generator,
        )

        # baseline training (SD: fixed_context, BD: variable_context)
        context_mask = None
        if self.cfg.variable_context.enabled:
            assert (
                not self.cfg.fixed_context.enabled
            ), "Cannot use both fixed and variable context"
            context_mask = bernoulli_tensor(
                (batch_size, n_tokens),
                self.cfg.variable_context.prob,
                device=self.device,
                generator=self.generator,
            ).bool()
        elif self.cfg.fixed_context.enabled:
            context_indices = self.cfg.fixed_context.indices or list(
                range(self.n_context_tokens)
            )
            context_mask = torch.zeros(
                (batch_size, n_tokens), dtype=torch.bool, device=xs.device
            )
            context_mask[:, context_indices] = True

        match self.cfg.noise_level:
            case "random_independent":  # independent noise levels (Diffusion Forcing)
                noise_levels = rand_fn((batch_size, n_tokens))
            case "random_uniform":  # uniform noise levels (Typical Video Diffusion)
                noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)

        if self.cfg.uniform_future.enabled:  # simplified training (Appendix A.5)
            noise_levels[:, self.n_context_tokens :] = rand_fn((batch_size, 1)).repeat(
                1, n_tokens - self.n_context_tokens
            )

        # treat frames that are not available as "full noise"
        noise_levels = torch.where(
            reduce(masks.bool(), "b t ... -> b t", torch.any),
            noise_levels,
            torch.full_like(
                noise_levels,
                1 if self.cfg.diffusion.is_continuous else self.timesteps - 1,
            ),
        )

        if context_mask is not None:
            # binary dropout training to enable guidance
            dropout = (
                (
                    self.cfg.variable_context
                    if self.cfg.variable_context.enabled
                    else self.cfg.fixed_context
                ).dropout
                if self.training
                else 0.0
            )
            context_noise_levels = bernoulli_tensor(
                (batch_size, 1),
                dropout,
                device=xs.device,
                generator=self.generator,
            )
            if not self.cfg.diffusion.is_continuous:
                context_noise_levels = context_noise_levels.long() * (
                    self.timesteps - 1
                )
            noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            # modify masks to exclude context frames from loss computation
            context_mask = rearrange(
                context_mask, "b t -> b t" + " 1" * len(masks.shape[2:])
            )
            masks = torch.where(context_mask, False, masks)

        return noise_levels, masks

    def _reweight_loss(self, loss, weight=None):
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()
  
    def forward(self, xs: Tensor, conditions: Tensor, masks: Tensor) -> Tensor:
        """
        Forward pass of the DFOT model.

        Args:
            x: The input tensor of shape (batch_size, n_tokens, *x_shape).
            conditions: The conditions tensor of shape (batch_size, n_tokens, *conditions_shape).
        """
        xs = self._normalize_x(xs)

        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        xs_pred, loss = self.diffusion_model(
            xs,
            conditions,
            k=noise_levels,
        )
        loss = self._reweight_loss(loss, masks)

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def sample_3d(self, xs: Tensor, actions: Tensor) -> Tensor:
        """
        Sample function for DFOT model.
        
        Args:
            xs: Clean images of shape [B, T, C, H, W]
            actions: Actions of shape [B, T+1, action_dim]
            
        Returns:
            xs_pred: Predicted images of shape [B, T, C, H, W]
        """
        B, T, C, H, W = xs.shape
        device = xs.device
        
        # Normalize input
        xs = self._normalize_x(xs)
        
        # Create x_input [B, T+1, C, H, W] with last frame as noise
        noise_frame = torch.randn(B, 1, C, H, W, device=device)
        if hasattr(self.cfg.diffusion, 'clip_noise'):
            noise_frame = torch.clamp(noise_frame, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        x_input = torch.cat([xs, noise_frame], dim=1)  # [B, T+1, C, H, W]
        
        # Create context mask: 1 for clean frames (first T), 0 for noise frame (T+1)
        context_mask = torch.ones(B, T+1, dtype=torch.long, device=device)
        context_mask[:, -1] = 0  # Last frame needs to be generated
        
        # Generate scheduling matrix for sampling
        scheduling_matrix = self._generate_scheduling_matrix(T+1, actions)
        
        # Main sampling loop
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]
            
            # Only update the frames that need to be generated
            x_input_prev = x_input.clone()
            
            # Sample step for diffusion model
            x_input = self.diffusion_model.sample_step(
                x_input,
                from_noise_levels,
                to_noise_levels,
                actions,  # Pass processed actions as conditions
                None,        # No conditions mask
                guidance_fn=None,  # No guidance as you mentioned
            )
            
            # Keep the context frames unchanged (only update frames that should be generated)
            context_mask_expanded = repeat(context_mask, 'b t -> b t c h w', c=C, h=H, w=W)
            x_input = torch.where(context_mask_expanded == 1, x_input_prev, x_input)
        
        # Extract the predicted frames
        xs_pred = x_input  # [B, T+1, C, H, W] 
        
        # Unnormalize output
        xs_pred = self._unnormalize_x(xs_pred)
        
        return xs_pred
    
    def sample(self, xs: Tensor, actions: Tensor) -> Tensor:
        """
        Sample function for DFOT model.
        
        Args:
            xs: Clean images of shape [B, T, C, H, W]
            actions: Actions of shape [B, T+1, action_dim]
            
        Returns:
            xs_pred: Predicted images of shape [B, T, C, H, W]
        """
        B, T, D = xs.shape
        device = xs.device
        
        # Normalize input
        xs = self._normalize_x(xs)
        
        # Create x_input [B, T+1, C, H, W] with last frame as noise
        noise_frame = torch.randn(B, 1, D, device=device)
        if hasattr(self.cfg.diffusion, 'clip_noise'):
            noise_frame = torch.clamp(noise_frame, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        x_input = torch.cat([xs, noise_frame], dim=1)  # [B, T+1, C, H, W]
        
        # Create context mask: 1 for clean frames (first T), 0 for noise frame (T+1)
        context_mask = torch.ones(B, T+1, dtype=torch.long, device=device)
        context_mask[:, -1] = 0  # Last frame needs to be generated
        
        # Generate scheduling matrix for sampling
        scheduling_matrix = self._generate_scheduling_matrix(T+1, actions)
        
        # Main sampling loop
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]
            
            # Only update the frames that need to be generated
            x_input_prev = x_input.clone()
            
            # Sample step for diffusion model
            x_input = self.diffusion_model.sample_step(
                x_input,
                from_noise_levels,
                to_noise_levels,
                actions,  # Pass processed actions as conditions
                None,        # No conditions mask
                guidance_fn=None,  # No guidance as you mentioned
            )
            
            # Keep the context frames unchanged (only update frames that should be generated)
            context_mask_expanded = repeat(context_mask, 'b t -> b t d', d=D)
            x_input = torch.where(context_mask_expanded == 1, x_input_prev, x_input)
        
        # Extract the predicted frames
        xs_pred = x_input  # [B, T+1, C, H, W] 
        
        # Unnormalize output
        xs_pred = self._unnormalize_x(xs_pred)
        
        return xs_pred
    
    
    def sample_ldm(self, xs: Tensor, actions: Tensor, guidance_scale: float = 1) -> Tensor:
        """
        Sample function for DFOT model.
        
        Args:
            xs: Clean images of shape [B, T, C, H, W]
            actions: Actions of shape [B, T+1, action_dim]
            
        Returns:
            xs_pred: Predicted images of shape [B, T, C, H, W]
        """
        B, T, D = xs.shape
        device = xs.device
        
        # Normalize input
        xs = self._normalize_x(xs)
        
        # Create x_input [B, T+1, C, H, W] with last frame as noise
        noise_frame = torch.randn(B, 1, D, device=device)
        if hasattr(self.cfg.diffusion, 'clip_noise'):
            noise_frame = torch.clamp(noise_frame, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        x_input = torch.cat([xs, noise_frame], dim=1)  # [B, T+1, C, H, W]
        
        # Create context mask: 1 for clean frames (first T), 0 for noise frame (T+1)
        context_mask = torch.ones(B, T+1, dtype=torch.long, device=device)
        context_mask[:, -1] = 0  # Last frame needs to be generated
        context = x_input.clone()
        # Generate scheduling matrix for sampling
        scheduling_matrix = self._generate_scheduling_matrix(T+1, actions, is_ldm=True)
        
        # Main sampling loop
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]
            
            # Only update the frames that need to be generated
            x_input_prev = x_input.clone()
            def composed_guidance_fn(
                xk: torch.Tensor,
                pred_x0: torch.Tensor,
                alpha_cumprod: torch.Tensor,
            ) -> torch.Tensor:
                loss = (
                    F.mse_loss(pred_x0, context, reduction="none")
                    * alpha_cumprod.sqrt()
                )
                _context_mask = rearrange(
                    context_mask.bool(),
                    "b t -> b t 1",
                )
                # scale inversely proportional to the number of context frames
                loss = torch.sum(
                    loss
                    * _context_mask
                    / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                )
                likelihood = -guidance_scale * 0.5 * loss
                return likelihood
            # Sample step for diffusion model
            x_input = self.diffusion_model.sample_step(
                x_input,
                from_noise_levels,
                to_noise_levels,
                actions,  # Pass processed actions as conditions
                None,        # No conditions mask
                guidance_fn=composed_guidance_fn,
            )
            
            # Keep the context frames unchanged (only update frames that should be generated)
            context_mask_expanded = repeat(context_mask, 'b t -> b t d', d=D)
            x_input = torch.where(context_mask_expanded == 1, x_input_prev, x_input)
        
        # Extract the predicted frames
        xs_pred = x_input  # [B, T+1, D] 
        
        # Unnormalize output
        xs_pred = self._unnormalize_x(xs_pred)
        
        return xs_pred
    
    def _generate_scheduling_matrix(self, length: int, actions: Tensor, is_ldm=False) -> Tensor:
        """
        Generate scheduling matrix for the sampling process.
        
        Args:
            length: Length of the sequence (T+1)
            actions: Actions tensor for potential conditioning
            
        Returns:
            scheduling_matrix: Tensor of shape [num_steps+1, B, length]
        """
        B = actions.shape[0]
        device = actions.device
        
        # Get sampling timesteps
        num_steps = getattr(self.cfg.diffusion, 'sampling_timesteps', 50)
        
        # Create base scheduling matrix
        scheduling_matrix = torch.full((num_steps + 1, B, length), -1, dtype=torch.long, device=device)
        
        # For the last frame, create a noise schedule from max to 0
        if self.cfg.diffusion.is_continuous:
            # For continuous diffusion, use timesteps from 1.0 to 0.0
            noise_schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        else:
            # For discrete diffusion, use timesteps from max to 0
            noise_schedule = torch.linspace(
                self.timesteps - 1, -1, num_steps + 1, device=device
            ).long()
        # Apply noise schedule only to the last frame (index -1)
        # Keep context frames (indices 0 to T-1) as -1 (clean)# Apply noise schedule to all frames
        if is_ldm:
            scheduling_matrix[:, :, :] = noise_schedule.unsqueeze(1).unsqueeze(2).expand(-1, B, scheduling_matrix.shape[2])
        else:
            scheduling_matrix[:, :, -1] = noise_schedule.unsqueeze(1).expand(-1, B)

        
        return scheduling_matrix


class DFOT_wm(DFOT):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.wm = AutoEncoderSLOW(cfg)
