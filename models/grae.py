import ast
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from .transformer import TransformerConvBlock
from .utils import pairwise_loss, uniform_loss_temporal
import torch.distributed as dist
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs'


def parse_map_size(value, default=(4, 8, 8)):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return tuple(int(dim) for dim in value)

def differentiable_all_gather_variable_length(tensor, mask):
    """
    处理变长序列的全局 Gather。
    1. 找到所有卡中最大的 L。
    2. 将当前卡的 tensor 和 mask 填充(Pad)到最大 L。
    3. 执行 all_gather 并保留梯度。
    
    Args:
        tensor: [B, L, D]
        mask: [B, L] (假设 True 为有效数据，False 为 Padding)
    """
    if not dist.is_initialized():
        return tensor, mask

    device = tensor.device
    local_L = tensor.size(1)
    
    # --- 步骤 1: 找到全局最大的 L ---
    t_local_L = torch.tensor([local_L], dtype=torch.long, device=device)
    dist.all_reduce(t_local_L, op=dist.ReduceOp.MAX)
    max_L = t_local_L.item()

    # --- 步骤 2: Pad 到最大长度 ---
    if local_L < max_L:
        pad_size = max_L - local_L
        # F.pad 的参数顺序是 (最后一维左, 最后一维右, 倒数第二维左, ...)
        # 对 tensor (B, L, D) 的 L 维进行 pad -> (0, 0, 0, pad_size)
        tensor_padded = F.pad(tensor, (0, 0, 0, pad_size)) 
        # 对 mask (B, L) 的 L 维进行 pad -> (0, pad_size)
        # 注意：填充部分设为 False (无效)
        mask_padded = F.pad(mask, (0, pad_size), value=False)
    else:
        tensor_padded = tensor
        mask_padded = mask

    # --- 步骤 3: 正常的 Differentiable All Gather ---
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tensors_gather = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    masks_gather = [torch.zeros_like(mask_padded) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor_padded.contiguous())
    dist.all_gather(masks_gather, mask_padded.contiguous())
    # 梯度回传 Trick：替换回当前卡的带梯度的 tensor
    tensors_gather[rank] = tensor_padded
    global_tensor = torch.cat(tensors_gather, dim=0) # [WorldSize*B, max_L, D]
    global_mask = torch.cat(masks_gather, dim=0)     # [WorldSize*B, max_L]

    return global_tensor, global_mask

class GRVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        full_config = config
        loss_kwargs = full_config.get("loss", {}).get("kwargs", {})
        self.compute_slow_loss = float(loss_kwargs.get("slow_weight", 0.0)) != 0.0
        self.compute_uniform_loss = float(loss_kwargs.get("uniform_weight", 0.0)) != 0.0
        config = config['model']
        self.model_name = config.get('model_name', "stabilityai/sd-vae-ft-ema")
        self.layers_transformer = config.get('layers_transformer', 1)
        self.rope = config.get('rope', True)
        self.window_size = config.get('window_size', 15)
        self.map_size = parse_map_size(config.get('map_size', (4, 8, 8)))
        self.uniform_max_points = config.get('uniform_max_points', None)
        self.uniform_chunk_size = config.get('uniform_chunk_size', 512)
        self.ae_micro_batch_size = int(config.get('ae_micro_batch_size', 0) or 0)
        dim_forward = config.get('dim_forward', 512)
        proj_dim = config.get('proj_dim', 4)
        self.transformer = nn.ModuleList([
            TransformerConvBlock(self.map_size, dim_forward, num_heads=8, mlp_ratio=4.0, adaLN_zero=None,
                rope=self.rope, window_size=self.window_size)
            for _ in range(self.layers_transformer)
        ])
        self.projector = nn.Linear(math.prod(self.map_size), proj_dim)
        if self.model_name in ["stabilityai/sd-vae-ft-ema"]:
            config = AutoencoderKL.load_config(self.model_name)
            self.model = AutoencoderKL.from_config(config)
        elif self.model_name in ["tiny-vae"]:
            config = AutoencoderKL.load_config(CONFIG_PATH / 'tiny_vae.json')
            self.model = AutoencoderKL.from_config(config)
            print(self.model)
        elif self.model_name in ["vae-64", "vae-32-small"]:
            config_name = 'vae_32_small.json' if self.model_name == "vae-32-small" else 'vae_64.json'
            config = AutoencoderKL.load_config(CONFIG_PATH / config_name)
            self.model = AutoencoderKL.from_config(config)
            print(self.model)
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def _apply_ae_in_chunks(self, x: torch.Tensor, fn):
        if self.ae_micro_batch_size <= 0 or x.size(0) <= self.ae_micro_batch_size:
            return fn(x)
        outputs = []
        for start in range(0, x.size(0), self.ae_micro_batch_size):
            outputs.append(fn(x[start:start + self.ae_micro_batch_size]))
        return torch.cat(outputs, dim=0)

    def _encode_parameters(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_ae_in_chunks(
            x,
            lambda chunk: self.model.encode(chunk).latent_dist.parameters,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "tiny-vae", "vae-64", "vae-32-small"]:
            return self._apply_ae_in_chunks(
                x,
                lambda chunk: self.model.encode(chunk).latent_dist.sample(),
            )
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 3:
            B, L, D = latent.shape
            latent = latent.reshape(B * L, -1, *self.map_size[1:])
        else:
            B, L, *_ = latent.shape
            latent = latent.reshape(B * L, -1, *self.map_size[1:])
        
        output = self._decode(latent)
        output = output.reshape(B, L, *output.shape[1:])
        return output
    
    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._apply_ae_in_chunks(
            latent,
            lambda chunk: self.model.decode(chunk).sample,
        )

    def get_latents(self, batch: dict) -> torch.Tensor:
        x = batch['image']
        padding_mask = batch['padding_mask'].bool()
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        params = self._encode_parameters(x)
        _, c, h, w = params.shape
        params = params.reshape(B, L, c, h, w)

        for block in self.transformer:
            params = block(params)
        params = params.reshape(B*L, c, h, w)
        gauss = DiagonalGaussianDistribution(params)
        z = gauss.mode()
        return z.reshape(B, L, -1)
    
    def get_val_latents(self, batch: dict) -> torch.Tensor:
        x = batch['image']
        padding_mask = batch['padding_mask'].bool()
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        params = self._encode_parameters(x)
        _, c, h, w = params.shape
        params = params.reshape(B, L, c, h, w)

        for block in self.transformer:
            params = block(params)
        params = params.reshape(B*L, c, h, w)
        gauss = DiagonalGaussianDistribution(params)
        z = gauss.mode()
        return z.reshape(B, L, -1, *self.map_size[1:])

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch['image']
        padding_mask = batch['padding_mask'].bool()
        
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        params = self._encode_parameters(x)
        _, c, h, w = params.shape
        params = params.reshape(B, L, c, h, w)

        for block in self.transformer:
            params = block(params)
        params = params.reshape(B*L, c, h, w)
        gauss = DiagonalGaussianDistribution(params)
        if self.training:
            z = gauss.sample()
        else:
            z = gauss.mode()
        rec = self._decode(z)
        rec = rec.reshape(B, L, C, H, W)

        # Loss
        latent_mean = gauss.mode()
        latent_mean = latent_mean.reshape(B, L, -1)
        zero_loss = latent_mean.new_zeros(())
        uniform_loss_value = zero_loss
        slow_loss_value = zero_loss
        if self.compute_uniform_loss or self.compute_slow_loss:
            projected_latent = self.projector(latent_mean)
            projected_latent = F.normalize(projected_latent, p=2.0, dim=-1, eps=1e-8)

            if self.compute_uniform_loss:
                global_latent, global_mask = differentiable_all_gather_variable_length(
                    projected_latent,
                    padding_mask
                )
                uniform_loss_value = uniform_loss_temporal(
                    x=global_latent,
                    mask=global_mask,
                    neg_gap=100,
                    dist_threshold=100,
                    max_points=self.uniform_max_points,
                    chunk_size=self.uniform_chunk_size,
                )

            if self.compute_slow_loss:
                latent1 = projected_latent[:, 1:, :]
                latent2 = projected_latent[:, :-1, :]
                valid_mask = padding_mask[:, 1:]
                slow_loss_value = pairwise_loss(
                    latent1=latent1,
                    latent2=latent2,
                    valid_mask=valid_mask,
                    threshold=0,
                    pos_window=100,
                )

        kl_loss = gauss.kl()[padding_mask.reshape(-1)].mean()
        batch['uniform_loss'] = uniform_loss_value
        batch['kl_loss'] = kl_loss
        batch['slow_loss'] = slow_loss_value
        batch['uniform_loss'] = uniform_loss_value
        batch['pred'] = rec
        batch['latent'] = latent_mean

        return batch

class TemporalVAE(GRVAE):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch['image']
        padding_mask = batch['padding_mask'].bool()
        
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        params = self._encode_parameters(x)
        _, c, h, w = params.shape
        params = params.reshape(B, L, c, h, w)

        for block in self.transformer:
            params = block(params)
        params = params.reshape(B*L, c, h, w)
        gauss = DiagonalGaussianDistribution(params)
        if self.training:
            z = gauss.sample()
        else:
            z = gauss.mode()
        rec = self._decode(z)
        rec = rec.reshape(B, L, C, H, W)

        # Loss
        latent_mean = gauss.mode()
        latent_mean = latent_mean.reshape(B, L, -1)
        uniform_loss_value = latent_mean.new_zeros(())
        slow_loss_value = latent_mean.new_zeros(())

        kl_loss = gauss.kl()[padding_mask.reshape(-1)].mean()
        batch['uniform_loss'] = uniform_loss_value
        batch['kl_loss'] = kl_loss
        batch['slow_loss'] = slow_loss_value
        batch['pred'] = rec
        batch['latent'] = latent_mean

        return batch

if __name__ == "__main__":
    model = GRVAE(model_name="stabilityai/sd-vae-ft-ema")
    x = torch.randn(1, 3, 64, 64)
    latent = model.model.encode(x).latent_dist.parameters
    print(latent.shape)
    rec = model.decode(latent)
    print(rec.shape)
