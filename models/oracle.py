from typing import Dict

import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs'


class FourierEmbedding(nn.Module):
    def __init__(self, pos_dim=2, angle_dim=2, num_frequencies=6):
        super().__init__()
        self.pos_dim = pos_dim
        self.angle_dim = angle_dim
        self.num_frequencies = num_frequencies
        freq_bands = torch.arange(1, num_frequencies + 1) * torch.pi  # (L,)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        """
        x: (B, 4) = [x, y, cos, sin]
        """
        B, D = x.shape
        assert D == self.pos_dim + self.angle_dim

        pos = x[:, :self.pos_dim]        # (B, 2)
        angle = x[:, self.pos_dim:]      # (B, 2)

        pos_expanded = pos.unsqueeze(-1) * self.freq_bands

        # (B, 2, 2L)
        pos_fourier = torch.cat(
            [torch.sin(pos_expanded), torch.cos(pos_expanded)],
            dim=-1
        )

        # (B, 2 * 2L)
        pos_fourier = pos_fourier.reshape(B, -1)

        # concat: 原始 pos + Fourier(pos) + angle
        return torch.cat([pos, pos_fourier, angle], dim=-1)

    def get_output_dim(self) -> int:
        # pos: 2
        # fourier(pos): 2 * 2L
        # angle: 2
        return self.pos_dim * (1 + 2 * self.num_frequencies) + self.angle_dim

class RenderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = config['model'] if isinstance(config, dict) else config.model
        self.model_name = config.get('model_name', "stabilityai/sd-vae-ft-ema")
        self.pos_dim = config.get('pos_dim', 2)
        self.angle_dim = config.get('angle_dim', 2)
        self.num_frequencies = config.get('num_frequencies', 6)
        self.map_size = tuple(config.get('map_size', (4, 8, 8)))

        if self.model_name in ["stabilityai/sd-vae-ft-ema"]:
            vae_config = AutoencoderKL.load_config(self.model_name)
            self.model = AutoencoderKL.from_config(vae_config)
        elif self.model_name in ["tiny-vae"]:
            vae_config = AutoencoderKL.load_config(CONFIG_PATH / 'tiny_vae.json')
            self.model = AutoencoderKL.from_config(vae_config)
        elif self.model_name in ["vae-64"]:
            vae_config = AutoencoderKL.load_config(CONFIG_PATH / 'vae_64.json')
            self.model = AutoencoderKL.from_config(vae_config)
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

        del self.model.encoder
        
        self.fourier_embedding = FourierEmbedding(
            pos_dim=self.pos_dim,
            angle_dim=self.angle_dim,
            num_frequencies=self.num_frequencies,
        )
        latent_dim = self.map_size[0] * self.map_size[1] * self.map_size[2]
        self.latent_proj = nn.Linear(self.fourier_embedding.get_output_dim(), latent_dim)

    def decode(self, f_in: torch.Tensor) -> torch.Tensor:
        B, L, _ = f_in.shape
        f_in = f_in.reshape(B * L, -1)
        f_in = self.fourier_embedding(f_in)
        f_in = self.latent_proj(f_in)
        f_in = f_in.reshape(B * L, *self.map_size)
        out = self.model.decode(f_in).sample
        out = out.reshape(B, L, *out.shape[1:])
        return out

    def get_latents(self, x: Dict) -> torch.Tensor:
        return x['pose']  # x['pose'] shape: [B, L, 4]

    def forward(self, batch_data: Dict) -> Dict:
        x = batch_data['pose']
        out = self.decode(x)
        zero = out.new_zeros(())
        
        batch_data['latent'] = x
        batch_data['pred'] = out
        batch_data['kl_loss'] = zero
        batch_data['slow_loss'] = zero
        batch_data['uniform_loss'] = zero

        return batch_data
