import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from rotary_embedding_torch import RotaryEmbedding
from omegaconf import DictConfig
from typing import Optional
from ..modules.embeddings import StochasticTimeEmbedding

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            rope: bool = False,
            window_size: int = 0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.register_buffer("sliding_mask", None, persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if rope:
            self.RoPE = RotaryEmbedding(
                self.head_dim // 2,
                cache_max_seq_len=500,  
                theta_rescale_factor=100
            )
        else:
            self.RoPE = None


    @staticmethod
    def _build_sliding_mask(L: int, w: int, device):
        if w <= 0:
            return None  
        m = torch.tril(torch.ones(L, L, device=device))
        m -= torch.tril(torch.ones(L, L, device=device), diagonal=-w)
        return m.float().masked_fill(m == 0, float("-inf")).masked_fill(m == 1, 0.0)

    def _get_mask(self, seq_len: int, device):
        if self.window_size == 0:
            return None, True                       # causal
        # allocate or enlarge
        if (
            self.sliding_mask is None
            or self.sliding_mask.size(0) < seq_len
        ):
            self.sliding_mask = self._build_sliding_mask(
                seq_len, self.window_size, device
            )
        # just slice if current buffer is large enough
        return self.sliding_mask[:seq_len, :seq_len], False

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        # 3, B, H, N, D
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # B, H, N, D
        q, k = self.q_norm(q), self.k_norm(k)

        if self.RoPE is not None:
            q = self.RoPE.rotate_queries_or_keys(q)
            k = self.RoPE.rotate_queries_or_keys(k)

        attn_mask, is_causal = self._get_mask(N, x.device)

        # Use built-in causal masking
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=is_causal
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, adaLN_zero=True, rope=True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        if adaLN_zero:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None
    
    def forward(self, x, c):
        if self.adaLN_modulation is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            attn_out, attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_msa * attn_out
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            attn_out, attn = self.attn(self.norm1(x))
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size, adaLN_zero=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        if adaLN_zero:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None

    def forward(self, x, c):
        if self.adaLN_modulation is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x
    
class BottleneckLayer(nn.Module):
    """
    The bottleneck layer of DiT.
    """
    def __init__(self, hidden_size, bottleneck_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(bottleneck_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, bottleneck_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 2 * bottleneck_dim, bias=True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, 4 * bottleneck_dim),
            nn.SiLU(),
            nn.Linear(4 * bottleneck_dim, hidden_size)
        )
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.linear(x)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.mlp(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, input_size, decoder_dim, num_layers):
        super().__init__()
        self.mlp = Mlp(in_features=input_size, hidden_features=decoder_dim, out_features=decoder_dim, act_layer=nn.SiLU, drop=0)
        self.blocks = nn.Sequential(*[
            Mlp(in_features=decoder_dim, hidden_features=decoder_dim, act_layer=nn.SiLU, drop=0)
            for _ in range(num_layers-1)
        ])

    def forward(self, x):
        x = self.mlp(x)
        for block in self.blocks:
            x = x + block(x)
        return x
    
class DiT2D(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        x_shape: Optional[torch.Size] = None,
        max_tokens: Optional[int] = None,
        external_cond_dim: Optional[int] = None,
        external_cond_levels: Optional[int] = None,
        use_causal_mask: bool = True,
        action_continuous: bool = False,
    ):
        if not use_causal_mask:
            raise NotImplementedError(
                "Non-causal masking is not yet implemented for DiT2D backbone"
            )
        if x_shape is not None or max_tokens is not None or external_cond_dim is not None:
            warnings.warn("x_shape, max_tokens and external_cond_dim are not used in this model")

        super().__init__()

        input_size = cfg.input_size
        dim_feedforward = cfg.dim_feedforward
        nhead = cfg.nhead
        num_layers = cfg.num_layers

        self.action_levels = external_cond_levels + 1 # +1 for the padding index
        noise_level_dim = cfg.noise_level_dim
        self.action_continuous = action_continuous
        if self.action_continuous:
            self.action_proj = nn.Linear(external_cond_levels, dim_feedforward // 2)
        else:
            self.action_emb = nn.Embedding(self.action_levels, dim_feedforward // 2)
        self.input_proj = nn.Linear(input_size, dim_feedforward)
        self.transformer = nn.ModuleList([
            DiTBlock(dim_feedforward, num_heads=nhead, mlp_ratio=2.0, 
                     adaLN_zero=True, rope=True)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(dim_feedforward, input_size)

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=noise_level_dim,
            time_embed_dim=dim_feedforward // 2,
            use_fourier=cfg.get("use_fourier_noise_embedding", False),
        )

    def forward(self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        actions_mask: Optional[torch.Tensor] = None,
    ):
        assert actions_mask is None, "Actions mask is not supported for DiT2D."
        emb = self.noise_level_pos_embedding(noise_levels)
        if self.action_continuous:
            actions = self.action_proj(actions)
        else:
            actions = torch.where(actions < 0, actions.new_full(actions.shape, self.action_levels-1), actions)
            if actions.dim() == 3:
                actions = actions.squeeze(-1)
            actions = self.action_emb(actions.long())
        emb = torch.cat([emb, actions], dim=-1)

        f = self.input_proj(x)
        for layer in self.transformer:
            f = layer(f, emb)
        output = self.output_proj(f)
        
        return output