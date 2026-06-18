import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from rotary_embedding_torch import RotaryEmbedding

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
            return_attn: bool = False,
            rope: bool = False,
            single_encoder: bool = False,
            window_size: int = 0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self._return_attn = False
        self.single_encoder = single_encoder
        self.window_size = window_size
        self.register_buffer("sliding_mask", None, persistent=False)

        print(f'Using window size: {self.window_size}......')
        if self.single_encoder:
            print(f'Using single encoder......')

        # if rope:
        self.RoPE = RotaryEmbedding(
            self.head_dim // 2,
            cache_max_seq_len=500,  
            theta_rescale_factor=100
        )
        print(f'Using RoPE......')
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @property
    def return_attn(self):
        return self._return_attn
    
    @return_attn.setter
    def return_attn(self, value: bool):
        assert isinstance(value, bool)
        self._return_attn = value

    @staticmethod
    def create_sliding_window_mask(seq_len, window_size):
        if window_size == 0:
            return None, True
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
        mask -= torch.tril(torch.ones(seq_len, seq_len), diagonal=-window_size)
        attn_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return attn_mask, False
    
    @staticmethod
    def _build_sliding_mask(L: int, w: int, device):
        if w <= 0:
            return None   # causal flag will be True
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
        
        # if self.RoPE is not None:
        q = self.RoPE.rotate_queries_or_keys(q)
        k = self.RoPE.rotate_queries_or_keys(k)

        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, N, self.head_dim)
        v = v.reshape(B * self.num_heads, N, self.head_dim)

        attn_mask, is_causal = self._get_mask(N, x.device)

        # Use built-in causal masking
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=is_causal
        )

        x = x.reshape(B, self.num_heads, N, self.head_dim).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, adaLN_zero=True, return_attn=False, rope=False, dropout=0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, return_attn=return_attn, rope=rope, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        if adaLN_zero:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None

    @property
    def return_attn(self):
        return self.attn.return_attn
    
    @return_attn.setter
    def return_attn(self, value: bool):
        self.attn.return_attn = value
    
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
        if self.return_attn:
            return x, attn
        else:
            return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)

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

class TransformerAdaLN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        input_size = config.model.input_size
        output_size = config.model.output_size
        cond_size = config.model.cond_size 
        nhead = config.model.nhead
        num_layers = config.model.num_layers
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.dropout
        
        self.noise_t = getattr(config.model, 'noise_t', 0)
        self.causal_mask = None
        self.dim_feedforward = dim_feedforward
        
        self.data_embedding = nn.Linear(input_size, dim_feedforward)
        self.cond_proj = nn.Embedding(cond_size, dim_feedforward)
        
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)
        
        # 使用自定义的Transformer
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=dim_feedforward,
                num_heads=nhead
            ) for _ in range(num_layers)
        ])
        self.decoder = FinalLayer(
            hidden_size=dim_feedforward,
            output_size=output_size
        )
        print(f'current noise_t: {self.noise_t}')

    def forward(self, batch):
        # 分离输入数据和条件
        inputs = batch['inputs']
        inputs, cond = inputs[:, :, :-1], inputs[:, :, -1]
        
        # inputs B, T, D
        # random noise for training
        if self.training and self.noise_t > 0:
            T = inputs.shape[1]
            # 生成一个形状为 [T] 的随机张量，取值范围 [0, self.noise_t)
            noise_levels = torch.rand(T, device=inputs.device) * self.noise_t
            # 将 noise_levels 重塑为 [1, T, 1]，以便广播到 [B, T, D]
            noise_levels = noise_levels.view(1, T, 1)
            # 生成与 inputs 同形状的随机噪声，并按每个 time step 对应的 noise level 缩放
            noise = torch.randn_like(inputs) * noise_levels
            inputs = inputs + noise
            
        # 数据嵌入
        x = self.data_embedding(inputs) * math.sqrt(self.dim_feedforward)
        x = self.pos_encoder(x)
        
        # 条件嵌入
        cond_emb = self.cond_proj(cond.long())  # (B, T) -> (B, T, D)
        
        # Transformer处理
        for block in self.blocks:
            x = block(x, cond_emb)
        
        batch['pred'] = self.decoder(x, cond_emb)
        return batch

class TransformerAdaLNwithBottleneck(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        input_size = config.model.input_size
        output_size = config.model.output_size
        cond_size = config.model.cond_size 
        nhead = config.model.nhead
        num_layers = config.model.num_layers
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.dropout
        bottleneck_dim = config.model.bottleneck_dim
        decoder_dim = config.model.decoder_dim
        
        self.noise_t = getattr(config.model, 'noise_t', 0)
        self.causal_mask = None
        self.dim_feedforward = dim_feedforward
        
        self.data_embedding = nn.Linear(input_size - 1, dim_feedforward)
        self.action_proj = nn.Embedding(cond_size, bottleneck_dim)
        
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)
        self.encoder_blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=dim_feedforward,
                num_heads=nhead,
                adaLN_zero=False
            ) for _ in range(num_layers//2)
        ])
        self.bottleneck = BottleneckLayer(dim_feedforward, bottleneck_dim)
        self.decoder_blocks = DecoderBlock(dim_feedforward, decoder_dim, num_layers - num_layers//2)
        
        self.decoder = FinalLayer(
            hidden_size=decoder_dim,
            output_size=output_size,
            adaLN_zero=False
        )
        print(f'current noise_t: {self.noise_t}')

    def forward(self, batch):
        inputs = batch['inputs']
        x_data = inputs[:, :, :-1]  # [B, T, D-1]
        action = inputs[:, :, -1].long()  # [B, T]
        
        action_emb = self.action_proj(action)  # [B, T, D]
        zero_emb = torch.zeros_like(action_emb)

        if self.training and self.noise_t > 0:
            T = x_data.shape[1]
            noise_levels = torch.rand(T, device=x_data.device) * self.noise_t
            noise = torch.randn_like(x_data) * noise_levels.view(1, T, 1)
            x_data = x_data + noise
        
        x = self.data_embedding(x_data) * math.sqrt(self.dim_feedforward)
        x = self.pos_encoder(x)
        
        # 编码器处理（无条件）
        for block in self.encoder_blocks:
            x = block(x, zero_emb)
        
        x = self.bottleneck(x, action_emb)
        
        # 解码器处理（无条件）
        x = self.decoder_blocks(x)
        
        batch['pred'] = self.decoder(x, zero_emb)
        return batch

class TransformerModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        input_size = config.model.input_size
        output_size = config.model.output_size
        nhead = config.model.nhead
        num_layers = config.model.num_layers
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.dropout

        self.causal_mask = None
        self.dim_feedforward = dim_feedforward
        self.embedding = nn.Linear(input_size, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"  
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(dim_feedforward, output_size)

    def forward(self, batch):
        x, padding_mask = batch['inputs'], batch['padding_mask']
        B, T, _ = x.shape
        if self.causal_mask is None or self.causal_mask.size(0) != T:
            self.causal_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)
        
        x = self.embedding(x) * math.sqrt(self.dim_feedforward)
        x = self.pos_encoder(x)
        if padding_mask is None:
            x = self.transformer_encoder(x, mask=self.causal_mask, is_causal=True) 
        else:
            x = self.transformer_encoder(x, 
                                     mask=self.causal_mask,
                                     src_key_padding_mask=~padding_mask,
                                     is_causal=True)
        batch['pred'] = self.decoder(x)
        return batch

class TransformerClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        vocab_size = config.model.vocab_size
        cond_size = config.model.cond_size
        num_classes = config.model.num_classes
        dim_feedforward = config.model.dim_feedforward
        nhead = config.model.nhead
        num_layers = config.model.num_layers
        dropout = config.model.dropout

        self.dim_feedforward = dim_feedforward
        self.token_embedding = nn.Embedding(vocab_size, dim_feedforward)
        
        self.cond_method = config.model.cond_method  # 'concat' 或 'add'
        if self.cond_method == 'concat':
            self.cond_proj = nn.Embedding(cond_size, dim_feedforward)
            self.input_proj = nn.Linear(2*dim_feedforward, dim_feedforward)
        elif self.cond_method == 'add':
            self.cond_proj = nn.Embedding(cond_size, dim_feedforward)
        
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_feedforward, num_classes)
        self.register_buffer("causal_mask", None)

    def forward(self, batch):
        inputs = batch['inputs']       # (B, T, 1+1)
        # inputs = inputs.long()
        tokens, cond = inputs[:, :, 0], inputs[:, :, 1]  
        padding_mask = ~batch['padding_mask'] if 'padding_mask' in batch else None  # (B, T)
        
        B, T = tokens.shape
        
        if self.causal_mask is None or self.causal_mask.size(0) != T:
            self.causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool) * float('-inf'), diagonal=1
            ).to(self.device)
        
        x = self.token_embedding(tokens) * math.sqrt(self.dim_feedforward)  # (B, T, D)
        
        if self.cond_method == 'concat':
            cond_emb = self.cond_proj(cond)  # (B, T, D)
            # cond_emb = cond_emb.expand(-1, T, -1)         # (B, T, D)
            x = torch.cat([x, cond_emb], dim=-1)          # (B, T, 2D)
            x = self.input_proj(x)                        # (B, T, D)
        elif self.cond_method == 'add':
            cond_emb = self.cond_proj(cond)      
            x = x + cond_emb           

        x = self.pos_encoder(x)
        x = self.transformer(
            x,
            mask=self.causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True
        )
        
        logits = self.classifier(x)  # (B, T, num_classes)
        batch['logits'] = logits
        return batch

    # def training_step(self, batch, batch_idx):
    #     logits = self(batch)['logits']
    #     targets = batch['targets']
        
    #     # 计算交叉熵损失（自动处理log_softmax）
    #     loss = F.cross_entropy(
    #         logits.view(-1, logits.size(-1)),
    #         targets.view(-1),
    #         ignore_index=-100  # 假设用-100表示需要忽略的位置
    #     )
    #     self.log('train_loss', loss)
    #     return loss

class TransformerConvBlock(nn.Module):
    def __init__(self, map_size, hidden_size, num_heads, mlp_ratio=4.0, adaLN_zero=True, rope=False, dropout=0, **block_kwargs):
        super().__init__()
        # assert math.prod(map_size) == hidden_size, "map_size should be the same as hidden_size"
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        if adaLN_zero:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None

    def forward(self, x, c=None):
        B, L, C, H, W = x.shape
        x = x.reshape(B, L, C*H*W)
        if self.adaLN_modulation is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            attn_out, attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_msa * attn_out
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            attn_out, attn = self.attn(self.norm1(x))
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        return x.reshape(B, L, C, H, W)
