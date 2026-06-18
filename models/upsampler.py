import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.act(self.conv1(x))
        c2 = self.act(self.conv2(torch.cat([x, c1], dim=1)))
        c3 = self.act(self.conv3(torch.cat([x, c1, c2], dim=1)))
        c4 = self.act(self.conv4(torch.cat([x, c1, c2, c3], dim=1)))
        c5 = self.conv5(torch.cat([x, c1, c2, c3, c4], dim=1))
        return x + c5 * 0.2


class RRDB(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_feat: int = 64,
        num_blocks: int = 23,
        num_grow: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        if scale not in (2, 4, 8):
            raise ValueError("RRDBNet supports scale 2, 4, or 8 only.")

        self.conv_first = nn.Conv2d(in_channels, num_feat, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)

        up_layers = []
        for _ in range(int(math.log(scale, 2))):
            up_layers.extend(
                [
                    nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.upsample = nn.Sequential(*up_layers)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk
        feat = self.upsample(feat)
        feat = self.act(self.conv_hr(feat))
        return self.conv_last(feat)


class Upsampler(nn.Module):
    """
    Wrap a fixed low-res model and learn an upsampling head.
    Forward: pose -> base_model -> pred(low-res) -> upsampler -> pred(high-res)
    """
    def __init__(self, config):
        super().__init__()
        config = config['model'] if isinstance(config, dict) else config.model

        base_cfg = config.get('base_model', {})
        base_type = base_cfg.get('type', None)
        if base_type is None:
            raise ValueError("Upsampler requires model.base_model.type to load the fixed model.")

        self.freeze_base = bool(base_cfg.get('freeze', True))
        self.target_key = config.get('target_key', 'image')
        self.upsample_mode = config.get('upsample_mode', 'bilinear')
        self.align_corners = config.get('align_corners', False)
        self.scale_factor = config.get('scale_factor', None)
        self.sr_backbone = config.get('sr_backbone', 'native')

        in_channels = int(config.get('in_channels', 3))
        out_channels = int(config.get('out_channels', in_channels))
        self.upscale = int(config.get('upscale', 4))

        if self.sr_backbone == "esrgan":
            esrgan_num_feat = int(config.get('esrgan_num_feat', 64))
            esrgan_num_rrdb = int(config.get('esrgan_num_rrdb', 23))
            esrgan_growth = int(config.get('esrgan_growth_channels', 32))
            self.sr = RRDBNet(
                in_channels=in_channels,
                out_channels=out_channels,
                num_feat=esrgan_num_feat,
                num_blocks=esrgan_num_rrdb,
                num_grow=esrgan_growth,
                scale=self.upscale,
            )
        else:
            hidden_channels = int(config.get('hidden_channels', 64))
            num_rrdb = int(config.get('num_rrdb', 6))
            growth_channels = int(config.get('growth_channels', 32))

            self.pre = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
            self.rrdb_trunk = nn.Sequential(
                *[RRDB(hidden_channels, growth_channels) for _ in range(num_rrdb)]
            )
            self.trunk_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

            if self.upscale != 4:
                raise ValueError("Upsampler currently supports upscale=4 only.")

            self.up1 = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.up2 = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.post = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        base_model_cfg = base_cfg.get('config', None)
        if base_model_cfg is None:
            base_model_cfg = {'model': dict(base_cfg)}

        self.base_model = self._load_base_model(base_type, base_model_cfg, base_cfg.get('ckpt_path', None))
        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

    def _load_base_model(self, base_type, base_model_cfg, ckpt_path):
        from . import LitModel, dict_model
        if ckpt_path:
            lit = LitModel.load_from_checkpoint(
                ckpt_path,
                model_cls=base_type,
                map_location="cpu",
                strict=False,
            )
            return lit.model
        else:
            raise ValueError(f"No checkpoint path provided for base model: {base_type}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self

    def forward(self, batch: dict) -> dict:
        if self.freeze_base:
            with torch.no_grad():
                batch = self.base_model(batch)
        else:
            batch = self.base_model(batch)

        low_res = batch['pred']
        if low_res.dim() == 5:
            bsz, timesteps, channels, height, width = low_res.shape
            low_res_flat = low_res.reshape(bsz * timesteps, channels, height, width)
        elif low_res.dim() == 4:
            bsz, channels, height, width = low_res.shape
            timesteps = 1
            low_res_flat = low_res
        else:
            raise ValueError(f"Unsupported pred shape: {low_res.shape}")

        target = batch.get(self.target_key, None)
        if target is not None:
            target_h, target_w = target.shape[-2:]
            expected_h = height * self.upscale
            expected_w = width * self.upscale
            if expected_h != target_h or expected_w != target_w:
                low_res_flat = F.interpolate(
                    low_res_flat,
                    size=(target_h // self.upscale, target_w // self.upscale),
                    mode=self.upsample_mode,
                    align_corners=self.align_corners if self.upsample_mode in ["linear", "bilinear", "bicubic", "trilinear"] else None,
                )
        elif self.scale_factor is not None:
            if int(self.scale_factor) != self.upscale:
                raise ValueError("scale_factor must match upscale when using RRDB PixelShuffle.")
        else:
            raise ValueError("Upsampler requires target_key or scale_factor to determine output size.")

        if self.sr_backbone == "esrgan":
            up = self.sr(low_res_flat)
        else:
            feat = self.pre(low_res_flat)
            trunk = self.trunk_conv(self.rrdb_trunk(feat))
            feat = feat + trunk
            feat = self.up1(feat)
            feat = self.up2(feat)
            up = self.post(feat)

        if low_res.dim() == 5:
            up = up.reshape(bsz, timesteps, up.size(1), up.size(2), up.size(3))

        batch['pred'] = up

        # Ensure optional loss keys exist for GAN losses.
        for key in ["kl_loss", "slow_loss", "uniform_loss"]:
            if key not in batch:
                batch[key] = torch.tensor(0.0, device=up.device, dtype=up.dtype)

        return batch
