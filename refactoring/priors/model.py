import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Port of part of InverseBench (EDM framework, Karras et al. 2022) to a standalone module for use in pyFLIP.
#
# The network is designed to work on 256x256 spline nodes of JWST WSS OPD maps, and the prior is trained on those.
# "spline nodes" are the control points of a cubic spline interpolation of the OPD map, which is the way that
# stpsf up/downscales these maps to the 1024x1024 pupil plane.
#
# There are two kinds of priors: one on the full OPD map (kind 'map'), and one on the drift between consecutive WSS maps (kind 'drift').
# Experimentally the first one can perform worse if the telescope reaches a state not represented in the training set,
# but the evolution (drift) is much more consistent to that of the training set, so this performs better but needs some initialization.

############ NETWORK

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x

class Conv2d(nn.Module):
    # 'kernel' = 0 means no convolution, only the (optional) 2x up/downsampling with a [1, 1] box filter
    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel, kernel)) if kernel else None
        self.bias = nn.Parameter(torch.zeros(out_channels)) if kernel and bias else None
        f = torch.ones(1, 1, 2, 2) / 4
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        if self.up:
            f = self.resample_filter.to(x.dtype)
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2)
        if self.down:
            f = self.resample_filter.to(x.dtype)
            x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2)
        if self.weight is not None:
            x = F.conv2d(x, self.weight.to(x.dtype), padding=self.weight.shape[-1] // 2)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype).reshape(1, -1, 1, 1)
        return x

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)

class UNetBlock(nn.Module):
    # Residual block with the noise-level embedding injected as an adaptive scale/shift, optional self-attention
    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False, attention=False, channels_per_head=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = out_channels // channels_per_head if attention else 0

        self.norm0 = GroupNorm(num_channels=in_channels)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * 2)
        self.norm1 = GroupNorm(num_channels=out_channels)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3)

        self.skip = None
        if out_channels != in_channels or up or down:
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=1 if out_channels != in_channels else 0, up=up, down=down)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1)
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))
        scale, shift = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype).chunk(chunks=2, dim=1)
        x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        x = self.conv1(x)
        x = x + (self.skip(orig) if self.skip is not None else orig)

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).permute(0, 3, 2, 1).unbind(2)
            a = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1)
            x = self.proj(a.reshape(*x.shape)) + x
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = (1 / self.max_positions) ** (freqs / (self.num_channels // 2))
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)

class DhariwalUNet(nn.Module):
    # ADM U-Net (Dhariwal & Nichol 2021) as written in the EDM code base
    def __init__(self, img_resolution, in_channels, out_channels, model_channels=192, channel_mult=[1, 2, 3, 4],
                 channel_mult_emb=4, num_blocks=3, attn_resolutions=[32, 16, 8]):
        super().__init__()
        emb_channels = model_channels * channel_mult_emb
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64)

        # Mapping of the noise level to the embedding vector
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels)

        # Encoder
        self.enc = nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3)

    def forward(self, x, noise_labels):
        emb = self.map_noise(noise_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        return self.out_conv(F.silu(self.out_norm(x)))

class EDMPrecond(nn.Module):
    # EDM preconditioning: turns the raw U-Net F into the denoiser D(x, sigma) = c_skip x + c_out F(c_in x, c_noise)
    def __init__(self, img_resolution, img_channels, sigma_data=0.5, **unet_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.sigma_data = sigma_data
        self.model = DhariwalUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, **unet_kwargs)

    def forward(self, x, sigma):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip * x + c_out * self.model(c_in * x, c_noise.flatten())

############ PRIOR

class OPDPrior:
    """
    The denoiser D(x, sigma) of the OTE OPD prior and the mapping between its network units and an OPD in metres.
    - ckpt_path: InverseBench training checkpoint (its 'ema' weights are used)
    - stats_path: stats.npz of the training set (mean_map, std, A, pupil_1024)
    - net_kwargs: the 'model' section of the InverseBench pretrain config (img_resolution, img_channels, model_channels, ...)
    - kind: 'map' (network = full OPD map) or 'drift' (network = drift on top of a WSS map; scale from normalization_path)
    The 'condition' keyword of denoise is a hook for a future conditioned network (e.g. on a previous WFS map); it is
    passed on to the network as keyword arguments and unused for now.
    """
    def __init__(self, ckpt_path, stats_path, net_kwargs, kind='map', normalization_path=None, DEVICE='cuda'):
        self.device = DEVICE
        self.kind = kind
        self.net = EDMPrecond(**net_kwargs)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.net.load_state_dict(ckpt['ema'])
        self.net.to(DEVICE).eval().requires_grad_(False) # the network is fixed; gradients only flow to its input
        self.img_resolution = self.net.img_resolution
        self.img_channels = self.net.img_channels

        stats = np.load(stats_path)
        if kind == 'map':
            self.mean_map = torch.tensor(stats['mean_map'], dtype=torch.float64).to(DEVICE) # (256, 256), metres
            self.scale = 2. * float(stats['std']) # 1 network unit in metres
        elif kind == 'drift':
            self.mean_map = torch.zeros(stats['mean_map'].shape, dtype=torch.float64).to(DEVICE)
            self.scale = float(np.load(normalization_path)['scale'])
        else:
            raise ValueError("kind must be 'map' or 'drift'")
        self.A = torch.tensor(stats['A'], dtype=torch.float64).to(DEVICE) # (1024, 256) spline interpolation matrix
        self.pupil_1024 = torch.tensor(stats['pupil_1024'], dtype=torch.float64).to(DEVICE) # (1024, 1024), file orientation

    def latent_shape(self):
        return (1, self.img_channels, self.img_resolution, self.img_resolution)

    def denoise(self, x, sigma, condition=None):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device)
        return self.net(x, sigma, **(condition or {}))

    def to_opd(self, x):
        # (1, 1, 256, 256) network units -> (1, 1024, 1024) metres, pyFLIP orientation (flipped in y like the WFS map)
        y = x.to(torch.float64) * self.scale + self.mean_map
        opd_file = self.pupil_1024 * (self.A @ y @ self.A.T)
        return torch.flip(opd_file[:, 0], dims=[-2])

    def mean_opd(self):
        # Prior mean map (x = 0): the training set's mean for kind 'map', zero for kind 'drift'
        return self.to_opd(torch.zeros(self.latent_shape(), device=self.device))
