from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:  # pragma: no cover
    Mamba = None


def window_partition(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    batch, depth, height, width, channels = x.shape
    wd, wh, ww = window_size
    x = x.view(
        batch,
        depth // wd,
        wd,
        height // wh,
        wh,
        width // ww,
        ww,
        channels,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    batch: int,
    depth: int,
    height: int,
    width: int,
    channels: int,
) -> torch.Tensor:
    wd, wh, ww = window_size
    x = windows.view(batch, depth // wd, height // wh, width // ww, wd, wh, ww, channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch, depth, height, width, channels)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Scale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, drop: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class ClassMamba(nn.Module):
    def __init__(self, dim: int, drop: float = 0.0, qkv_bias: bool = False, d_state: int = 16, **kwargs):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is required when token_mixer is set to 'mamba'.")
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2, use_fast_path=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        query = self.q(cls_token)
        tokens = torch.cat([query, patch_tokens], dim=1)
        tokens = self.mamba(tokens, inference_params=inference_params)
        updated_cls = self.dropout(self.proj(tokens[:, :1, :]))
        return torch.cat([updated_cls, patch_tokens], dim=1)


class ClassAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, :1, :]
        tokens = x[:, 1:, :]
        batch, _, dim = x.shape
        head_dim = dim // self.num_heads
        q = self.q(cls_token).reshape(batch, 1, self.num_heads, head_dim).transpose(1, 2)
        k = self.k(tokens).reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        v = self.v(tokens).reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.dropout(attention.softmax(dim=-1))
        output = attention @ v
        output = output.transpose(1, 2).reshape(batch, 1, dim)
        output = self.dropout(self.proj(output))
        return torch.cat([output, tokens], dim=1)


class ClassRNN(nn.Module):
    def __init__(self, dim: int, drop: float = 0.0, **kwargs):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dropout(self.proj(x))


class MetaFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        token_mixer,
        drop: float = 0.0,
        drop_path: float = 0.0,
        prune_ln2_mlp: bool = False,
        prune_ln1_token_mixer: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        self.prune_ln1_token_mixer = prune_ln1_token_mixer
        self.prune_ln2_mlp = prune_ln2_mlp
        self.norm1 = nn.Identity() if prune_ln1_token_mixer else nn.LayerNorm(dim)
        self.norm2 = nn.Identity() if prune_ln2_mlp else nn.LayerNorm(dim)

        if prune_ln1_token_mixer:
            self.token_mixer = nn.Identity()
        elif token_mixer is ClassMamba:
            self.token_mixer = token_mixer(dim=dim, drop=drop, d_state=16)
        elif token_mixer is ClassAttention:
            self.token_mixer = token_mixer(dim=dim, drop=drop, num_heads=num_heads)
        else:
            self.token_mixer = token_mixer(dim=dim, drop=drop)

        self.mlp = nn.Identity() if prune_ln2_mlp else Mlp(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=1.0)
        self.res_scale2 = Scale(dim=dim, init_value=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.prune_ln1_token_mixer:
            x = x + self.drop_path1(self.res_scale1(self.token_mixer(self.norm1(x))))
        if not self.prune_ln2_mlp:
            x = x + self.drop_path2(self.res_scale2(self.mlp(self.norm2(x))))
        return x


class PatchMergingV2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, depth, height, width, channels = x.shape
        if (depth % 2 == 1) or (height % 2 == 1) or (width % 2 == 1):
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2, 0, depth % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        return self.reduction(self.norm(x))


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        token_mixer,
        num_layers: int,
        drop: float,
        drop_path: float,
        window_size: tuple[int, int, int],
        prune_flags: list[dict] | None,
    ):
        super().__init__()
        self.window_size = window_size
        blocks = []
        prune_flags = prune_flags or [None] * num_layers
        for block_index in range(num_layers):
            flags = prune_flags[block_index] if block_index < len(prune_flags) else {}
            flags = flags or {}
            blocks.append(
                MetaFormerBlock(
                    dim=in_channels,
                    token_mixer=token_mixer,
                    drop=drop,
                    drop_path=drop_path,
                    prune_ln2_mlp=flags.get("prune_ln2_mlp", False),
                    prune_ln1_token_mixer=flags.get("prune_ln1_token_mixer", False),
                    num_heads=num_heads,
                )
            )
        self.metaformer = nn.Sequential(*blocks)
        self.patch_merging = PatchMergingV2(dim=in_channels)
        self.channel_proj = nn.Conv3d(2 * in_channels, out_channels, kernel_size=1) if out_channels != 2 * in_channels else nn.Identity()
        self.skip_conv = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, depth, height, width = x.shape
        x = rearrange(x, "b c d h w -> b d h w c")

        wd, wh, ww = self.window_size
        pad_d = (wd - depth % wd) % wd
        pad_h = (wh - height % wh) % wh
        pad_w = (ww - width % ww) % ww
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x.permute(0, 4, 1, 2, 3)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            x = x.permute(0, 2, 3, 4, 1)

        _, depth_pad, height_pad, width_pad, channels = x.shape
        windows = window_partition(x, self.window_size)
        windows = self.metaformer(windows)
        x = window_reverse(windows, self.window_size, batch, depth_pad, height_pad, width_pad, channels)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :depth, :height, :width, :].contiguous()

        skip = rearrange(x, "b d h w c -> b c d h w")
        skip = self.skip_conv(skip)

        down = self.patch_merging(x)
        down = rearrange(down, "b d h w c -> b c d h w")
        down = self.channel_proj(down)
        return down, skip


class SlimFormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 14,
        patch_size: int | tuple[int, int, int] = (1, 2, 2),
        base_channels: int = 48,
        num_heads: tuple[int, int, int, int] = (4, 8, 16, 32),
        token_mixer=ClassMamba,
        drop: float = 0.2,
        drop_path: float = 0.2,
        metaformer_layers: tuple[int, int, int] = (2, 2, 2),
        prune_flags_list: list | None = None,
    ):
        super().__init__()
        patch_size_tuple = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.patch_embed = PatchEmbed(
            patch_size=patch_size_tuple,
            in_chans=in_channels,
            embed_dim=base_channels,
            norm_layer=nn.LayerNorm,
            spatial_dims=3,
        )
        self.pos_drop = nn.Dropout(p=drop)
        prune_flags_list = prune_flags_list or [None, None, None]

        self.encoder1 = EncoderStage(base_channels, base_channels * 2, num_heads[0], token_mixer, metaformer_layers[0], drop, drop_path, (8, 8, 8), prune_flags_list[0])
        self.encoder2 = EncoderStage(base_channels * 2, base_channels * 4, num_heads[1], token_mixer, metaformer_layers[1], drop, drop_path, (8, 8, 8), prune_flags_list[1])
        self.encoder3 = EncoderStage(base_channels * 4, base_channels * 8, num_heads[2], token_mixer, metaformer_layers[2], drop, drop_path, (8, 8, 8), prune_flags_list[2])
        self.bottleneck = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=base_channels * 8,
            out_channels=base_channels * 8,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(3, base_channels * 8, base_channels * 4, 3, 2, "instance", True)
        self.decoder2 = UnetrUpBlock(3, base_channels * 4, base_channels * 2, 3, 2, "instance", True)
        self.decoder1 = UnetrUpBlock(3, base_channels * 2, base_channels, 3, 2, "instance", True)
        self.seg_head = UnetOutBlock(3, base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.pos_drop(self.patch_embed(x))
        x1, skip1 = self.encoder1(x0)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x3 = self.bottleneck(x3)
        d3 = self.decoder3(x3, skip3)
        d2 = self.decoder2(d3, skip2)
        d1 = self.decoder1(d2, skip1)
        logits = self.seg_head(d1)
        return F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=True)


def build_model(model_config: dict, in_channels: int, out_channels: int) -> SlimFormer:
    mixer_name = model_config.get("token_mixer", "mamba").lower()
    mixer_map = {
        "mamba": ClassMamba,
        "attention": ClassAttention,
        "rnn": ClassRNN,
    }
    if mixer_name not in mixer_map:
        raise ValueError(f"Unsupported token_mixer: {mixer_name}")

    return SlimFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=tuple(model_config["patch_size"]),
        base_channels=int(model_config["base_channels"]),
        num_heads=tuple(model_config["num_heads"]),
        token_mixer=mixer_map[mixer_name],
        drop=float(model_config["drop"]),
        drop_path=float(model_config["drop_path"]),
        metaformer_layers=tuple(model_config["metaformer_layers"]),
        prune_flags_list=model_config.get("prune_flags_list"),
    )
