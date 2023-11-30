from .layers import Downsample, ResidualBlock, Upsample, LearnableUpsample

import torch.nn.functional as F
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels=None,
        down_block=True,
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.1,
        norm="gn",
        num_groups=32,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        now_channels = in_channels
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
            now_channels = out_channels

        self.down_block = None
        if down_block:
            self.down_block = Downsample(out_channels)

        self.last_layer = None
        if latent_channels:
            self.last_layer = nn.Conv2d(out_channels, latent_channels, 3, 1, 1)
            now_channels = latent_channels

    def forward(self, z):
        for block in self.res_blocks:
            z = block(z)
        if self.down_block:
            z = self.down_block(z)
        if self.last_layer:
            z = self.last_layer(z)

        return z


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up_block=True,
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.1,
        norm="gn",
        num_groups=32,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        now_channels = in_channels
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
            now_channels = out_channels

        self.up_block = None
        if up_block:
            self.up_block = Upsample(out_channels)

    def forward(self, z):
        for block in self.res_blocks:
            z = block(z)
        if self.up_block:
            z = self.up_block(z)

        return z


class LearnableDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up_block=True,
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.1,
        norm="gn",
        num_groups=32,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        now_channels = in_channels
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
            now_channels = out_channels

        self.up_block = None
        if up_block:
            self.up_block = LearnableUpsample(out_channels)

    def forward(self, z):
        for block in self.res_blocks:
            z = block(z)
        if self.up_block:
            z = self.up_block(z)

        return z
