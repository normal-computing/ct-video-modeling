from einops import rearrange

import torch.nn as nn
import torch.nn.functional as F


class SpatialDecoder(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super().__init__()

        self.layer_1 = nn.Conv3d(
            latent_channels,
            latent_channels,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )
        self.upsample_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.layer_2 = nn.Conv3d(
            latent_channels,
            in_channels * 64,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.layer_3 = nn.Conv3d(
            in_channels * 64,
            in_channels * 64,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )
        self.out_layer = nn.Conv3d(
            in_channels * 64,
            in_channels,
            (5, 3, 3),
            stride=1,
            padding=(2, 1, 1),
        )

    def forward(self, x):
        b, f, *_ = x.shape

        x = x.swapaxes(1, 2)
        x = F.relu(
            rearrange(
                self.upsample_1(rearrange(self.layer_1(x), "b c f h w -> (b f) c h w")),
                "(b f) c h w -> b c f h w",
                b=b,
                f=f,
            )
        )
        x = F.relu(
            rearrange(
                self.upsample_2(rearrange(self.layer_2(x), "b c f h w -> (b f) c h w")),
                "(b f) c h w -> b c f h w",
                b=b,
                f=f,
            )
        )
        return self.out_layer(F.relu(self.layer_3(x))).swapaxes(1, 2)


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels * 64,
                (5, 3, 3),
                stride=(1, 2, 2),
                padding=(2, 1, 1),
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels * 64,
                in_channels * 64,
                (5, 3, 3),
                stride=1,
                padding=(2, 1, 1),
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels * 64,
                latent_channels,
                (5, 3, 3),
                stride=(1, 2, 2),
                padding=(2, 1, 1),
            ),
            nn.ReLU(),
            nn.Conv3d(
                latent_channels,
                latent_channels,
                (5, 3, 3),
                stride=1,
                padding=(2, 1, 1),
            ),
        )

    def forward(self, x):
        x = x.swapaxes(1, 2)
        return self.spatial_encoder(x).swapaxes(1, 2)
