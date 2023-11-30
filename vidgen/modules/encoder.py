from .layers import ResidualBlock, Downsample, get_activation

import torch.nn.functional as F
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(
        self,
        img_channels,
        base_channels,
        latent_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        initial_pad=0,
        output_activation=None,
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad
        self.output_activation = None
        if output_activation:
            self.output_activation = get_activation(output_activation)

        now_channels = base_channels
        self.init_conv = nn.Conv2d(img_channels, now_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels,
                        out_channels,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))

        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=True,
                ),
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=False,
                ),
            ]
        )

        self.encoder_conv_out = torch.nn.Conv2d(
            now_channels,
            latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        x = self.init_conv(x)

        for layer in self.downs:
            x = layer(x)

        for layer in self.mid:
            x = layer(x)

        x = self.encoder_conv_out(x)
        if self.output_activation:
            return self.output_activation(x)
        return x
