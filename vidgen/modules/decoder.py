from .layers import ResidualBlock, Upsample, get_activation, get_norm

import torch.nn.functional as F
import torch.nn as nn
import torch


class Decoder(nn.Module):
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
        self.ups = nn.ModuleList()

        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                now_channels = out_channels

        self.activation = activation
        self.initial_pad = initial_pad
        self.output_activation = None
        if output_activation:
            self.output_activation = get_activation(output_activation)

        self.decoder_conv_in = torch.nn.ConvTranspose2d(
            latent_channels,
            now_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(
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

            if i != 0:
                self.ups.append(Upsample(now_channels))

        self.out_norm = get_norm(norm, now_channels, num_groups)
        self.out_conv = nn.Conv2d(now_channels, img_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_conv_in(x)

        for layer in self.ups:
            x = layer(x)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        if self.initial_pad != 0:
            ip = self.initial_pad
            return x[:, :, ip:-ip, ip:-ip]

        if self.output_activation:
            return self.output_activation(x)
        return x
