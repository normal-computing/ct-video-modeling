from vidgen.modules.blocks import EncoderBlock, DecoderBlock
from vidgen.utils import instantiate_from_config

from torchdiffeq import odeint
from torchcde import (
    hermite_cubic_coefficients_with_backward_differences,
    CubicSpline,
)

from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch


class CDEFunction(nn.Module):
    def __init__(self, in_channels, base_channels, channel_mults):
        super().__init__()
        in_channels = in_channels * 2

        now_channels = base_channels
        channels = [in_channels * 2, now_channels]
        for i in channel_mults:
            now_channels *= i
            channels.append(now_channels)
        channels.append(in_channels)

        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            if i < len(channels) - 2:
                layers.append(nn.SiLU())
        self.func = nn.Sequential(*layers)

    def set_shape(self, Y):
        *_, c, h, w = Y.shape
        self.shape = (c * 2, h, w)
        self.spline_shape = (c, h, w)

    def fit_spline(self, Y, ts):
        Y = Y.flatten(start_dim=2)
        self.static = CubicSpline(
            hermite_cubic_coefficients_with_backward_differences(Y, t=ts), t=ts
        )

    def forward(self, t, z):
        z = z.unflatten(1, self.shape)
        d = self.static.derivative(t).unflatten(1, self.spline_shape)
        e = self.static.evaluate(t).unflatten(1, self.spline_shape)
        p = self.func(torch.cat([z, e, d], dim=1)).flatten(start_dim=1)
        return p


class MaskEncoder(nn.Module):
    def __init__(self, base_channels, latent_channels, channel_mults):
        super().__init__()

        self.first_layer = nn.Conv2d(1, base_channels, 3, 1, 1)
        channels = [base_channels * mult for mult in [1] + channel_mults]

        self.encoder = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder.append(EncoderBlock(channels[i], channels[i + 1]))

        self.out_layer = nn.Conv2d(channels[-1], latent_channels, 3, 1, 1)

    def forward(self, m):
        m = self.first_layer(m)
        for block in self.encoder:
            m = block(m)
        return self.out_layer(m)


class ContinuousTimeUNet(nn.Module):
    def __init__(
        self,
        function,
        mask_encoder,
        in_channels,
        base_channels,
        latent_channels,
        channel_mults,
        output_channels=None,
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.1,
        norm="gn",
        num_groups=32,
        integration_method="midpoint",
        atol=1e-2,
        rtol=1e-2,
    ):
        super().__init__()

        self.function = instantiate_from_config(function)
        self.mask_encoder = instantiate_from_config(mask_encoder)

        now_channels = base_channels
        self.first_layer = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        self.encoder = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            last_channels = latent_channels if i == len(channel_mults) - 1 else None
            self.encoder.append(
                EncoderBlock(
                    now_channels,
                    out_channels,
                    latent_channels=last_channels,
                    num_res_blocks=num_res_blocks,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
            now_channels = out_channels

        self.mid_layer = nn.Conv2d(latent_channels * 2, now_channels, 3, 1, 1)

        self.decoder = nn.ModuleList()
        channel_mults_reversed = list(reversed(channel_mults))
        mirror_channel_mults = channel_mults_reversed[1:] + [1]
        for i, mult in enumerate(mirror_channel_mults):
            out_channels = base_channels * mult
            if i > 0:
                now_channels *= 2
            self.decoder.append(
                DecoderBlock(
                    now_channels,
                    out_channels,
                    num_res_blocks=num_res_blocks,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
            now_channels = out_channels

        if output_channels is None:
            output_channels = in_channels
        self.last_layer = nn.Conv2d(now_channels, output_channels, 3, 1, 1)

        self.integration_method = integration_method
        self.atol = atol
        self.rtol = rtol

    def encode(self, block, x, ts=None, knot_ts=None, ret_skips=True):
        b, t, *_ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        z = rearrange(block(x), "(b t) c h w -> b t c h w", b=b, t=t)

        if ret_skips:
            *_, c, h, w = z.shape
            spline = CubicSpline(
                hermite_cubic_coefficients_with_backward_differences(
                    z.flatten(start_dim=2), t=knot_ts
                ),
                t=knot_ts,
            )
            ct_skips = [
                torch.cat([spline.evaluate(t).unflatten(1, (c, h, w))], dim=1)
                for t in ts
            ]
            ct_skips = torch.stack(ct_skips, dim=1)

            return z, ct_skips

        return z

    def decode(self, block, x):
        b, t, *_ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        z = rearrange(block(x), "(b t) c h w -> b t c h w", b=b, t=t)
        return z

    def forward(self, z, m0, solver_ts):
        skips = []
        z = self.encode(self.first_layer, z, ret_skips=False)
        for i, block in enumerate(self.encoder):
            if i < len(self.encoder) - 1:
                z, sc = self.encode(block, z, solver_ts)
                skips.append(sc)
            else:
                z = self.encode(block, z, ret_skips=False)

        *_, ch, height, width = z.shape
        self.function.set_shape(z)
        self.function.fit_spline(z, solver_ts)

        m0 = torch.cat([self.mask_encoder(m0), z[:, 0]], dim=1)
        z = (
            odeint(
                self.function,
                m0.flatten(start_dim=1),
                solver_ts,
                rtol=self.rtol,
                atol=self.atol,
                method=self.integration_method,
            )
            .swapaxes(0, 1)
            .unflatten(2, (ch * 2, height, width))
        )

        z = self.decode(self.mid_layer, z)
        for i, block in enumerate(self.decoder):
            if i > 0:
                sc = skips.pop()
                z = self.decode(block, torch.cat([z, sc], dim=2))
            else:
                z = self.decode(block, z)

        y = self.decode(self.last_layer, z)
        return y
