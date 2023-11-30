from vidgen.modules.blocks import EncoderBlock, LearnableDecoderBlock
from vidgen.utils import instantiate_from_config

from torchdiffeq import odeint
from torchcde import (
    # hermite_cubic_coefficients_with_backward_differences,
    natural_cubic_spline_coeffs,
    CubicSpline,
)

from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch


class CDEFunction(nn.Module):
    def __init__(self, in_channels, base_channels, channel_mults):
        super().__init__()
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
        # layers.append(nn.Tanh())
        self.func = nn.Sequential(*layers)

    def set_shape(self, Y):
        *_, c, h, w = Y.shape
        self.shape = (c, h, w)

    def fit_spline(self, Y, ts):
        Y = Y.flatten(start_dim=2)
        self.static = CubicSpline(natural_cubic_spline_coeffs(Y, t=ts), t=ts)

    def forward(self, t, z):
        z = z.unflatten(1, self.shape)
        d = self.static.derivative(t).unflatten(1, self.shape)
        # e = self.static.evaluate(t).unflatten(1, self.shape)
        return self.func(torch.cat([z, d], dim=1)).flatten(start_dim=1)


class ContinuousTimeUNet(nn.Module):
    def __init__(
        self,
        function,
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
        step_size=1e-2,
        integration_method="midpoint",
        atol=1e-2,
        rtol=1e-2,
    ):
        super().__init__()

        function_base_channels = function["params"]["in_channels"]

        now_channels = base_channels
        self.first_layer = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        self.encoder = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            self.encoder.append(
                EncoderBlock(
                    now_channels,
                    out_channels,
                    num_res_blocks=num_res_blocks,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
            self.encoder.append(
                nn.Conv2d(out_channels, function_base_channels, 3, 1, 1)
            )
            self.encoder.append(instantiate_from_config(function))
            now_channels = out_channels

        self.mid_layer = nn.Conv2d(latent_channels, now_channels, 3, 1, 1)

        self.decoder = nn.ModuleList()
        channel_mults_reversed = list(reversed(channel_mults))
        mirror_channel_mults = channel_mults_reversed[1:] + [1]
        for i, mult in enumerate(mirror_channel_mults):
            out_channels = base_channels * mult
            if i > 0:
                self.decoder.append(
                    nn.Conv2d(function_base_channels, now_channels, 3, 1, 1)
                )
                now_channels *= 2
            else:
                self.decoder.append(nn.Identity())
            self.decoder.append(
                LearnableDecoderBlock(
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
        self.step_size = step_size
        self.atol = atol
        self.rtol = rtol

    def encode(self, encoder, x, function=None, ts=None, knot_ts=None, ret_skips=True):
        b, t, *_ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        z = rearrange(encoder(x), "(b t) c h w -> b t c h w", b=b, t=t)

        if function:
            base_block, function = function
            sz = rearrange(
                base_block(rearrange(z, "b t c h w -> (b t) c h w")),
                "(b t) c h w -> b t c h w",
                b=b,
                t=t,
            )

            *_, ch, height, width = sz.shape
            function.set_shape(sz)
            function.fit_spline(sz, knot_ts)

            z_T = (
                odeint(
                    function,
                    sz[:, 0].flatten(start_dim=1),
                    ts,
                    rtol=self.rtol,
                    atol=self.atol,
                    method=self.integration_method,
                    options={"step_size": self.step_size},
                )
                .swapaxes(0, 1)
                .unflatten(2, (ch, height, width))
            )

            if ret_skips:
                return z, z_T
            return z_T
        return z

    def decode(self, block, x):
        b, t, *_ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        z = rearrange(block(x), "(b t) c h w -> b t c h w", b=b, t=t)
        return z

    def forward(self, z, solver_ts, spline_ts=None):
        if spline_ts is None:
            spline_ts = solver_ts

        skips = []
        z = self.encode(self.first_layer, z, ret_skips=False)
        for i in range(0, len(self.encoder), 3):
            encoder_block = self.encoder[i]
            function = self.encoder[i + 1 : i + 3]
            if i < len(self.encoder) - 3:
                z, sc = self.encode(
                    encoder_block,
                    z,
                    function=function,
                    ts=solver_ts,
                    knot_ts=spline_ts,
                )
                skips.append(sc)
            else:
                z = self.encode(
                    encoder_block,
                    z,
                    function=function,
                    ts=solver_ts,
                    knot_ts=spline_ts,
                    ret_skips=False,
                )

        z = self.decode(self.mid_layer, z)
        for i in range(0, len(self.decoder), 2):
            base_block, block = self.decoder[i : i + 2]
            if i > 0:
                sc = self.decode(base_block, skips.pop())
                z = self.decode(block, torch.cat([z, sc], dim=2))
            else:
                z = self.decode(block, z)

        return self.decode(self.last_layer, z)
