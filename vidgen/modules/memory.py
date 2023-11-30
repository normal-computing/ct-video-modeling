from einops import rearrange
import torch.nn as nn
import torch


class ConvolutionalLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        projection_channels,
        kernel_size=7,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.x_function = nn.Conv2d(
            projection_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding="same",
        )
        self.h_function = nn.Conv2d(
            projection_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding="same",
        )
        self.h_out = None
        if projection_channels != hidden_channels:
            self.h_out = nn.Conv2d(
                hidden_channels,
                projection_channels,
                kernel_size=kernel_size,
                padding="same",
            )

    def forward(self, x_t, hidden):
        h_t, c_t = hidden

        ix_t, fx_t, gx_t, ox_t = self.x_function(x_t).split(self.hidden_channels, dim=1)
        ih_t, fh_t, gh_t, oh_t = self.h_function(h_t).split(self.hidden_channels, dim=1)
        i_t = torch.sigmoid(ix_t + ih_t)
        f_t = torch.sigmoid(fx_t + fh_t)
        g_t = torch.tanh(gx_t + gh_t)
        o_t = torch.sigmoid(ox_t + oh_t)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        if self.h_out:
            h_t = self.h_out(h_t)

        return h_t, (h_t, c_t)


class ConvolutionalLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        num_layers=1,
        projection_channels=None,
        kernel_size=11,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.projection_channels = projection_channels
        if projection_channels is None:
            self.projection_channels = hidden_channels

        self.x_in = nn.Conv2d(
            input_channels,
            projection_channels,
            kernel_size=kernel_size,
            padding="same",
        )

        cell_list = []
        in_channels = input_channels
        for i in range(num_layers):
            cell_list.append(
                ConvolutionalLSTMCell(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    projection_channels=projection_channels,
                )
            )
            if i == 0:
                in_channels = hidden_channels
        self.cells = nn.ModuleList(cell_list)

    def forward(self, x, hidden=None):
        if hidden is None:
            h_t = torch.zeros(
                (x.size(0),) + (self.projection_channels,) + (x.size(-2), x.size(-1)),
                dtype=x.dtype,
                device=x.get_device(),
            )
            c_t = torch.zeros(
                (x.size(0),) + (self.hidden_channels,) + (x.size(-2), x.size(-1)),
                dtype=x.dtype,
                device=x.get_device(),
            )
        else:
            h_t, c_t = hidden

        b, t, *_ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = rearrange(self.x_in(x), "(b t) c h w -> t b c h w", b=b, t=t)
        for cell in self.cells:
            hidden_states = []
            for x_t in x:
                o_t, (h_t, c_t) = cell(x_t, (h_t, c_t))
                hidden_states.append(o_t)
            x = hidden_states
        return o_t, (h_t, c_t)
