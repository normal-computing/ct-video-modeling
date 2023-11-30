from einops import rearrange

import torch.nn.functional as F
import torch.nn as nn
import torch

try:
    import xformers.ops

    use_xformers = True
except ImportError:
    import warnings

    warnings.warn("Could not import xformers, using default")
    use_xformers = False


def get_activation(activation):
    activation_dict = {"sigmoid": F.sigmoid, "tanh": F.tanh}
    assert activation in activation_dict.keys(), "Invalid activation function"
    return activation_dict[activation]


def init_linear(inp, out, std=1e-8):
    ret = nn.Linear(inp, out)
    nn.init.normal_(ret.weight, std=std)
    nn.init.normal_(ret.bias, std=std)
    return ret


def memory_efficient_operation(layer, x, operating_bs=8):
    operated = []
    for i in range(0, x.size(0), operating_bs):
        operated.append(layer(x[i : i + operating_bs]))
    return torch.cat(operated, dim=0)


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


def add_position_encoding(x, t, n=10000):
    assert x.dim() in [2, 4]
    needs_reshaping = False
    if x.dim() == 4:
        _, c, h, w = x.shape
        x = x.flatten(start_dim=1)
        needs_reshaping = True
        d = c * h * w
    else:
        _, d = x.shape

    device = x.get_device()
    if device < 0:
        device = "cpu"

    position = torch.arange(int(d // 2) + 1, dtype=x.dtype, device=device)
    pe = torch.zeros((d,), dtype=x.dtype, device=device)
    denominator = 1 / torch.pow(n, 2 * position / d)
    pe[1::2] = torch.cos(t * denominator[:-1])
    if d % 2 == 0:
        denominator = denominator[:-1]
    pe[0::2] = torch.sin(t * denominator)
    y = x + pe

    if needs_reshaping:
        y = y.unflatten(1, (c, h, w))
    return y


class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.

    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.upsample(x)


class ResidualAttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        if use_xformers:
            return self.x_forward(x)
        else:
            return self.vanilla_forward(x)

    def vanilla_forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c).contiguous()
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c).contiguous()

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        return self.to_out(out) + x

    def x_forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        assert out.shape == (b, c, h, w)
        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    __doc__ = r"""Applies two conv blocks with residual connection.

    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        activation=F.relu,
        norm="gn",
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.attention = (
            nn.Identity()
            if not use_attention
            else ResidualAttentionBlock(out_channels, norm, num_groups)
        )

    def forward(self, x):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out


class TemporalConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        channel_mults=(1, 4),
        kernel_size=7,
    ):
        super().__init__()

        channels = [in_channels]
        for mult in channel_mults:
            channels.append(hidden_channels * mult)
        channels.append(in_channels)

        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv3d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            if i < len(channels) - 2:
                layers.append(nn.SiLU())

        self.model = nn.Sequential(*layers)

    def forward(self, sequence):
        return rearrange(
            self.model(rearrange(sequence, "b t c h w -> b c t h w")),
            "b c t h w -> b t c h w",
        )


class TemporalLinearAttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, hidden_dim, norm="gn", num_groups=32):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.norm = get_norm(norm, hidden_dim, num_groups)
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        if use_xformers:
            return self.x_forward(x)
        else:
            return self.vanilla_forward(x)

    def vanilla_forward(self, x):
        b, t, l = x.shape
        q, k, v = torch.split(
            self.to_qkv(self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)),
            self.hidden_dim,
            dim=-1,
        )
        q, v = q.swapaxes(1, -1), v.swapaxes(1, -1)

        dot_products = torch.bmm(q, k) * (l ** (-0.5))
        assert dot_products.shape == (b, l, l)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, l, t)
        out = out.swapaxes(1, -1)

        return self.to_out(out) + x

    def x_forward(self, x):
        b, t, l = x.shape
        q, k, v = torch.split(
            self.to_qkv(self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)),
            self.hidden_dim,
            dim=-1,
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        assert out.shape == (b, t, l)
        return self.to_out(out) + x


class TemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        proj_channels,
        num_heads=8,
        norm="gn",
        num_groups=32,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(
            in_channels,
            num_heads * hidden_channels * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.to_out = nn.Conv2d(
            hidden_channels * num_heads,
            proj_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def vanilla_forward(self, q, k, v):
        if q.dim() == 4:
            *_, h, l = q.shape
        else:
            assert q.dim() == 3
            q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-2)
        q, v = q.swapaxes(1, -1), v.swapaxes(1, -1)

        scores = []
        for i in range(h):
            qi, ki, vi = q[..., i, :], k[..., i, :], v[..., i, :]
            qk = torch.bmm(qi, ki) * (l ** (-0.5))
            attention = torch.softmax(qk, dim=-1)
            scores.append(torch.bmm(attention, vi).unsqueeze(-2))
        scores = torch.cat(scores, dim=-2)

        return scores.swapaxes(1, -1)

    def forward(self, sequence):
        b, t, _, h, w = sequence.shape

        sequence = rearrange(sequence, "b t c h w -> (b t) c h w")
        sequence = rearrange(
            self.to_qkv(self.norm(sequence)), "(b t) c h w -> (b h w) t c", t=t
        )
        q, k, v = torch.split(sequence, self.hidden_channels * self.num_heads, dim=-1)
        target_shape = (
            q.size(0),
            q.size(1),
            self.num_heads,
            self.hidden_channels,
        )  # (b h w), t, h, c
        q, k, v = (
            q.view(*target_shape).contiguous(),
            k.view(*target_shape).contiguous(),
            v.view(*target_shape).contiguous(),
        )

        if use_xformers:
            score = xformers.ops.memory_efficient_attention(q, k, v)
        else:
            score = self.vanilla_forward(q, k, v)

        # Project scores over heads dimension
        score = self.to_out(
            rearrange(score, "(b h w) t heads c -> (b t) (heads c) h w", h=h, w=w)
        )
        score = rearrange(score, "(b t) c h w -> b t c h w", b=b, t=t)

        return score


if __name__ == "__main__":
    # attn_t_block = TemporalAttentionBlock(3, 4, 8, num_groups=1).cuda()
    # conv_t_block = TemporalConvolutionBlock(3, 4, (1, 4)).cuda()
    # print(conv_t_block(torch.zeros((10, 3, 3, 10, 10)).cuda()).shape)

    attn_t_block = TemporalLinearAttentionBlock(64, norm=None).cuda()
    print(attn_t_block(torch.zeros((10, 3, 64)).cuda()).shape)
