import torch.nn as nn
import torch

from einops import rearrange

try:
    import xformers.ops

    use_xformers = True
except ImportError:
    import warnings

    warnings.warn("Could not import xformers, using default")
    use_xformers = False


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SpatioTemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm_t = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv_t = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_s = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out_t = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.to_out_s = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        if use_xformers:
            return self.x_forward(x)
        else:
            return self.vanilla_forward(x)

    def vanilla_forward(self, x):
        # Temporal attention
        x = self.norm_t(x)
        qkv = self.to_qkv_t(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t c d -> b c t d"), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self.to_out_t(out)

        # Spatial attention
        out = self.norm_s(out)
        qkv = self.to_qkv_s(out).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b c t d -> b t c d"), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        return self.to_out_s(out)

    def x_forward(self, x):
        # Temporal attention
        x = self.norm_t(x)
        q, k, v = self.to_qkv_t(x).chunk(3, dim=-1)
        out = xformers.ops.memory_efficient_attention(q, k, v)
        out = self.to_out_t(out)

        # Spatial attention
        out = self.norm_s(out)
        qkv = self.to_qkv_s(out).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t c d -> b c t d"), qkv)
        out = xformers.ops.memory_efficient_attention(q, k, v)
        out = rearrange(out, "b c t d -> b t c d")
        return self.to_out_s(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SpatioTemporalAttention(
                            dim, heads=heads, dim_head=dim_head, dropout=dropout
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
