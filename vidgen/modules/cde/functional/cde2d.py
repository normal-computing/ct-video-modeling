import torch
from torchcde.interpolation_linear import (
    linear_interpolation_coeffs,
    interpolation_base,
)


class EmptySpline(torch.nn.Module):
    def __init__(self, X):
        super().__init__()
        self.X = X.squeeze(1)

    def evaluate(self, t):
        return self.X

    def derivative(self, t):
        return torch.zeros_like(self.X)


class CubicSpline2D(interpolation_base.InterpolationBase):
    """Calculates the cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        x = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_coeffs(x)
        # ...at this point you can save coeffs, put it through PyTorch's Datasets and DataLoaders, etc...
        spline = CubicSpline(coeffs)
        point = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(point)
    """

    def __init__(self, coeffs, t=None, **kwargs):
        """
        Arguments:
            coeffs: As returned by `torchcde.natural_cubic_coeffs`.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
        """
        super(CubicSpline2D, self).__init__(**kwargs)

        if t is None:
            t = torch.linspace(
                0,
                coeffs.size(1),
                coeffs.size(1) + 1,
                dtype=coeffs.dtype,
                device=coeffs.device,
            )

        channels = coeffs.size(2) // 4
        if channels * 4 != coeffs.size(2):  # check that it's a multiple of 4
            raise ValueError("Passed invalid coeffs.")
        a, b, two_c, three_d = (
            coeffs[:, :, :channels],
            coeffs[:, :, channels : 2 * channels],
            coeffs[:, :, 2 * channels : 3 * channels],
            coeffs[:, :, 3 * channels :],
        )

        self.register_buffer("_t", t)
        self.register_buffer("_a", a)
        self.register_buffer("_b", b)
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self.register_buffer("_two_c", two_c)
        self.register_buffer("_three_d", three_d)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._b.dtype, device=self._b.device)
        maxlen = self._b.size(1) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = (
            0.5 * self._two_c[:, index, ...]
            + self._three_d[:, index, ...] * fractional_part / 3
        )
        inner = self._b[:, index, ...] + inner * fractional_part
        return self._a[:, index, ...] + inner * fractional_part

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = (
            self._two_c[:, index, ...] + self._three_d[:, index, ...] * fractional_part
        )
        deriv = self._b[:, index, ...] + inner * fractional_part
        return deriv


def _setup_hermite_cubic_coeffs_w_backward_differences(
    times, coeffs, derivs, device=None
):
    """Compute backward hermite from linear coeffs."""
    x_prev = coeffs[:, :-1, ...]
    x_next = coeffs[:, 1:, ...]
    # Let x_0 - x_{-1} = x_1 - x_0
    derivs_prev = torch.cat((derivs[:, [0], ...], derivs[:, :-1, ...]), axis=1)
    derivs_next = derivs
    x_diff = x_next - x_prev
    t_diff = (times[1:] - times[:-1]).view(-1, 1, 1, 1)
    # Coeffs
    a = x_prev
    b = derivs_prev
    two_c = 2 * (3 * (x_diff / t_diff - b) - derivs_next + derivs_prev) / t_diff
    three_d = (1 / t_diff**2) * (derivs_next - b) - (two_c) / t_diff
    coeffs = torch.cat([a, b, two_c, three_d], dim=2).to(device)
    return coeffs


def hermite_cubic_coefficients_with_backward_differences_2d(x, t=None):
    """Computes the coefficients for hermite cubic splines with backward differences.

    Arguments:
        As `torchcde.linear_interpolation_coeffs`.

    Returns:
        A tensor, which should in turn be passed to `torchcde.CubicSpline`.
    """
    # Linear coeffs
    *_, c, h, w = x.shape
    x = x.flatten(start_dim=2)
    coeffs = linear_interpolation_coeffs(x, t=t, rectilinear=None)
    coeffs = coeffs.unflatten(2, (c, h, w))

    if t is None:
        t = torch.linspace(
            0,
            coeffs.size(1) - 1,
            coeffs.size(1),
            dtype=coeffs.dtype,
            device=coeffs.device,
        )

    # Linear derivs
    derivs = coeffs[:, 1:, ...] - coeffs[:, :-1, ...] / (t[1:] - t[:-1]).view(
        -1, 1, 1, 1
    )

    # Use the above to compute hermite coeffs
    hermite_coeffs = _setup_hermite_cubic_coeffs_w_backward_differences(
        t, coeffs, derivs, device=coeffs.device
    )

    return hermite_coeffs
