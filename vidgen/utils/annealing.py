import torch.nn as nn
import math


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    Parameters:
        total_steps (int): Number of epochs to reach full KL divergence weight.
        shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
    """

    def __init__(
        self,
        total_steps: int,
        shape: str,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        disable=False,
        **kwargs
    ):
        self.total_steps = total_steps
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.current_step = 0
        if not disable:
            self.shape = shape
        else:
            self.shape = "none"

        self.logistic_smoothness = kwargs.get("logistic_smoothness", 1.0)

    def __call__(self):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        return self.slope()

    def slope(self):
        if self.shape == "linear":
            slope = self.current_step / self.total_steps
        elif self.shape == "cosine":
            slope = 0.5 + 0.5 * math.cos(
                math.pi * (self.current_step / self.total_steps - 1)
            )
        elif self.shape == "logistic":
            exponent = self.logistic_smoothness * (
                (self.total_steps / 2) - self.current_step
            )
            slope = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            slope = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        return max(self.min_weight, slope * self.max_weight)

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        else:
            pass


# class Annealer(nn.Module):
#     """
#     This class is used to anneal the KL divergence loss over the course of training VAEs.
#     After each call, the step() function should be called to update the current epoch.
#     Parameters:
#         total_steps (int): Number of epochs to reach full KL divergence weight.
#         shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
#     """

#     def __init__(self, min_weight: int, shape: str, disable=False, **kwargs):
#         super().__init__()
#         self.min_weight = min_weight
#         self.current_step = 0
#         if not disable:
#             self.shape = shape
#         else:
#             self.shape = "none"

#         self.logistic_smoothness = kwargs.get("logistic_smoothness", 0.1)
#         self.cosine_smoothness = kwargs.get("cosine_smooothness", 0.1)
#         self.linear_smoothness = kwargs.get("linear_smoothness", 0.1)

#     def __call__(self):
#         """
#         Args:
#             kld (torch.tensor): KL divergence loss
#         Returns:
#             out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
#         """
#         return self.slope()

#     def slope(self):
#         step_ratio = 1.0 / (self.current_step + 1)
#         if self.shape == "linear":
#             slope = max(step_ratio, self.min_weight)
#         elif self.shape == "cosine":
#             slope = 0.5 + 0.5 * math.cos(
#                 math.pi * self.current_step * self.cosine_smoothness
#             )
#             slope = max(slope, self.min_weight)
#         elif self.shape == "logistic":
#             exponent = self.current_step * self.logistic_smoothness
#             slope = 2 / (1 + math.exp(exponent))
#             slope = max(slope, self.min_weight)
#         elif self.shape == "none":
#             slope = 1.0
#         else:
#             raise ValueError(
#                 "Invalid shape for annealing function. Must be linear, cosine, or logistic."
#             )
#         return slope

#     def step(self):
#         self.current_step += 1
