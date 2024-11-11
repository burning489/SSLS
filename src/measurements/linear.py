import torch
from torch import Tensor

from .base import Measurement


class Linear(Measurement):
    """Linear (Identity) observation model.

    Args:
        noise_std (float): Noise level of the observation Gaussian noise.
    """

    def __init__(self, noise_std: float):
        super().__init__(noise_std)

    def _measure(self, x):
        return x

    def _grad_measure(self, x: Tensor) -> Tensor | float:
        return 1.0

    def measure(self, x: Tensor) -> Tensor:
        return self._measure(x) + torch.randn_like(x, device=x.device) * self.noise_std

    def score_likelihood(self, x: Tensor, y: Tensor) -> Tensor:
        return self._grad_measure(x) * (y - self._measure(x)) / self.noise_std**2
