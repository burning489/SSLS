import torch
from torch import Tensor

from .base import Measurement


class AveragePooling(Measurement):
    """Average pooling observation.

    Args:
        noise_std (float): Noise level of observation Gaussian noise.
        kernel_size (int): Kernel size of the average pooling.
    """

    def __init__(self, noise_std: float, kernel_size: int):
        super().__init__(noise_std)
        self.kernel_size = kernel_size

    def _measure(self, x: Tensor) -> Tensor:
        *batch, h, w = x.shape
        x = x.reshape(*batch, h // self.kernel_size, self.kernel_size, w // self.kernel_size, self.kernel_size)
        x = x.mean(dim=(-3, -1))
        return x

    def measure(self, x: Tensor) -> Tensor:
        y = self._measure(x)
        noise = torch.randn_like(y, device=x.device) * self.noise_std
        return y + noise
