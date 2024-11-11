import torch
from torch import Tensor

from .base import Measurement


class GridMask(Measurement):
    """Grid mask observation model.

    Args:
        noise_std (float): Noise level of observation Gaussian noise.
        stride (int): Stride of the observed point per direction.

    Examples:
        xxxxx      x x x
        xxxxx
        xxxxx ---> x x x
        xxxxx
        xxxxx      x x x

    TODO:
        Fix magic number 128.
    """

    def __init__(self, noise_std: float, stride: int):
        super().__init__(noise_std)
        mask = torch.zeros((128, 128))
        mask[::stride, ::stride] = 1.0
        self.mask = mask

    def _measure(self, x: Tensor) -> Tensor:
        mask = self.mask.to(x.device)
        x = x * mask
        return x

    def measure(self, x: Tensor) -> Tensor:
        y = self._measure(x)
        noise = torch.randn_like(y, device=x.device) * self.noise_std
        return y + noise
