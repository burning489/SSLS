import torch
from torch import Tensor

from .base import Measurement


class CenterMask(Measurement):
    """Center mask observation model.

    Examples:
        xxxx      xxxx
        xxxx ---> x  x
        xxxx ---> x  x
        xxxx      xxxx
    """

    def __init__(self, noise_std):
        super().__init__(noise_std)

    def _measure(self, x: Tensor) -> Tensor:
        *batch, h, w = x.shape
        mask = torch.ones_like(x, device=x.device)
        mask[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0
        x = x * mask
        return x

    def measure(self, x: Tensor) -> Tensor:
        y = self._measure(x)
        noise = torch.randn_like(y, device=x.device) * self.noise_std
        return y + noise
