import numpy as np
import torch
from torch import Tensor

from .base import Measurement


class RandomMask(Measurement):
    """Random mask observation model.

    Args:
        noise_std (float): Noise level of observation Gaussian noise.
        sparsity (float): Sparsity level of the mask strategy.

    TODO:
        Fix magic number 128.
    """

    def __init__(self, noise_std, sparsity):
        super().__init__(noise_std)
        total_elements = 128**2
        non_zero_elements = int(total_elements * (1 - sparsity))
        mask = np.ones(total_elements)
        mask[non_zero_elements:] = 0
        np.random.RandomState(0).shuffle(mask)
        self.mask = torch.from_numpy(mask.reshape(128, 128)).float()

    def _measure(self, x: Tensor) -> Tensor:
        mask = self.mask.to(x.device)
        x = x * mask
        return x

    def measure(self, x: Tensor) -> Tensor:
        y = self._measure(x)
        noise = torch.randn_like(y, device=x.device) * self.noise_std
        return y + noise
