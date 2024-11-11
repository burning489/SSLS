import torch
from torch import Tensor


class Measurement:
    """Abstract measurement class.

    Args:
        noise_std (float): Noise level (standard deviation) of the observation Gaussian noise.
    """

    def __init__(self, noise_std: float):
        self.noise_std = noise_std

    def _measure(self, x: Tensor) -> Tensor:
        """Clean measurement model without noise.

        Args:
            x (Tensor): (*shape), spatial state.

        Returns:
            Tensor: Measurement on x witnout noise.
        """
        raise NotImplementedError

    def _grad_measure(self, x: Tensor) -> Tensor:
        """Gradient of the clean measurement model.

        Args:
            x (Tensor): (*shape), spatial state.

        Returns:
            Tensor: The gradient of clean measurement w.r.t. x.
        """
        raise NotImplementedError

    def measure(self, x: Tensor) -> Tensor:
        """Noisy measurement model.

        Args:
            x (Tensor): (*shape), spatial state.

        Returns:
            Tensor: Measurement with noise (with noise level noise_std).
        """
        raise NotImplementedError

    def score_likelihood(self, x: Tensor, y: Tensor) -> Tensor:
        """Gradient log of the likelihood function (w.r.t. x).

        Args:
            x (Tensor): (*shape), spatial state.

        Returuns:
            Tensor.
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            mx = self._measure(x)
            log_p = -((y - mx) ** 2) / (2 * self.noise_std**2)
            return torch.autograd.grad(log_p.sum(), x)[0]
