import torch
from torch import Tensor

from .base import Dynamics


class Lorenz96(Dynamics):
    """Lorenz 96 dynamics with periodic boundary conditions.
    Reference: https://en.wikipedia.org/wiki/Lorenz_96_model.

    Args:
        dim (int): State dimension.
        prior_mean (float): Mean of prior Gaussian distribution.
        prior_std (float): Standard deviation of prior Gaussian distribution.
        dt (float): Time steps of dynamics.
        forcing (float): Lorenz forcing.
        perturb_std (float): Standard deviation of perturbation Gaussian noise at each transition.
        solver (str): ODE solver, support "Euler", "Heun" and "Runge-Kutta".
    """

    def __init__(
        self,
        dim: int = 10,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        dt: float = 0.01,
        forcing: float = 8.0,
        perturb_std: float = 0.1,
        solver: str = "Euler",
    ):
        super().__init__(shape=(dim,))
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.dt = dt
        self.forcing = forcing
        self.perturb_std = perturb_std
        solvers = {
            "Euler": self._euler,
            "Heun": self._heun,
            "Runge-Kutta": self._runge_kutta,
        }
        self.step_fn = solvers[solver]

    def prior(self, n_sample):
        return self.prior_mean + torch.randn((n_sample, *self.shape)) * self.prior_std

    def f(self, x: Tensor):
        """
        Update rule of Lorenz96 model.
        Args:
            x (Tensor): (n, dim), current state.

        Returns:
            Tensor: (n, dim), update term of Lorenz96 model."""
        x_p1, x_m2, x_m1 = [torch.roll(x, i, dims=-1) for i in [-1, 2, 1]]
        return (x_p1 - x_m2) * x_m1 - x + self.forcing

    def transition(self, x):
        noise = torch.randn_like(x, device=x.device)
        return x + self.dt * self.step_fn(x) + self.perturb_std * noise * self.dt ** (0.5)

    def _euler(self, x: Tensor):
        """
        Euler method.

        Args:
            x (Tensor): (n, dim), current state.

        Returns:
            Tensor: (n, dim), estimated velocity function by forward Euler's method.
        """
        return self.f(x)

    def _heun(self, x: Tensor):
        """
        Heun method.

        Args:
            x (Tensor): (n, dim), current state.

        Returns:
            Tensor: (n, dim), estimated velocity function by Heun's method.
        """
        k1 = self.f(x)
        k2 = self.f(x + self.dt * k1)
        return (k1 + k2) / 2

    def _runge_kutta(self, x: Tensor):
        """
        Runge-Kutta method.

        Args:
            x (Tensor): (n, dim), current state.

        Returns:
            Tensor: (n, dim), estimated velocity function by Runge-Kutta method.
        """
        k1 = self.f(x)
        k2 = self.f(x + self.dt * k1 / 2)
        k3 = self.f(x + self.dt * k2 / 2)
        k4 = self.f(x + self.dt * k3)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6
