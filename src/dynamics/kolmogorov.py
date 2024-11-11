import math

import jax
import jax.numpy as jnp
import jax.random as jrn
import jax_cfd.base as cfd
import numpy as np
import torch
from torch import Tensor

from .base import Dynamics


class KolmogorovFlow(Dynamics):
    """Kolmogorov flow dynamics.
    Reference: https://github.com/francois-rozet/sda/

    Args:
        grid_size (int): Size of per edge of the spatial grid.
        reynolds (float): Reynolds number.
        dt (float): Time steps intervals between observations.
        seed (int): RNG seed for jax (to generate initial prior states).
    """

    def __init__(
        self,
        grid_size: int = 128,
        reynolds: float = 1e3,
        dt: float = 0.2,
        seed: int = 42,
    ):
        super().__init__(shape=(2, grid_size, grid_size))
        self.seed = seed
        grid = cfd.grids.Grid(
            shape=(grid_size, grid_size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )
        bc = cfd.boundaries.periodic_boundary_conditions(2)
        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=1.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type="kolmogorov",
        )
        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds,
        )
        steps = 1 if dt_min > dt else math.ceil(dt / dt_min)
        step_fn = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=dt / steps,
                density=1.0,
                viscosity=1 / reynolds,
            ),
            steps=steps,
        )

        def _prior(key):
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=grid,
                maximum_velocity=3.0,
                peak_wavenumber=4.0,
            )
            return jnp.stack((u.data, v.data))

        self._prior = jax.jit(jnp.vectorize(_prior, signature="(K)->(C,H,W)"))

        def _transition(uv):
            u, v = cfd.initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=grid,
                bcs=(bc, bc),
            )
            u, v = step_fn((u, v))
            return jnp.stack((u.data, v.data))

        self._transition = jax.jit(jnp.vectorize(_transition, signature="(C,H,W)->(C,H,W)"))

    def prior(self, n_sample):
        key = jrn.PRNGKey(self.seed)
        keys = jrn.split(key, n_sample)
        x = np.array(self._prior(keys))
        return torch.tensor(x)

    def transition(self, x: Tensor) -> Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = np.array(self._transition(x))
        return torch.tensor(x, device=device)
