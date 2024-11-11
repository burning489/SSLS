import torch
from torch import Tensor


class Dynamics:
    """Abastract dynamics class.

    Args:
        shape (tuple): Shape of per state.
    """

    def __init__(self, shape):
        self.shape = shape

    def prior(self, n_sample: int) -> Tensor:
        """
        Derive samples from the prior distribution.

        Args:
            n_sample (int): The number of samples to derive from the prior distribution.

        Returns:
            Tensor: (n_sample, *shape), prior samples, allocated on the default PyTorch device.
        """
        raise NotImplementedError

    def transition(self, x: Tensor) -> Tensor:
        """
        Transition function from current state x to the next state.

        Args:
            x (Tensor): (n, *shape), current spatial state.

        Returns:
            Tensor: (n, *shape), next spatial state, allocated on the same device as x.
        """
        raise NotImplementedError

    def generate(self, x0: Tensor, steps: int) -> Tensor:
        """
        Generate a trajectoy of spatial state.

        Args:
            x0 (Tensor): (*shape,), initial spatial state.
            steps (int): Number of steps of the dynamics.

        Returns:
            Tensor: (steps+1, *shape), trajectory of spatial states, allocated on the same device as x0.
        """
        device = x0.device
        _, *shape = x0.shape
        assert tuple(shape) == self.shape
        states = torch.empty((steps + 1, *shape), device=device)
        states[:1] = x0
        for i in range(steps):
            states[i + 1] = self.transition(states[i])
        return states
