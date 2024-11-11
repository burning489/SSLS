from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor


def assemble_grad_potential(
    y: Tensor,
    score_likelihood: Callable[[Tensor, Tensor], Tensor],
    score_prior: Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    """Gradient of potential of the posterior distribution.

    Args:
        y (Tensor): Measurements.
        score_likelihood (Callable[[Tensor, Tensor], Tensor]):
            Function handle to compute score of likelihood,
            takes in (state, measurement) pair,
            returns corresponding score of likelihood.
        score_prior (Callable[[Tensor], Tensor]):
            Function handle to compute prior,
            takes in state and returns corresponding score of prior.

    Returns:
        Callable[[Tensor], Tensor]:
            Function handle that computes the gradient of posterior potential.
    """
    return lambda x: -score_likelihood(x, y) - score_prior(x)


@torch.no_grad()
def langevin_sampler(
    grad_potential_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    steps: int,
    dt: float,
    anneal_init: float,
    anneal_decay: float,
    anneal_steps: int,
) -> Tensor:
    """PyTorch version unadjusted Langevin algorithm with annealing.

    Args:
        grad_potential_fn (Callable[[Tensor], Tensor]):
            Function handle to compute the gradient of potential.
        x (Tensor): (n, *state_shape), spatial state.
        steps (int): Number of steps.
        dt (float): Uniform stepsize.
        anneal_init (float): Initial temperature for anealing LMC.
        anneal_decay (float): Temperature decay rate for anealing LMC.
        anneal_steps (int): Number of Temperature decays for anealing LMC.

    Returns:
        Tensor: (n, *state_shape), samples derived by the unadjusted Langevin algorithm.
    """
    # for _ in range(steps):
    #     x = x - grad_potential_fn(x) * dt + torch.randn_like(x, device=x.device) * (2 * dt) ** 0.5
    # return x
    phi = lambda x: (np.exp(x) - 1) / x
    lam = anneal_init
    gamma = anneal_decay
    for _ in range(anneal_steps):
        for _ in range(steps):
            noise = torch.randn_like(x, device=x.device) * (2 * dt * phi(-2 * lam * dt)) ** 0.5
            grad = grad_potential_fn(x)
            grad_norm = torch.sqrt(torch.mean(grad**2, dim=tuple(range(1, x.ndim)), keepdim=True))
            adj_ratio = torch.ones_like(grad_norm, device=x.device)
            tol = 10.0
            adj_ratio[grad_norm > tol] = tol / grad_norm[grad_norm > tol]
            grad = grad * adj_ratio
            x = np.exp(-lam * dt) * x - dt * phi(-lam * dt) * grad + noise
        lam = gamma * lam
    return x


def plot_lorenz_trajectory(states, assimilated_states, steps):
    mpl.rc("mathtext", fontset="cm")
    mpl.rc("font", family="serif", serif="DejaVu Serif")
    mpl.rc("figure", dpi=100, titlesize=9)
    mpl.rc("figure.subplot", wspace=0.2)
    mpl.rc("axes", grid=False, titlesize=6, labelsize=6, labelpad=0)
    mpl.rc("axes.spines", top=False, right=False)
    mpl.rc("xtick", labelsize=6, direction="in")
    mpl.rc("ytick", labelsize=6, direction="in")
    mpl.rc("xtick.major", pad=2)
    mpl.rc("ytick.major", pad=2)
    mpl.rc("grid", linestyle=":", alpha=0.8)
    mpl.rc("lines", linewidth=1, markersize=2)
    mpl.rc("legend", fontsize=9)
    dt = 1 / (states.shape[0] - 1)
    states = states.cpu()[: steps + 1]  # (steps, dim)
    assimilated_states = assimilated_states.cpu()[: steps + 1]  # (steps, nsample, dim)
    median_estimation = torch.median(assimilated_states, dim=1)[0]  # (steps, dim)
    _, dim = median_estimation.shape
    fig, axes = plt.subplots(nrows=1, ncols=dim, figsize=(8, 1.5))
    # fig.suptitle(f"Trajectory of Lorenz96$", y=1.2)
    t = np.arange(steps + 1) * dt
    for ncol in range(dim):
        ax = axes[ncol]
        ax.plot(t, states[:, ncol], label="Ground-truth", color="C0", marker="v", markevery=1)
        ax.plot(t, median_estimation[:, ncol], label="SSLS estimation", color="C1", marker="^", markevery=1)
        ax.grid(False)
    return fig


def plot_kolmogorov_vorticity(states, assimilated_states, steps):
    def vorticity(x):
        *batch, _, h, w = x.shape
        y = x.reshape(-1, 2, h, w)
        y = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode="circular")
        (du,) = torch.gradient(y[:, 0], dim=-1)
        (dv,) = torch.gradient(y[:, 1], dim=-2)
        y = du - dv
        y = y[:, 1:-1, 1:-1]
        y = y.reshape(*batch, h, w)
        return y

    mpl.rc("mathtext", fontset="cm")
    mpl.rc("font", family="serif", serif="DejaVu Serif")
    mpl.rc("figure", dpi=100, titlesize=9)
    mpl.rc("figure.subplot", wspace=0.4, hspace=0.5)
    mpl.rc("axes", grid=False, titlesize=6, labelsize=6, labelpad=0)
    mpl.rc("axes.spines", top=False, right=False)
    mpl.rc("xtick", labelsize=6, direction="in")
    mpl.rc("ytick", labelsize=6, direction="in")
    mpl.rc("xtick.major", pad=2)
    mpl.rc("ytick.major", pad=2)
    mpl.rc("grid", linestyle=":", alpha=0.8)
    mpl.rc("lines", linewidth=1, markersize=2)
    mpl.rc("legend", fontsize=9)
    states = states.cpu()[steps]  # (2, grid_size, grid_size)
    assimilated_states = assimilated_states.cpu()[steps]  # (n_sample, 2, grid_size, grid_size)
    median_estimation = torch.median(assimilated_states, dim=0)[0]  # (2, grid_size, grid_size)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))

    p1 = axes[0].imshow(vorticity(states), cmap=sns.cm.icefire)
    axes[0].set_title(r"True $\omega$")
    fig.colorbar(p1, ax=axes[0], shrink=0.3)

    p2 = axes[1].imshow(vorticity(median_estimation), cmap=sns.cm.icefire)
    axes[1].set_title(r"Estimated $\omega$")
    fig.colorbar(p2, ax=axes[1], shrink=0.3)

    for ax in axes.flatten():
        ax.axis("off")
    return fig
