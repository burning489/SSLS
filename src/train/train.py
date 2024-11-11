import os
import sys
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..dynamics.base import Dynamics
from ..measurements.base import Measurement
from ..utils import assemble_grad_potential, langevin_sampler


def denoising_score_matching(
    device: torch.device,
    logger: SummaryWriter,
    score_fn: Callable[[Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    data: Tensor,
    batch_size: int,
    n_epoch: int,
    sigma: float,
    step: int,
):
    """Denoising score matching with fixed noise level.

    Args:
        device (torch.device): PyTorch working device.
        logger (SummaryWriter): Tensorboard logger.
        score_fn (Callable[[Tensor], Tensor]):
            Function handle that computes the score function at fixed noise level.
        optimizer (Optimizer): Optimizer.
        data (Tensor): Data samples.
        batch_size (int): Batch size of data loader.
        n_epoch (int): Number of training epochs.
        sigma (float): Noise level of the denoising score matching.
        step (int): Current time step.
    """
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with tqdm(range(n_epoch), mininterval=5.0, maxinterval=50.0, leave=False, desc="epoch", file=sys.stdout) as pbar:
        for i in pbar:
            epoch_loss = 0.0
            for batch_no, batch in enumerate(loader, start=1):
                (x0,) = batch
                x0 = x0.to(device)  # (B, *state_shape)
                z = torch.randn_like(x0, device=device)
                xt = x0 + sigma * z
                score = score_fn(xt)
                loss = nn.MSELoss()(score * sigma, -z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(
                    {
                        "batch_no": batch_no,
                        "epoch_avg_loss": epoch_loss / batch_no,
                    },
                    refresh=False,
                )
            logger.add_scalar(f"train/loss_step{step}", epoch_loss / batch_no, i + 1)


def trainer(
    workdir: str,
    device: torch.device,
    logger: SummaryWriter,
    dynamics: Dynamics,
    measurement: Measurement,
    model: nn.Module,
    prior: Tensor,
    states: Tensor,
    observations: Tensor,
    batch_size: int,
    n_epoch: int,
    lr: float,
    denoising_sigma: float,
    lmc_steps: int,
    lmc_stepsize: float,
    anneal_init: float,
    anneal_decay: float,
    anneal_steps: int,
    plot_callback: Callable,
) -> Tensor:
    """Trainer.

    Args:
        workdir (str): Path to store intermediate results.
        device (torch.device): PyTorch working device.
        logger (SummaryWriter): Tensorboard logger.
        dynamics (Dynamics): Physical model.
        measurement (Measurement): Measurement model.
        model (Module): Network for score matching.
        prior (Tensor): (n_train, *state_shape), samples of states derived from the prior distribution.
        states (Tensor): (steps, *state_shape), a ground-truth state trajectory.
        observations (Tensor): (steps, *measurement_shape), noisy observations of states.
        batch_size (int): Batch size of data loader.
        n_epoch (int): Number of epochs for each time step.
        lr (float): Learning rate.
        denoising_sigma (float): Noise level of the denoising score matching.
        lmc_steps (int): Number of the Langevin Monte Carlo.
        lmc_stepsize (float): Step size of the Langevin Monte Carlo.
        anneal_init (float): Initial temperature for anealing LMC.
        anneal_decay (float): Temperature decay rate for anealing LMC.
        anneal_steps (int): Number of Temperature decays for anealing LMC.
        plot_callback (Callable): Callback function to evaluate per time step in Tensorboard.

    Returns:
        Tensor: (steps, n_train, *state_shape), assimilated states given observations.
    """
    n_train, *shape = prior.shape
    steps = observations.shape[0]
    assimilated_states = torch.empty((steps, n_train, *shape), device=device)

    with tqdm(range(steps), maxinterval=50.0, desc="state step", file=sys.stdout) as pbar:
        for i in pbar:
            """Prepare data and network."""
            prior_mean, prior_std = prior.mean(dim=0), prior.std(dim=0)
            # normalize states for stable input to the network
            normalized_prior = (prior - prior_mean) / prior_std
            """model predicts the noise from the noised normalized state, as in DDPM, 
            normalized_score_fn predicts the score for the normalized state."""
            normalized_score_fn: Callable[[Tensor], Tensor] = lambda x: -model(x) / denoising_sigma

            """Denoising score matching."""
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            denoising_score_matching(
                device=device,
                logger=logger,
                score_fn=normalized_score_fn,
                optimizer=optimizer,
                data=normalized_prior,
                batch_size=batch_size,
                n_epoch=n_epoch,
                sigma=denoising_sigma,
                step=i,
            )
            model.eval()
            torch.save(model.state_dict(), os.path.join(workdir, "ckpt", f"model_step_{i+1}.pt"))

            """Posterior sampling and prior update.
            score_fn predicts the score for the original (unnormalized) states.
            Y = aX + b => s_X(x) = a s_Y(ax+b)."""
            score_fn: Callable[[Tensor], Tensor] = lambda x: normalized_score_fn((x - prior_mean) / prior_std) / prior_std
            # potential gradient = - score_likelihood - score_prior
            grad_potential_fn = assemble_grad_potential(
                y=observations[i],
                score_likelihood=measurement.score_likelihood,
                score_prior=score_fn,
            )

            """Debugging scales between likelihood and prior."""
            with torch.no_grad():
                score_likelihood = measurement.score_likelihood(prior, observations[i])
                score_likelihood = torch.mean(score_likelihood**2)
                score_prior = score_fn(prior)
                score_prior = torch.mean(score_prior**2)
                logger.add_scalars(
                    f"debug",
                    {
                        "likelihood": score_likelihood,
                        "prior": score_prior,
                    },
                    i,
                )

            with torch.no_grad():
                posterior: Tensor = langevin_sampler(
                    grad_potential_fn=grad_potential_fn,
                    x=prior,
                    steps=lmc_steps,
                    dt=lmc_stepsize,
                    anneal_init=anneal_init,
                    anneal_decay=anneal_decay,
                    anneal_steps=anneal_steps,
                )  # (n_train, *shape)
                assimilated_states[i] = posterior
                prior = dynamics.transition(posterior)

            """Postprocessing."""
            mean_estimation = torch.mean(posterior, dim=0)  # (*shape, )
            median_estimation = torch.median(posterior, dim=0)[0]  # (*shape, )
            mean_rmse = torch.sqrt(torch.mean((mean_estimation - states[i]) ** 2))
            median_rmse = torch.sqrt(torch.mean((median_estimation - states[i]) ** 2))
            pbar.set_postfix(
                {
                    "mean(RMSE)": mean_rmse.item(),
                    "median(RMSE)": median_rmse.item(),
                },
                refresh=False,
            )
            logger.add_scalars(
                f"eval",
                {
                    "mean(RMSE)": mean_rmse.item(),
                    "median(RMSE)": median_rmse.item(),
                },
                i,
            )
            logger.add_figure(
                "Trajectory",
                plot_callback(states, assimilated_states, i),
                i,
            )
    return assimilated_states
