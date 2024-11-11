import datetime
import json
import os

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.dynamics import KolmogorovFlow, Lorenz96
from src.measurements import AveragePooling, CenterMask, GridMask, Linear, RandomMask
from src.networks import UNet, UNet1D
from src.train.train import trainer
from src.utils import plot_kolmogorov_vorticity, plot_lorenz_trajectory


@click.group()
def main():
    pass


@main.command()
@click.option("-c", "--config", type=str, default="configs/lorenz96.json", help="path to json configs")
@click.option("--description", type=str, default="lorenz96", help="prefix of work directory")
@click.option("-s", "--seed", type=int, default=42, help="global random seed")
@click.option("-d", "--device", type=str, default="cuda:0", help="PyTorch device")
def lorenz96(config, description, seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    with open(config, "r") as f:
        cfg = json.load(f)

    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    workdir = os.path.join("lorenz_results", f"{description}-{timestamp}")
    os.makedirs(os.path.join(workdir, "ckpt"), exist_ok=False)
    print(f"Results will be saved in {workdir}...")
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(cfg, f)

    dynamics = Lorenz96(
        dim=cfg["dynamics"]["dim"],
        prior_mean=cfg["dynamics"]["prior_mean"],
        prior_std=cfg["dynamics"]["prior_std"],
        dt=cfg["dynamics"]["dt"],
        forcing=cfg["dynamics"]["forcing"],
        perturb_std=cfg["dynamics"]["perturb_std"],
        solver=cfg["dynamics"]["solver"],
    )

    measurement = Linear(noise_std=cfg["measurement"]["noise_std"])

    """Generate a chain of ground-truth and corresponding measurements, 
    by perburbing from the stable state."""
    x0 = cfg["dynamics"]["forcing"] * torch.ones((1, cfg["dynamics"]["dim"]), device=device)  # (1, dim)
    x0[0][0] += 0.01
    states: torch.Tensor = dynamics.generate(
        x0=x0,
        steps=cfg["dynamics"]["steps"],
    )  # (steps+1, dim)
    observations: torch.Tensor = measurement.measure(states)

    """Shifted prior guess."""
    prior: torch.Tensor = dynamics.prior(n_sample=cfg["train"]["n_train"]).to(device)  # (n_train, dim)

    model: nn.Module = UNet1D(
        in_channels=cfg["network"]["in_channels"],
        out_channels=cfg["network"]["out_channels"],
        channels=cfg["network"]["channels"],
    ).to(device)

    logger = SummaryWriter(log_dir=workdir, flush_secs=60)

    assimilated_states: torch.Tensor = trainer(
        workdir=workdir,
        device=device,
        logger=logger,
        dynamics=dynamics,
        measurement=measurement,
        model=model,
        prior=prior,
        states=states,
        observations=observations,
        batch_size=cfg["train"]["batch_size"],
        n_epoch=cfg["train"]["n_epoch"],
        lr=cfg["train"]["lr"],
        denoising_sigma=cfg["train"]["denoising_sigma"],
        lmc_steps=cfg["langevin"]["steps"],
        lmc_stepsize=cfg["langevin"]["stepsize"],
        anneal_init=cfg["langevin"]["anneal_init"],
        anneal_decay=cfg["langevin"]["anneal_decay"],
        anneal_steps=cfg["langevin"]["anneal_steps"],
        plot_callback=plot_lorenz_trajectory,
    )  # (steps+1, n_train, dim)

    np.savez(
        os.path.join(workdir, "results.npz"),
        states=states.cpu().numpy(),  # (steps, dim)
        observations=observations.cpu().numpy(),  # (steps, dim)
        assimilated_states=assimilated_states.cpu().numpy(),  # (steps, n_train, dim)
    )

    logger.close()


@main.command()
@click.option("-c", "--config", type=str, default="configs/kolmogorov.json", help="path to json configs")
@click.option("--description", type=str, default="kolmogorov", help="prefix of work directory")
@click.option("-s", "--seed", type=int, default=42, help="global random seed")
@click.option("-d", "--device", type=str, default="cuda:0", help="PyTorch device")
def kolmogorov(config, description, seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    with open(config, "r") as f:
        cfg = json.load(f)

    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    workdir = os.path.join("kolmogorov_results", f"{description}-{timestamp}")
    os.makedirs(os.path.join(workdir, "ckpt"), exist_ok=False)
    print(f"Results will be saved in {workdir}...")
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(cfg, f)

    dynamics = KolmogorovFlow(
        grid_size=cfg["dynamics"]["grid_size"],
        reynolds=cfg["dynamics"]["reynolds"],
        dt=cfg["dynamics"]["dt"],
        seed=seed,
    )

    if cfg["measurement"]["type"] == "AvgPool":
        measurement = AveragePooling(
            noise_std=cfg["measurement"]["noise_std"],
            kernel_size=cfg["measurement"]["kernel_size"],
        )
    elif cfg["measurement"]["type"] == "GridMask":
        measurement = GridMask(
            noise_std=cfg["measurement"]["noise_std"],
            stride=cfg["measurement"]["stride"],
        )
    elif cfg["measurement"]["type"] == "CenterMask":
        measurement = CenterMask(
            noise_std=cfg["measurement"]["noise_std"],
        )
    elif cfg["measurement"]["type"] == "RandomMask":
        measurement = RandomMask(
            noise_std=cfg["measurement"]["noise_std"],
            sparsity=cfg["measurement"]["sparsity"],
        )
    elif cfg["measurement"]["type"] == "Linear":
        measurement = Linear(noise_std=cfg["measurement"]["noise_std"])

    """Generate a chain of ground-truth and corresponding measurements,
    by a 50 step shift from the starting time."""
    x0: torch.Tensor = dynamics.prior(n_sample=1).to(device)  # (1, 2, grid_size, grid_size)
    states: torch.Tensor = dynamics.generate(
        x0=x0,
        steps=cfg["dynamics"]["steps"] + 50,
    )[
        50:, ...
    ]  # (steps+1, 2, grid_size, grid_size)
    observations: torch.Tensor = measurement.measure(states)

    """Prior guess."""
    prior: torch.Tensor = dynamics.prior(n_sample=cfg["train"]["n_train"]).to(device)  # (n_train, 2, grid_size, grid_size)

    model: nn.Module = UNet(
        in_channels=2,
        out_channels=2,
        model_channels=cfg["network"]["model_channels"],
        channel_mult=cfg["network"]["channel_mult"],
        num_res_blocks=cfg["network"]["num_res_blocks"],
        attn_resolutions=cfg["network"]["attn_resolutions"],
        resolution=cfg["dynamics"]["grid_size"],
        dropout=cfg["network"]["dropout"],
        resample_with_conv=cfg["network"]["resample_with_conv"],
    ).to(device)

    logger = SummaryWriter(log_dir=workdir, flush_secs=60)

    assimilated_states: torch.Tensor = trainer(
        workdir=workdir,
        device=device,
        logger=logger,
        dynamics=dynamics,
        measurement=measurement,
        model=model,
        prior=prior,
        states=states,
        observations=observations,
        batch_size=cfg["train"]["batch_size"],
        n_epoch=cfg["train"]["n_epoch"],
        lr=cfg["train"]["lr"],
        denoising_sigma=cfg["train"]["denoising_sigma"],
        lmc_steps=cfg["langevin"]["steps"],
        lmc_stepsize=cfg["langevin"]["stepsize"],
        anneal_init=cfg["langevin"]["anneal_init"],
        anneal_decay=cfg["langevin"]["anneal_decay"],
        anneal_steps=cfg["langevin"]["anneal_steps"],
        plot_callback=plot_kolmogorov_vorticity,
    )  # (steps+1, n_train, dim)

    np.savez(
        os.path.join(workdir, "results.npz"),
        states=states.cpu().numpy(),  # (steps, 2, grid_size, grid_size)
        observations=observations.cpu().numpy(),
        assimilated_states=assimilated_states.cpu().numpy(),  # (steps, n_train, 2, grid_size, grid_size)
    )

    logger.close()


if __name__ == "__main__":
    main()
