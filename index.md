# Nonlinear Assimilation with <span style="color: red;">S</span>core-based <span style="color: red;">S</span>equential <span style="color: red;">L</span>angevin <span style="color: red;">S</span>ampling

Zhao Ding<sup>1</sup>, Chenguang Duan<sup>1</sup>, Yuling Jiao<sup>1</sup>, Jerry Zhijian Yang<sup>1</sup>, Cheng Yuan<sup>2</sup> and Pingwen Zhang<sup>1,3</sup>

Arranged in alphabetical order.

<sup>1</sup> Wuhan University, <sup>2</sup> Central China Normal University, <sup>3</sup> Peking University

[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/burning489/SSLS)
[![arXiv](https://img.shields.io/badge/arXiv-2411.13443-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2411.13443)

## Brief
We propose nonlinear assimilation method called score-based sequential Langevin sampling (SSLS) within a Bayesian recursive framework.

## Problem Setup
Consider the following system:
$$
  \begin{align*}
    \mathbf{X}^{k} & =f_{k-1}(\mathbf{X}^{k-1}, \eta^{k-1}), \quad k > 1, \\
    \mathbf{Y}^{k} & =g_{k}(\mathbf{X}^{k}, \xi^{k}), \quad k \geq 1,
  \end{align*}
$$
where $\mathbf{X}_k$ is the latent states of interests evolved by $f_k$, and $\mathbf{Y}_k$ is the observations under measurement $g_k$. Here we assume that $\eta^k$ and $\xi^k$ are noises with known distributions. 

> __The goal of Data Assimilation__:
Combine historical observations with dynamics simulation to provide the best estimate of the current states.


## Recursive Bayesian Framework
Our work is carried out under the recursive Bayesian framework described below:
$$
\begin{align*}
  & {\color{blue} {p(\mathbf{x}^k | \mathbf{y}^{[k]})}} \\
  \propto~ & p(\mathbf{y}^k | \mathbf{x}^k, \mathbf{y}^{[k-1]}) p(\mathbf{x}^k, \mathbf{y}^{[k-1]}) \\
  \propto~ & \underbrace{p(\mathbf{y}^k | \mathbf{x}^k)}_{\text{likelihood}} \underbrace{\int \overbrace{p(\mathbf{x}^k | \mathbf{x}^{k-1})}^{\text{transition}} {\color{blue} \overbrace{ {p(\mathbf{x}^{k-1} | \mathbf{y}^{[k-1]})}}^{\text{last posterior}}} \, \mathrm{d} \mathbf{x}^{k-1}}_{\text{prior}} \\
  \propto~ & \underbrace{p(\mathbf{y}^k | \mathbf{x}^k)}_{\text{likelihood}} \underbrace{p(\mathbf{x}^{k} | \mathbf{y}^{[k-1]})}_{\text{prior}}
\end{align*}
$$
We maintain an ensemble of particles to estimate the prior and posterior distribution throughout the assimilation process. 
At each step, the prior samples are obtained by running the dynamics simulation starting from the posterior particles of the last time point.

## Langevin Monte Carlo
The posterior score now can be decomposed as the sum of likelihood score and prior score:
$$ \underbrace{\nabla \log p (\mathbf{x}^k|\mathbf{y}^{[k]})}_\text{score of posterior} = \nabla \log \underbrace{p(\mathbf{y}^k|\mathbf{x}^k)}_\text{likelihood} + \underbrace{\nabla \log p(\mathbf{x}^k|\mathbf{y}^{[k-1]})}_\text{score of prior}. $$
The likelihood score can be computed with known measurement model and noises.
As for the prior score, we exploit the score matching technique at each time step based on the prior ensemble.

After assembling the posterior score, we can use any Langevin-type sampling method to derive samples from the posterior distribution, starting from the transitioned ensemble from last time step:
$$
\mathrm{d} \mathbf{X}_t^k = \nabla \log p(\mathbf{X}_t^k | \mathbf{y}^{[k]}) \, \mathrm{d}t + \sqrt{2} \, \mathrm{d} \mathbf{B}_t, \ \mathbf{X}_0^k \sim p(\mathbf{x}^k|\mathbf{y}^{[k-1]}), \ t \in [0, \infty).
$$


## Flow Chart
We provide a flow chart below.

<img src="./asset/bayesian.png" width="85%"/>


## Pseudocode
We provide the python-like pseudocode below.
```python
# start from an initial prior
prior = sample_from_prior()
for i in range(k+1):
    # sliced / implicit / denoising
    prior_score = score_matching(prior)
    # assemble posterior
    posterior_score = lambda x: grad_log_likelihood(x, y[i]) + prior_score(x)
    # any Langevin-type sampling method
    posterior = langevin(prior, posterior_score)
    # dynamics transition to get best guess for next step
    prior = dynamics_transition(posterior)
```

## Numerical Results

Numerical examples demonstrate its outstanding performance in high-dimensional and nonlinear scenarios, as well as in situations with sparse or partial measurements.
Please refer to our paper for more results.

![Kolmogorov Flow AvgPool](asset/AvgPool.gif)
![Kolmogorov Flow GridMask](asset/GridMask.gif)
![Kolmogorov Flow CenterMask](asset/CenterMask.gif)


## How to Cite
If you find our work useful for your research, please consider citing

```bib
@misc{ding2024nonlinearassimilationscorebasedsequential,
      title={Nonlinear Assimilation with Score-based Sequential Langevin Sampling}, 
      author={Zhao Ding and Chenguang Duan and Yuling Jiao and Jerry Zhijian Yang and Cheng Yuan and Pingwen Zhang},
      year={2024},
      eprint={2411.13443},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2411.13443}, 
}
```