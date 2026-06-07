```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Synthetic Recovery Check

Generated: `2026-06-04T20:28:48`

This report is produced by `BEGE_GARCH/Full_BEGE/BEGE_Full_Anchor_ARX11.py`.

## Simulation Design

The synthetic sample uses the ARX(1,1) mean process

$$
\pi_t = c + \rho_1 \pi_{t-1} + \phi_1 SPF_t + u_t.
$$

The residual is generated from the Full BEGE recursion

$$
u_t = \sigma_p (G_{p,t} - p_t) - \sigma_n (G_{n,t} - n_t),
$$

where `G_{p,t}` is drawn from `Gamma(shape=p_t, scale=1)` and `G_{n,t}` is drawn independently from `Gamma(shape=n_t, scale=1)`. The centered gamma draws have conditional mean zero, so the residual definition remains `u_t = pi_t - hat(pi)_t`.

The shape states follow

$$
\begin{aligned}
p_t &= p_0 + \rho_p p_{t-1} + \frac{\phi_p^+}{2\sigma_p^2}(u_{t-1}^+)^2 + \frac{\phi_p^-}{2\sigma_p^2}(u_{t-1}^-)^2,\\
n_t &= n_0 + \rho_n n_{t-1} + \frac{\phi_n^+}{2\sigma_n^2}(u_{t-1}^+)^2 + \frac{\phi_n^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The simulation draws `20000` observations after a burn-in of `5000` observations. The random seed is `20261208`. The SPF process is an exogenous stationary AR(1) process with mean `0.80`, autoregressive coefficient `0.65`, and innovation standard deviation `0.12`.

## Estimation Setup

The estimator maximizes the same Full BEGE log likelihood used in the project code. It runs `3` `L-BFGS-B` starts centered at the true parameter vector with jitter scale `0.03`. Exact true-parameter start included: `False`. Stability constraints and the unconditional variance bound `sigma_p^2 p0 + sigma_n^2 n0 <= 0.75` are imposed by the objective feasibility screen.

EWMA implied-variance bounds are not enforced during this synthetic recovery run. The final estimate is still rechecked against those bounds.

The true parameters were chosen away from the optimizer bounds and with comfortable stability margins:

- True p-process stability margin: `0.730000`
- True n-process stability margin: `0.720000`
- True unconditional variance margin: `0.497000`

## Estimation Results

| parameter       |   true |   estimate |       error |   abs_error |   relative_abs_error |
|:----------------|-------:|-----------:|------------:|------------:|---------------------:|
| const           |   0.08 |  0.105681  |  0.0256806  |  0.0256806  |           0.321008   |
| Inflation_lag_1 |   0.25 |  0.243308  | -0.00669175 |  0.00669175 |           0.026767   |
| SPF             |   0.7  |  0.673966  | -0.0260342  |  0.0260342  |           0.0371918  |
| p0              |   1.5  |  1.47472   | -0.0252788  |  0.0252788  |           0.0168525  |
| n0              |   1.3  |  1.31133   |  0.0113284  |  0.0113284  |           0.00871419 |
| rho_p           |   0.1  |  0.0598501 | -0.0401499  |  0.0401499  |           0.401499   |
| rho_n           |   0.12 |  0.115811  | -0.00418927 |  0.00418927 |           0.0349106  |
| phi_p_plus      |   0.24 |  0.216916  | -0.0230843  |  0.0230843  |           0.0961847  |
| phi_p_minus     |   0.1  |  0.0666917 | -0.0333083  |  0.0333083  |           0.333083   |
| phi_n_plus      |   0.09 |  0.0997206 |  0.00972059 |  0.00972059 |           0.108007   |
| phi_n_minus     |   0.23 |  0.235207  |  0.00520701 |  0.00520701 |           0.0226392  |
| sigma_p         |   0.25 |  0.262627  |  0.0126271  |  0.0126271  |           0.0505082  |
| sigma_n         |   0.35 |  0.351479  |  0.00147862 |  0.00147862 |           0.00422464 |

## Likelihood and Diagnostics

- Optimizer success: `True`
- Optimizer message: `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`
- Log likelihood at estimate: `-16892.764692`
- Log likelihood at true parameters: `-16896.233591`
- AIC at estimate: `33811.529384`
- BIC at estimate: `33914.274722`
- Maximum absolute parameter error: `0.04014991`
- Maximum relative absolute parameter error: `0.40149907`
- Estimated p-process stability margin: `0.798346`
- Estimated n-process stability margin: `0.716725`
- Estimated unconditional variance margin: `0.486286`
- Estimated EWMA implied-variance-bound check: `True`
- Maximum estimated shape state: `26.509324`
- Maximum estimated implied variance: `2.806889`

The parameter comparison CSV is written to `BEGE_GARCH/Full_BEGE/results/synthetic_full_arx11_recovery.csv`.

A finite random sample does not make the MLE exactly equal to the data-generating parameters. This run is a recovery check: with a long sample and a well-conditioned interior parameter vector, the estimate should be close to the true vector and the sample likelihood at the estimate should be at least as high as the likelihood at the truth.
