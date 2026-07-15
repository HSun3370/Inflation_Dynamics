```{raw:typst}
#set page(margin: auto)
```

# EGARCH BEGE Specification

This folder contains an extension of the Full BEGE model in which the shape
processes follow EGARCH-type recursions instead of the GJR-type recursions
used by the canonical five BEGE specifications. It was added as an explicit
extension of the documented BEGE menu; the canonical menu in
`BEGE_GARCH/README.md` is unchanged.

## Residual Dynamics

As in every BEGE specification, the inflation residual is

$$
u_t = \sigma_p \omega_{p,t} - \sigma_n \omega_{n,t},
\qquad
\omega_{p,t}\sim\tilde{\Gamma}(p_t,1),\quad
\omega_{n,t}\sim\tilde{\Gamma}(n_t,1),
$$

with centered gamma shocks, so the conditional variance is

$$
\sigma_t^2 = \sigma_p^2 p_t + \sigma_n^2 n_t .
$$

Define the standardized residual and the component volatilities

$$
z_t = \frac{u_t}{\sigma_t},
\qquad
\sigma_t^p = \sigma_p \sqrt{p_t},
\qquad
\sigma_t^n = \sigma_n \sqrt{n_t}.
$$

## EGARCH Shape Recursions

The EGARCH BEGE model applies an EGARCH(1,1)-type update to each component
volatility, using the project-wide fixed $\sqrt{2/\pi}$ centering convention
for the magnitude term:

$$
\begin{aligned}
\ln\left(\sigma_t^p\right)^2 &= \omega_p^\sigma
  + \beta_p \ln\left(\sigma_{t-1}^p\right)^2
  + \alpha_p\left(|z_{t-1}| - \sqrt{2/\pi}\right)
  + \gamma_p z_{t-1},\\
\ln\left(\sigma_t^n\right)^2 &= \omega_n^\sigma
  + \beta_n \ln\left(\sigma_{t-1}^n\right)^2
  + \alpha_n\left(|z_{t-1}| - \sqrt{2/\pi}\right)
  + \gamma_n z_{t-1}.
\end{aligned}
$$

Here $\alpha$ is the magnitude (size) loading and $\gamma$ is the sign
(asymmetry) loading, following standard EGARCH notation. (The original
proposal note wrote $\alpha_n$ for both loadings; the second loading is
denoted $\gamma_n$ here so the two effects are separately parameterized.)

Because $\ln(\sigma_t^p)^2 = \ln p_t + 2\ln\sigma_p$, this is equivalent to an
EGARCH recursion directly on the *log shapes*, which is the parameterization
used in the code:

$$
\begin{aligned}
\ln p_t &= \omega_p + \beta_p \ln p_{t-1}
  + \alpha_p\left(|z_{t-1}| - \sqrt{2/\pi}\right) + \gamma_p z_{t-1},\\
\ln n_t &= \omega_n + \beta_n \ln n_{t-1}
  + \alpha_n\left(|z_{t-1}| - \sqrt{2/\pi}\right) + \gamma_n z_{t-1},
\end{aligned}
$$

with the intercept mapping
$\omega_p = \omega_p^\sigma - 2(1-\beta_p)\ln\sigma_p$ (and analogously for
$n$). The two parameterizations describe the same model; $\omega_p$ and
$\sigma_p$ are separately identified because $\sigma_p$ also enters the BEGE
density directly as the gamma scale, exactly as in the GJR-type BEGE models.

Unlike the GJR-type recursions, the two shape processes are *coupled* through
$z_{t-1} = u_{t-1}/\sigma_{t-1}$, which depends on both lagged shapes; the
recursion is therefore computed jointly.

## Parameters, Bounds, and Stability

Parameter order in estimation output:

$$
[\text{mean params}],\ \omega_p,\ \omega_n,\ \beta_p,\ \beta_n,\ \alpha_p,\ \alpha_n,\ \gamma_p,\ \gamma_n,\ \sigma_p,\ \sigma_n .
$$

| Parameter group | Bound |
|---|---|
| Log-shape intercepts $\omega_p,\omega_n$ | $[-5,5]$ |
| Persistence $\beta_p,\beta_n$ | $[0,1)$ |
| Magnitude loadings $\alpha_p,\alpha_n$ | $[0,2]$ |
| Sign loadings $\gamma_p,\gamma_n$ | $[-2,2]$ |
| Scale parameters $\sigma_p,\sigma_n$ | $[10^{-5},2]$ |

The stability restriction for each log-shape process is the log-linear AR
condition

$$
\beta_p < 1, \qquad \beta_n < 1 ,
$$

enforced during optimization (with the same $10^{-6}$ margin used by the
GJR-type stability constraints) and re-checked at collection. The mean-process
menu, mean-parameter bounds, mean-stationarity checks, EWMA implied-variance
bounds, density evaluator, random-search protocol, and selection rules are
identical to the other BEGE specifications (see `BEGE_GARCH/README.md`).

## Initialization and Numerical Guards

- Pre-sample states use the parameter-implied unconditional log-shape
  backcast $\ln p_1 = \omega_p/(1-\beta_p)$ and
  $\ln n_1 = \omega_n/(1-\beta_n)$ (the shock terms have zero mean under the
  $\sqrt{2/\pi}$ centering convention).
- Shape paths are floored at $10^{-4}$, the same numerical floor used by
  `gjr_recursion`.
- Log shapes are capped at $50$ to avoid overflow; such explosive paths are
  subsequently rejected by the EWMA implied-variance bounds.

## Starting Values

Volatility starts are sampled inside the bounds with the stability
restriction satisfied: $\beta \sim U[0.2, 0.98]$, a target unconditional
shape $\bar{s} = e^{U[\ln 0.05, \ln 5]}$ mapped to
$\omega = (1-\beta)\ln\bar{s}$, $\alpha \sim U[0, 0.8]$,
$\gamma \sim U[-0.5, 0.5]$, and $\sigma_p, \sigma_n \sim U[10^{-5}, 2]$. Mean
starts use the same OLS-centered $\pm 2\,SE$ intervals as the other BEGE
models.

## Estimation Scripts

- `BEGE_EGARCH1.py` — seed-level random-search job (same CLI as the Full BEGE
  job script `Full_BEGE/BEGE_GJR1.py`).
- `collect_egarch_results.py` — merges `output/raw/draw_*.csv`, applies the
  admissibility checks, computes standard errors for the best model, and
  writes `results/best_model.md` plus this folder's `README.md`.
- `SimulationEGARCH.sh` — Slurm submission script for the canonical
  server-side protocol (50 seed jobs × 40 draws × 25 starts = 50,000 starts
  per mean process).

The estimator is `BEGE_FullEGARCH_MLE` in `BEGE_GARCH/BEGE_GARCH.py`, and the
collection layer treats this model as family `full_egarch` in
`BEGE_GARCH/bege_batch.py`.
