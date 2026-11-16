```{raw:typst}
#set page(margin: auto)
```

# BEGE-EGARCH Specification
  
The inflation residual is

$$
u_t = \sigma_p \omega_{p,t} - \sigma_n \omega_{n,t},
\qquad
\omega_{p,t}\sim\tilde{\Gamma}(p_t,1),\quad
\omega_{n,t}\sim\tilde{\Gamma}(n_t,1),
$$

with conditionally independent centered gamma shocks, so the conditional variance is

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

The BEGE-EGARCH model applies an EGARCH(1,1)-type update to each component
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

Because $\ln(\sigma_t^p)^2 = \ln p_t + 2\ln\sigma_p$, this is equivalent to an
EGARCH recursion directly on the *log shapes*:

$$
\begin{aligned}
\ln p_t &= \omega_p + \beta_p \ln p_{t-1}
  + \alpha_p\left(|z_{t-1}| - \sqrt{2/\pi}\right) + \gamma_p z_{t-1},\\
\ln n_t &= \omega_n + \beta_n \ln n_{t-1}
  + \alpha_n\left(|z_{t-1}| - \sqrt{2/\pi}\right) + \gamma_n z_{t-1},
\end{aligned}
$$

with the intercept mapping
$\omega_j = \omega_j^\sigma - 2(1-\beta_j)\ln\sigma_j$, for $j\in\{p,n\}$. The two parameterizations describe the same model; $\omega_p$ and
$\sigma_p$ are separately identified because $\sigma_p$ also enters the BEGE
density directly as the gamma scale, exactly as in the GJR-type BEGE models.

The two shape processes are *coupled* through
$z_{t-1} = u_{t-1}/\sigma_{t-1}$, which depends on both lagged shapes; the
recursion is therefore computed jointly.

## Parameters and Stability

Parameter order in estimation output:

$$
[\text{mean params}],\ \omega_p,\ \omega_n,\ \beta_p,\ \beta_n,\ \alpha_p,\ \alpha_n,\ \gamma_p,\ \gamma_n,\ \sigma_p,\ \sigma_n .
$$

The persistence restriction $\beta_p<1$, $\beta_n<1$ is enforced during
optimization, with the same margin $\delta=10^{-6}$ as the GJR-type stability
constraints, and re-checked at collection. The parameter bounds, random-start
draws, initial shape states, and numerical guards (shape floor and log-shape
cap) are listed with those of the other BEGE specifications in the parameter
table of [Estimation and Random Starts](../README.md#estimation-and-random-starts).
The mean-process menu, mean-parameter bounds, mean-stationarity checks, density
evaluator, random-search protocol, and selection rules are identical to the
other BEGE specifications (see `BEGE_GARCH/README.md`).

## Estimation Scripts

- `BEGE_EGARCH1.py` — seed-level random-search job (same CLI as the Full BEGE
  job script `Full_BEGE/BEGE_GJR1.py`).
- `collect_egarch_results.py` — merges the raw `draw_*.csv` files (written to
  `/scratch/<UNI>/Inflation_Dynamics/BEGE_GARCH/BEGE_EGARCH/output/raw/` on the
  CBS Research Grid), applies the admissibility checks, computes standard
  errors for the best model, and writes `results/best_model.md` plus this
  folder's `README.md`.
- `SimulationEGARCH.sh` — CBS Research Grid (`grid_run`/SGE) submission script
  for the canonical server-side protocol (10 seed jobs × 40 draws × 25 starts =
  10,000 starts per mean process).

The estimator is `BEGE_FullEGARCH_MLE` in `BEGE_GARCH/BEGE_GARCH.py`, and the
collection layer treats this model as family `full_egarch` in
`BEGE_GARCH/bege_batch.py`.
