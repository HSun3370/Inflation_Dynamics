```{raw:typst}
#set page(margin: auto)
```

# Constant-p Full BEGE Best Model Summary

Generated: `2026-06-18T10:11:11`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/constant.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_1_1.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 28 | 25 | -166.7909 | 357.5819 | 398.0296 |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability restrictions: `yes`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Standard errors: `OPG inverse fallback`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.4727 | 0.4727 | 0.4727 |
| $n_t$ | 0.4590 | 0.9268 | 7.3165 |
| $\sigma_t^2$ | 0.1990 | 0.2674 | 1.2023 |
| $s_t^2$ | -0.6797 | 0.0355 | 0.0878 |
| $k_t^2$ | 0.2795 | 0.3396 | 1.1602 |

Mean process:

$$
\pi_{t+1} = 0.1831 + 0.1660\,\pi_t + 0.0787\,\pi_{t-1} + 0.2999\,SPF_t + 0.3076\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.5281\,\omega_{p,t} - 0.3825\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.4727,\\
n_t &= 0.2113 + 0.4747\,n_{t-1} + \frac{0.7577}{2(0.3825)^2}\,(u_{t-1}^+)^2 + \frac{0.2362}{2(0.3825)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1831 | 0.0290 |
| $\rho_1$ | 0.1660 | 0.0376 |
| $\rho_2$ | 0.0787 | 0.0273 |
| $\phi_1$ | 0.2999 | 0.1631 |
| $\phi_2$ | 0.3076 | 0.1028 |
| $p_0$ | 0.4727 | 0.1424 |
| $n_0$ | 0.2113 | 0.0517 |
| $\rho_n$ | 0.4747 | 0.0291 |
| $\phi_n^+$ | 0.7577 | 0.0098 |
| $\phi_n^-$ | 0.2362 | 0.0348 |
| $\sigma_p$ | 0.5281 | 0.1365 |
| $\sigma_n$ | 0.3825 | 0.0364 |
