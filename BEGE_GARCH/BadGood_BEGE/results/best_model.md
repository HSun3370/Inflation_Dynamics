```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-06-18T10:10:22`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/constant.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_1_1.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,1) | 39 | 14 | -163.5594 | 351.1188 | 391.5664 |

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
| $p_t$ | 0.4097 | 0.5799 | 1.7285 |
| $n_t$ | 0.2985 | 0.7108 | 3.8717 |
| $\sigma_t^2$ | 0.1416 | 0.2554 | 1.0320 |
| $s_t^2$ | -0.2196 | 0.0384 | 0.0683 |
| $k_t^2$ | 0.1754 | 0.2991 | 1.1456 |

Mean process:

$$
\pi_{t+1} = 0.2119 + 0.2240\,\pi_t + 0.2742\,\pi_{t-1} + 0.2699\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.4778\,\omega_{p,t} - 0.4042\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.1575 + 0.5795\,p_{t-1} + \frac{0.2143}{2(0.4778)^2}\,u_{t-1}^2,\\
n_t &= 0.1900 + 0.2831\,n_{t-1} + \frac{0.6633}{2(0.4042)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.2119 | 0.0283 |
| $\rho_1$ | 0.2240 | 0.0281 |
| $\rho_2$ | 0.2742 | 0.0140 |
| $\phi_1$ | 0.2699 | 0.0447 |
| $p_0$ | 0.1575 | 0.0329 |
| $n_0$ | 0.1900 | 0.0469 |
| $\rho_p$ | 0.5795 | 0.0474 |
| $\rho_n$ | 0.2831 | 0.0315 |
| $\phi_p$ | 0.2143 | 0.0433 |
| $\phi_n$ | 0.6633 | 0.0058 |
| $\sigma_p$ | 0.4778 | 0.0509 |
| $\sigma_n$ | 0.4042 | 0.0328 |
