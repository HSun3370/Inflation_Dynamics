```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-06-18T10:11:34`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/constant.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_1_1.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 3 | 39 | -167.9018 | 361.8035 | 405.6218 |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability restrictions: `yes`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Standard errors: `sandwich`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 4.5793 | 7.2032 | 28.6664 |
| $n_t$ | 0.0584 | 0.0697 | 0.6744 |
| $\sigma_t^2$ | 0.1699 | 0.2766 | 1.0806 |
| $s_t^2$ | -1.2028 | -0.0927 | 0.0660 |
| $k_t^2$ | 0.3575 | 0.4378 | 3.8913 |

Mean process:

$$
\pi_{t+1} = 0.1246 + 0.2617\,\pi_t + 0.1743\,\pi_{t-1} + 0.2915\,SPF_t + 0.1749\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1486\,\omega_{p,t} - 0.9893\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.2658 + 0.7076\,p_{t-1} + \frac{0.4250}{2(0.1486)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.0508 + 0.1317\,n_{t-1} + \frac{1.2626}{2(0.9893)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1246 | 0.0726 |
| $\rho_1$ | 0.2617 | 0.0767 |
| $\rho_2$ | 0.1743 | 0.0758 |
| $\phi_1$ | 0.2915 | 0.3736 |
| $\phi_2$ | 0.1749 | 0.3151 |
| $p_0$ | 1.2658 | 0.7514 |
| $n_0$ | 0.0508 | 0.0486 |
| $\rho_p$ | 0.7076 | 0.1141 |
| $\rho_n$ | 0.1317 | 0.0583 |
| $\phi_p^+$ | 0.4250 | 0.1637 |
| $\phi_n^-$ | 1.2626 | 0.8901 |
| $\sigma_p$ | 0.1486 | 0.0423 |
| $\sigma_n$ | 0.9893 | 0.5903 |