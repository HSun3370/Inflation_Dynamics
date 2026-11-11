```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-11-10T21:17:07`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 5 | 9 | -174.0997 | 372.1993 | 413.4563 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 3.7238 | 5.0290 | 22.1597 |
| $n_t$ | 0.0818 | 0.1655 | 1.2644 |
| $\sigma_t^2$ | 0.1580 | 0.2405 | 1.3234 |
| $s_t^2$ | -0.6256 | -0.0570 | -0.0147 |
| $k_t^2$ | 0.1415 | 0.2712 | 1.9747 |

Mean process:

$$
\pi_{t+1} = 0.1328 + 0.1869\,\pi_t + 0.1159\,\pi_{t-1} + 0.5715\,SPF_t + 0.0572\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1778\,\omega_{p,t} - 0.7019\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.9789 + 0.4501\,p_{t-1} + \frac{0.6014}{2(0.1778)^2}\,(u_{t-1}^+)^2 + \frac{0.1025}{2(0.1778)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.0406 + 0.4501\,n_{t-1} + \frac{0.6014}{2(0.7019)^2}\,(u_{t-1}^+)^2 + \frac{0.1025}{2(0.7019)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1328 | 0.0779 |
| $\rho_1$ | 0.1869 | 0.0762 |
| $\rho_2$ | 0.1159 | 0.0769 |
| $\phi_1$ | 0.5715 | 0.2638 |
| $\phi_2$ | 0.0572 | 0.2460 |
| $p_0$ | 1.9789 | 0.5479 |
| $n_0$ | 0.0406 | 0.0321 |
| $\rho$ | 0.4501 | 0.1629 |
| $\phi^+$ | 0.6014 | 0.2171 |
| $\phi^-$ | 0.1025 | 0.1402 |
| $\sigma_p$ | 0.1778 | 0.0460 |
| $\sigma_n$ | 0.7019 | 0.3852 |
