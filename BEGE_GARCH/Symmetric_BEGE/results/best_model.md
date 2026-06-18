```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-06-18T10:10:15`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/constant.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_1_1.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 35 | 12 | -167.9012 | 359.8025 | 400.2501 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.6347 | 1.1647 | 5.6629 |
| $n_t$ | 0.3243 | 0.5949 | 2.8917 |
| $\sigma_t^2$ | 0.1638 | 0.3005 | 1.4610 |
| $s_t^2$ | -0.2097 | -0.0432 | -0.0235 |
| $k_t^2$ | 0.1876 | 0.3441 | 1.6727 |

Mean process:

$$
\pi_{t+1} = 0.2006 + 0.1978\,\pi_t + 0.2405\,\pi_{t-1} + 0.2257\,SPF_t + 0.1197\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.3592\,\omega_{p,t} - 0.5026\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.1762 + 0.6852\,p_{t-1} + \frac{0.4128}{2(0.3592)^2}\,(u_{t-1}^+)^2 + \frac{0.0809}{2(0.3592)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.0900 + 0.6852\,n_{t-1} + \frac{0.4128}{2(0.5026)^2}\,(u_{t-1}^+)^2 + \frac{0.0809}{2(0.5026)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.2006 | 0.0397 |
| $\rho_1$ | 0.1978 | 0.0273 |
| $\rho_2$ | 0.2405 | 0.0389 |
| $\phi_1$ | 0.2257 | 0.2154 |
| $\phi_2$ | 0.1197 | 0.1590 |
| $p_0$ | 0.1762 | 0.0688 |
| $n_0$ | 0.0900 | 0.0393 |
| $\rho$ | 0.6852 | 0.0587 |
| $\phi^+$ | 0.4128 | 0.0560 |
| $\phi^-$ | 0.0809 | 0.0871 |
| $\sigma_p$ | 0.3592 | 0.0670 |
| $\sigma_n$ | 0.5026 | 0.0621 |
