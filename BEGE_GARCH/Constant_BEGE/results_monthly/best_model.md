```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Constant BEGE Best Model Summary (Monthly)

Generated: `2026-11-11T21:30:34`
Total estimations: `2400`
Successful estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate across mean processes. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/by_mean/ARX_2_2.csv)
- [Constant BEGE shape summary](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results_monthly/constant_bege_initial_shapes_by_mean.csv)

## Selected Best Model

Best admissible estimate ranked by log likelihood across mean processes.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 4 | 24 | 60.9996 | -103.9991 | -63.1428 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 3.0494 | 3.0494 | 3.0494 |
| $n_t$ | 0.2669 | 0.2669 | 0.2669 |
| $\sigma_t^2$ | 0.0578 | 0.0578 | 0.0578 |
| $s_t^2$ | -0.0071 | -0.0071 | -0.0071 |
| $k_t^2$ | 0.0150 | 0.0150 | 0.0150 |

Mean process:

$$
\pi_{t+1} = 0.0376 + 0.4707\,\pi_t - 0.0266\,\pi_{t-1} + 0.2932\,SPF_t + 0.2327\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1055\,\omega_{p,t} - 0.2988\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 3.0494,\qquad \bar{n} = 0.2669,\\
\operatorname{Var}_t(u_t) &= (0.1055)^2\,3.0494 + (0.2988)^2\,0.2669.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0376 | 0.0148 |
| $\rho_1$ | 0.4707 | 0.0365 |
| $\rho_2$ | -0.0266 | 0.0453 |
| $\phi_1$ | 0.2932 | 0.3957 |
| $\phi_2$ | 0.2327 | 0.3835 |
| $\bar{p}$ | 3.0494 | 1.2318 |
| $\bar{n}$ | 0.2669 | 0.0748 |
| $\sigma_p$ | 0.1055 | 0.0243 |
| $\sigma_n$ | 0.2988 | 0.0561 |
