```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Constant-n Full BEGE Best Model Summary (Monthly)

Generated: `2026-11-11T21:33:08`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results_monthly/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 10 | 4 | 94.7731 | -165.5461 | -111.0711 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 4.3017 | 6.9305 | 25.0450 |
| $n_t$ | 0.1175 | 0.1175 | 0.1175 |
| $\sigma_t^2$ | 0.0337 | 0.0450 | 0.1233 |
| $s_t^2$ | -0.0084 | -0.0069 | 0.0034 |
| $k_t^2$ | 0.0121 | 0.0124 | 0.0144 |

Mean process:

$$
\pi_{t+1} = 0.0237 + 0.4602\,\pi_t - 0.0576\,\pi_{t-1} + 0.3447\,SPF_t + 0.2442\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.0657\,\omega_{p,t} - 0.3585\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.9859 + 0.7366\,p_{t-1} + \frac{0.3465}{2(0.0657)^2}\,(u_{t-1}^+)^2 + \frac{0.0965}{2(0.0657)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.1175
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0237 | 0.0102 |
| $\rho_1$ | 0.4602 | 0.1637 |
| $\rho_2$ | -0.0576 | 0.1745 |
| $\phi_1$ | 0.3447 | 0.4592 |
| $\phi_2$ | 0.2442 | 0.0285 |
| $p_0$ | 0.9859 | 0.1459 |
| $n_0$ | 0.1175 | 0.0669 |
| $\rho_p$ | 0.7366 | 0.1883 |
| $\phi_p^+$ | 0.3465 | 0.5270 |
| $\phi_p^-$ | 0.0965 | 0.0168 |
| $\sigma_p$ | 0.0657 | 0.0225 |
| $\sigma_n$ | 0.3585 | 0.1754 |
