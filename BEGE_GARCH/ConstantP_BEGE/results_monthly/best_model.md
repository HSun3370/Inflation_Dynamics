```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Constant-p Full BEGE Best Model Summary (Monthly)

Generated: `2026-11-11T21:32:27`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results_monthly/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 4 | 33 | 104.2254 | -184.4508 | -129.9758 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.3795 | 2.3795 | 2.3795 |
| $n_t$ | 0.1825 | 0.8207 | 5.6290 |
| $\sigma_t^2$ | 0.0276 | 0.0399 | 0.1325 |
| $\kappa_{3,t}$ | -0.0252 | 0.0005 | 0.0039 |
| $\kappa_{4,t}$ | 0.0019 | 0.0033 | 0.0140 |

Mean process:

$$
\pi_{t+1} = 0.0404 + 0.4272\,\pi_t - 0.0456\,\pi_{t-1} + 0.2433\,SPF_t + 0.3361\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1007\,\omega_{p,t} - 0.1388\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 2.3795,\\
n_t &= 0.0110 + 0.6863\,n_{t-1} + \frac{0.4449}{2(0.1388)^2}\,(u_{t-1}^+)^2 + \frac{0.1802}{2(0.1388)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0404 | 0.0202 |
| $\rho_1$ | 0.4272 | 0.0292 |
| $\rho_2$ | -0.0456 | 0.0734 |
| $\phi_1$ | 0.2433 | 0.0404 |
| $\phi_2$ | 0.3361 | 0.1424 |
| $p_0$ | 2.3795 | 0.0197 |
| $n_0$ | 0.0110 | 0.0549 |
| $\rho_n$ | 0.6863 | 0.0251 |
| $\phi_n^+$ | 0.4449 | 0.0701 |
| $\phi_n^-$ | 0.1802 | 0.0126 |
| $\sigma_p$ | 0.1007 | 0.0049 |
| $\sigma_n$ | 0.1388 | 0.0249 |
