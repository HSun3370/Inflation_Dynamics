```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Symmetric-p/full-n BEGE Best Model Summary (Monthly)

Generated: `2026-11-15T19:38:45`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows the single likelihood-best admissible estimate among Constant, ARX(1,1), ARX(2,1), ARX(2,2). The CSV files retain the complete research search. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results_monthly/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 2 | 26 | 106.0133 | -184.0265 | -120.4723 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.5023 | 2.9244 | 4.9928 |
| $n_t$ | 0.1274 | 0.5788 | 3.9214 |
| $\sigma_t^2$ | 0.0253 | 0.0408 | 0.1493 |
| $\kappa_{3,t}$ | -0.0284 | -0.0004 | 0.0030 |
| $\kappa_{4,t}$ | 0.0017 | 0.0039 | 0.0197 |

Mean process:

$$
\pi_{t+1} = 0.0362 + 0.4309\,\pi_t - 0.0521\,\pi_{t-1} + 0.2071\,SPF_t + 0.3805\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.0924\,\omega_{p,t} - 0.1652\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.2784 + 0.8808\,p_{t-1} + \frac{0.0295}{2(0.0924)^2}\,u_{t-1}^2,\\
n_t &= 0.0107 + 0.6551\,n_{t-1} + \frac{0.4801}{2(0.1652)^2}\,(u_{t-1}^+)^2 + \frac{0.2070}{2(0.1652)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0362 | 0.0213 |
| $\rho_1$ | 0.4309 | 0.1622 |
| $\rho_2$ | -0.0521 | 0.0565 |
| $\phi_1$ | 0.2071 | 0.1125 |
| $\phi_2$ | 0.3805 | 0.1695 |
| $p_0$ | 0.2784 | 0.0400 |
| $n_0$ | 0.0107 | 0.0208 |
| $\rho_p$ | 0.8808 | 0.0176 |
| $\rho_n$ | 0.6551 | 0.1657 |
| $\phi_p$ | 0.0295 | 0.0164 |
| $\phi_n^+$ | 0.4801 | 0.2389 |
| $\phi_n^-$ | 0.2070 | 0.1322 |
| $\sigma_p$ | 0.0924 | 0.0127 |
| $\sigma_n$ | 0.1652 | 0.0786 |
