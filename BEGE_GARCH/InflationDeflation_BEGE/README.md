```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-11-11T21:35:04`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 2 | 40 | -173.8289 | 373.6579 | 418.3529 | 230 | 1969Q2--2026Q3 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 4.5238 | 6.8190 | 28.3296 |
| $n_t$ | 0.0536 | 0.0669 | 0.6198 |
| $\sigma_t^2$ | 0.1665 | 0.2536 | 0.9870 |
| $\kappa_{3,t}$ | -1.0440 | -0.0789 | 0.0900 |
| $\kappa_{4,t}$ | 0.3157 | 0.4089 | 3.4248 |

Mean process:

$$
\pi_{t+1} = 0.1430 + 0.2318\,\pi_t + 0.1700\,\pi_{t-1} + 0.4332\,SPF_t + 0.0564\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1524\,\omega_{p,t} - 0.9760\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.6246 + 0.6314\,p_{t-1} + \frac{0.5123}{2(0.1524)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.0458 + 0.1461\,n_{t-1} + \frac{1.1652}{2(0.9760)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1430 | 0.0795 |
| $\rho_1$ | 0.2318 | 0.0743 |
| $\rho_2$ | 0.1700 | 0.0894 |
| $\phi_1$ | 0.4332 | 0.3948 |
| $\phi_2$ | 0.0564 | 0.3364 |
| $p_0$ | 1.6246 | 0.6573 |
| $n_0$ | 0.0458 | 0.0535 |
| $\rho_p$ | 0.6314 | 0.1894 |
| $\rho_n$ | 0.1461 | 0.0630 |
| $\phi_p^+$ | 0.5123 | 0.2537 |
| $\phi_n^-$ | 1.1652 | 1.0621 |
| $\sigma_p$ | 0.1524 | 0.0338 |
| $\sigma_n$ | 0.9760 | 0.7287 |