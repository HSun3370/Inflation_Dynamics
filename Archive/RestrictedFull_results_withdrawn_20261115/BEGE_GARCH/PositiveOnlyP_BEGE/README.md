```{raw:typst}
#set page(margin: auto)
```

# Positive-only-p/full-n BEGE Best Model Summary

Generated: `2026-11-15T06:07:09`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2399`

This report shows the single likelihood-best admissible estimate among Constant, ARX(1,1), ARX(2,1), ARX(2,2). The CSV files retain the complete research search. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 9 | 21 | -171.3853 | 370.7707 | 418.9038 | 230 | 1969Q2--2026Q3 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 3.8341 | 4.9796 | 13.6280 |
| $n_t$ | 0.0163 | 0.1899 | 2.3986 |
| $\sigma_t^2$ | 0.1247 | 0.2257 | 1.2265 |
| $\kappa_{3,t}$ | -0.7586 | -0.0181 | 0.0370 |
| $\kappa_{4,t}$ | 0.0313 | 0.1534 | 1.6750 |

Mean process:

$$
\pi_{t+1} = 0.1208 + 0.1746\,\pi_t + 0.1618\,\pi_{t-1} + 0.5564\,SPF_t + 0.0604\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1713\,\omega_{p,t} - 0.5776\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.8822 + 0.7619\,p_{t-1} + \frac{0.1999}{2(0.1713)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.0000 + 0.2784\,n_{t-1} + \frac{0.9684}{2(0.5776)^2}\,(u_{t-1}^+)^2 + \frac{0.4748}{2(0.5776)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1208 | 7683.8688 |
| $\rho_1$ | 0.1746 | 24464.1716 |
| $\rho_2$ | 0.1618 | 11878.8437 |
| $\phi_1$ | 0.5564 | 3262.4371 |
| $\phi_2$ | 0.0604 | 38612.4362 |
| $p_0$ | 0.8822 | 86378.7535 |
| $n_0$ | 0.0000 | 43.2085 |
| $\rho_p$ | 0.7619 | 12590.7202 |
| $\rho_n$ | 0.2784 | 691.0356 |
| $\phi_p^+$ | 0.1999 | 66406.6653 |
| $\phi_n^+$ | 0.9684 | 75011.8675 |
| $\phi_n^-$ | 0.4748 | 73520.6284 |
| $\sigma_p$ | 0.1713 | 6385.7192 |
| $\sigma_n$ | 0.5776 | 36040.2517 |