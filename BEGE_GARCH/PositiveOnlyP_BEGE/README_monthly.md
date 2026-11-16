```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Positive-only-p/full-n BEGE Best Model Summary (Monthly)

Generated: `2026-11-15T20:58:11`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows the single likelihood-best admissible estimate among Constant, ARX(1,1), ARX(2,1), ARX(2,2). The CSV files retain the complete research search. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results_monthly/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results_monthly/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results_monthly/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results_monthly/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results_monthly/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/PositiveOnlyP_BEGE/results_monthly/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 3 | 17 | 108.0674 | -188.1348 | -124.5806 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.5377 | 3.0907 | 6.3849 |
| $n_t$ | 0.1109 | 0.4985 | 3.3058 |
| $\sigma_t^2$ | 0.0245 | 0.0410 | 0.1596 |
| $\kappa_{3,t}$ | -0.0281 | -0.0008 | 0.0026 |
| $\kappa_{4,t}$ | 0.0017 | 0.0041 | 0.0213 |

Mean process:

$$
\pi_{t+1} = 0.0346 + 0.4363\,\pi_t - 0.0416\,\pi_{t-1} + 0.2445\,SPF_t + 0.3275\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.0896\,\omega_{p,t} - 0.1759\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.3350 + 0.8622\,p_{t-1} + \frac{0.0879}{2(0.0896)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.0099 + 0.6273\,n_{t-1} + \frac{0.4729}{2(0.1759)^2}\,(u_{t-1}^+)^2 + \frac{0.2698}{2(0.1759)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0346 | 0.0150 |
| $\rho_1$ | 0.4363 | 0.0388 |
| $\rho_2$ | -0.0416 | 0.0384 |
| $\phi_1$ | 0.2445 | 0.0920 |
| $\phi_2$ | 0.3275 | 0.1106 |
| $p_0$ | 0.3350 | 0.3539 |
| $n_0$ | 0.0099 | 0.0197 |
| $\rho_p$ | 0.8622 | 0.1231 |
| $\rho_n$ | 0.6273 | 0.0601 |
| $\phi_p^+$ | 0.0879 | 0.1150 |
| $\phi_n^+$ | 0.4729 | 0.1139 |
| $\phi_n^-$ | 0.2698 | 0.1099 |
| $\sigma_p$ | 0.0896 | 0.0171 |
| $\sigma_n$ | 0.1759 | 0.0321 |