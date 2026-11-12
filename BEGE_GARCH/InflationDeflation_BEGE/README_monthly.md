```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Inflation/Deflation BEGE-GJR Best Model Summary (Monthly)

Generated: `2026-11-11T21:36:23`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results_monthly/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results_monthly/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results_monthly/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results_monthly/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results_monthly/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/InflationDeflation_BEGE/results_monthly/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 5 | 29 | 101.9370 | -177.8741 | -118.8594 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 3.6510 | 5.7035 | 20.8087 |
| $n_t$ | 0.1064 | 0.1481 | 0.7109 |
| $\sigma_t^2$ | 0.0275 | 0.0414 | 0.1406 |
| $s_t^2$ | -0.0177 | -0.0018 | 0.0077 |
| $k_t^2$ | 0.0040 | 0.0055 | 0.0235 |

Mean process:

$$
\pi_{t+1} = 0.0340 + 0.4722\,\pi_t - 0.0588\,\pi_{t-1} + 0.2677\,SPF_t + 0.2754\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.0703\,\omega_{p,t} - 0.2658\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.8235 + 0.7577\,p_{t-1} + \frac{0.3787}{2(0.0703)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.0512 + 0.5124\,n_{t-1} + \frac{0.3396}{2(0.2658)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0340 | 0.0532 |
| $\rho_1$ | 0.4722 | 2.4481 |
| $\rho_2$ | -0.0588 | 3.3825 |
| $\phi_1$ | 0.2677 | 25.2729 |
| $\phi_2$ | 0.2754 | 25.7884 |
| $p_0$ | 0.8235 | 1.6263 |
| $n_0$ | 0.0512 | 0.5907 |
| $\rho_p$ | 0.7577 | 0.4778 |
| $\rho_n$ | 0.5124 | 0.4753 |
| $\phi_p^+$ | 0.3787 | 0.4834 |
| $\phi_n^-$ | 0.3396 | 8.8117 |
| $\sigma_p$ | 0.0703 | 0.1676 |
| $\sigma_n$ | 0.2658 | 0.7681 |