```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Full BEGE Best Model Summary (Monthly)

Generated: `2026-11-11T21:34:20`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results_monthly/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results_monthly/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results_monthly/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results_monthly/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results_monthly/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results_monthly/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 4 | 14 | 108.0682 | -186.1364 | -118.0426 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.5474 | 3.1032 | 6.4142 |
| $n_t$ | 0.1101 | 0.4960 | 3.3106 |
| $\sigma_t^2$ | 0.0246 | 0.0411 | 0.1605 |
| $s_t^2$ | -0.0281 | -0.0008 | 0.0026 |
| $k_t^2$ | 0.0017 | 0.0041 | 0.0214 |

Mean process:

$$
\pi_{t+1} = 0.0348 + 0.4377\,\pi_t - 0.0428\,\pi_{t-1} + 0.2479\,SPF_t + 0.3231\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.0894\,\omega_{p,t} - 0.1762\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.3449 + 0.8587\,p_{t-1} + \frac{0.0908}{2(0.0894)^2}\,(u_{t-1}^+)^2 + \frac{0.0000}{2(0.0894)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.0103 + 0.6236\,n_{t-1} + \frac{0.4771}{2(0.1762)^2}\,(u_{t-1}^+)^2 + \frac{0.2729}{2(0.1762)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0348 | 0.0139 |
| $\rho_1$ | 0.4377 | 0.0394 |
| $\rho_2$ | -0.0428 | 0.0371 |
| $\phi_1$ | 0.2479 | 0.0459 |
| $\phi_2$ | 0.3231 | 0.0787 |
| $p_0$ | 0.3449 | 0.0259 |
| $n_0$ | 0.0103 | 0.0193 |
| $\rho_p$ | 0.8587 | 0.0144 |
| $\rho_n$ | 0.6236 | 0.0389 |
| $\phi_p^+$ | 0.0908 | 0.0175 |
| $\phi_p^-$ | 0.0000 | NA |
| $\phi_n^+$ | 0.4771 | 0.0544 |
| $\phi_n^-$ | 0.2729 | 0.0494 |
| $\sigma_p$ | 0.0894 | 0.0092 |
| $\sigma_n$ | 0.1762 | 0.0291 |