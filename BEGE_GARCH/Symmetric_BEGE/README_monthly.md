```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Symmetric BEGE Best Model Summary (Monthly)

Generated: `2026-11-11T21:36:07`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results_monthly/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results_monthly/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results_monthly/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results_monthly/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results_monthly/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Symmetric_BEGE/results_monthly/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 6 | 34 | 105.3633 | -186.7265 | -132.2515 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 3.0911 | 4.4877 | 13.9162 |
| $n_t$ | 0.1015 | 0.2856 | 1.5462 |
| $\sigma_t^2$ | 0.0240 | 0.0415 | 0.1592 |
| $\kappa_{3,t}$ | -0.0176 | -0.0014 | 0.0010 |
| $\kappa_{4,t}$ | 0.0021 | 0.0048 | 0.0236 |

Mean process:

$$
\pi_{t+1} = 0.0366 + 0.4415\,\pi_t - 0.0519\,\pi_{t-1} + 0.3642\,SPF_t + 0.2079\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.0790\,\omega_{p,t} - 0.2163\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.8387 + 0.6952\,p_{t-1} + \frac{0.2927}{2(0.0790)^2}\,(u_{t-1}^+)^2 + \frac{0.1302}{2(0.0790)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.0171 + 0.6952\,n_{t-1} + \frac{0.2927}{2(0.2163)^2}\,(u_{t-1}^+)^2 + \frac{0.1302}{2(0.2163)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0366 | 0.0158 |
| $\rho_1$ | 0.4415 | 0.0391 |
| $\rho_2$ | -0.0519 | 0.0387 |
| $\phi_1$ | 0.3642 | 0.1220 |
| $\phi_2$ | 0.2079 | 0.1223 |
| $p_0$ | 0.8387 | 0.2040 |
| $n_0$ | 0.0171 | 0.0133 |
| $\rho$ | 0.6952 | 0.0642 |
| $\phi^+$ | 0.2927 | 0.0702 |
| $\phi^-$ | 0.1302 | 0.0719 |
| $\sigma_p$ | 0.0790 | 0.0135 |
| $\sigma_n$ | 0.2163 | 0.0423 |