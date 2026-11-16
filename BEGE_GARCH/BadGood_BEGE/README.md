```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-11-11T21:31:11`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 8 | 25 | -171.7231 | 369.4462 | 414.1413 | 230 | 1969Q2--2026Q3 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.9600 | 1.4959 | 2.5820 |
| $n_t$ | 0.0564 | 0.3205 | 2.9021 |
| $\sigma_t^2$ | 0.1435 | 0.2708 | 0.9136 |
| $\kappa_{3,t}$ | -0.5492 | 0.0423 | 0.1216 |
| $\kappa_{4,t}$ | 0.1142 | 0.2427 | 1.1674 |

Mean process:

$$
\pi_{t+1} = 0.1135 + 0.1391\,\pi_t + 0.1815\,\pi_{t-1} + 0.2989\,SPF_t + 0.3526\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.3420\,\omega_{p,t} - 0.4920\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.0327 + 0.9506\,p_{t-1} + \frac{0.0270}{2(0.3420)^2}\,u_{t-1}^2,\\
n_t &= 0.0154 + 0.3286\,n_{t-1} + \frac{0.6624}{2(0.4920)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1135 | 0.0528 |
| $\rho_1$ | 0.1391 | 0.0615 |
| $\rho_2$ | 0.1815 | 0.0639 |
| $\phi_1$ | 0.2989 | 0.2452 |
| $\phi_2$ | 0.3526 | 0.2179 |
| $p_0$ | 0.0327 | 0.0106 |
| $n_0$ | 0.0154 | 0.0381 |
| $\rho_p$ | 0.9506 | 0.0163 |
| $\rho_n$ | 0.3286 | 0.0345 |
| $\phi_p$ | 0.0270 | 0.0127 |
| $\phi_n$ | 0.6624 | 0.0354 |
| $\sigma_p$ | 0.3420 | 0.0422 |
| $\sigma_n$ | 0.4920 | 0.0887 |