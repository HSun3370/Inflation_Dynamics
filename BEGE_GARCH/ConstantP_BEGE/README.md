```{raw:typst}
#set page(margin: auto)
```

# Constant-p Full BEGE Best Model Summary

Generated: `2026-11-11T21:32:37`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantP_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 1 | 29 | -172.9488 | 369.8975 | 411.1545 | 230 | 1969Q2--2026Q3 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.6949 | 2.6949 | 2.6949 |
| $n_t$ | 0.0594 | 0.3695 | 4.5392 |
| $\sigma_t^2$ | 0.1586 | 0.2172 | 1.0060 |
| $s_t^2$ | -0.6780 | 0.0081 | 0.0591 |
| $k_t^2$ | 0.0611 | 0.1277 | 1.0230 |

Mean process:

$$
\pi_{t+1} = 0.1379 + 0.1596\,\pi_t + 0.0970\,\pi_{t-1} + 0.6421\,SPF_t + 0.0452\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.2338\,\omega_{p,t} - 0.4349\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 2.6949,\\
n_t &= 0.0166 + 0.4089\,n_{t-1} + \frac{0.9859}{2(0.4349)^2}\,(u_{t-1}^+)^2 + \frac{0.1810}{2(0.4349)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1379 | 0.0632 |
| $\rho_1$ | 0.1596 | 0.0951 |
| $\rho_2$ | 0.0970 | 0.0866 |
| $\phi_1$ | 0.6421 | 0.3139 |
| $\phi_2$ | 0.0452 | 0.2588 |
| $p_0$ | 2.6949 | 0.4869 |
| $n_0$ | 0.0166 | 0.0707 |
| $\rho_n$ | 0.4089 | 0.1355 |
| $\phi_n^+$ | 0.9859 | 0.2241 |
| $\phi_n^-$ | 0.1810 | 0.1760 |
| $\sigma_p$ | 0.2338 | 0.0279 |
| $\sigma_n$ | 0.4349 | 0.1437 |