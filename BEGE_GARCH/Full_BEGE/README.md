```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-11-10T23:01:03`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 10 | 27 | -168.9699 | 367.9398 | 419.5110 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.9649 | 1.2508 | 2.4952 |
| $n_t$ | 0.0441 | 0.3165 | 4.5645 |
| $\sigma_t^2$ | 0.1492 | 0.2606 | 1.2520 |
| $s_t^2$ | -0.7194 | 0.0707 | 0.1649 |
| $k_t^2$ | 0.1248 | 0.2401 | 1.4071 |

Mean process:

$$
\pi_{t+1} = 0.1986 + 0.0611\,\pi_t + 0.1222\,\pi_{t-1} + 0.3420\,SPF_t + 0.3609\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.3644\,\omega_{p,t} - 0.4527\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.0417 + 0.9436\,p_{t-1} + \frac{0.0081}{2(0.3644)^2}\,(u_{t-1}^+)^2 + \frac{0.0402}{2(0.3644)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.0101 + 0.3603\,n_{t-1} + \frac{1.0023}{2(0.4527)^2}\,(u_{t-1}^+)^2 + \frac{0.2684}{2(0.4527)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1986 | 0.0055 |
| $\rho_1$ | 0.0611 | 0.0245 |
| $\rho_2$ | 0.1222 | 0.0784 |
| $\phi_1$ | 0.3420 | 0.0474 |
| $\phi_2$ | 0.3609 | 0.1467 |
| $p_0$ | 0.0417 | 0.0135 |
| $n_0$ | 0.0101 | 0.0092 |
| $\rho_p$ | 0.9436 | 0.0122 |
| $\rho_n$ | 0.3603 | 0.0044 |
| $\phi_p^+$ | 0.0081 | 0.0490 |
| $\phi_p^-$ | 0.0402 | 0.0143 |
| $\phi_n^+$ | 1.0023 | 0.0014 |
| $\phi_n^-$ | 0.2684 | 0.0018 |
| $\sigma_p$ | 0.3644 | 0.0506 |
| $\sigma_n$ | 0.4527 | 0.0066 |