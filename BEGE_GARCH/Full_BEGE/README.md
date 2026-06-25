```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-06-18T10:11:22`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Full_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by stabilized log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 13 | 33 | -165.2805 | 360.5611 | 411.1206 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.9806 | 1.1858 | 2.3678 |
| $n_t$ | 0.0849 | 0.2556 | 2.8363 |
| $\sigma_t^2$ | 0.1834 | 0.2969 | 1.1546 |
| $s_t^2$ | -0.9826 | 0.0427 | 0.1478 |
| $k_t^2$ | 0.1925 | 0.3661 | 2.0843 |

Mean process:

$$
\pi_{t+1} = 0.1937 + 0.1415\,\pi_t + 0.1380\,\pi_{t-1} + 0.2283\,SPF_t + 0.3635\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.3802\,\omega_{p,t} - 0.5787\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.0347 + 0.9561\,p_{t-1} + \frac{0.0000}{2(0.3802)^2}\,(u_{t-1}^+)^2 + \frac{0.0334}{2(0.3802)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.0295 + 0.5611\,n_{t-1} + \frac{0.7784}{2(0.5787)^2}\,(u_{t-1}^+)^2 + \frac{0.0720}{2(0.5787)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1937 | 0.0783 |
| $\rho_1$ | 0.1415 | 0.0611 |
| $\rho_2$ | 0.1380 | 0.0618 |
| $\phi_1$ | 0.2283 | 0.2319 |
| $\phi_2$ | 0.3635 | 0.1626 |
| $p_0$ | 0.0347 | 0.0277 |
| $n_0$ | 0.0295 | 0.0258 |
| $\rho_p$ | 0.9561 | 0.0242 |
| $\rho_n$ | 0.5611 | 0.0879 |
| $\phi_p^+$ | 0.0000 | NA |
| $\phi_p^-$ | 0.0334 | 0.0159 |
| $\phi_n^+$ | 0.7784 | 0.1702 |
| $\phi_n^-$ | 0.0720 | 0.0947 |
| $\sigma_p$ | 0.3802 | 0.0621 |
| $\sigma_n$ | 0.5787 | 0.1108 |
