```{raw:typst}
#set page(margin: auto)
```

# Constant-n Full BEGE Best Model Summary

Generated: `2026-06-18T10:11:30`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 31 | 40 | -170.2704 | 364.5408 | 404.9885 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.6223 | 1.1422 | 5.6821 |
| $n_t$ | 0.4055 | 0.4055 | 0.4055 |
| $\sigma_t^2$ | 0.2157 | 0.2840 | 0.8798 |
| $s_t^2$ | -0.0950 | -0.0455 | 0.3862 |
| $k_t^2$ | 0.3302 | 0.3840 | 0.8532 |

Mean process:

$$
\pi_{t+1} = 0.1460 + 0.1849\,\pi_t + 0.0812\,\pi_{t-1} + 0.9787\,SPF_t + -0.3699\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.3623\,\omega_{p,t} - 0.5750\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.1443 + 0.7377\,p_{t-1} + \frac{0.4074}{2(0.3623)^2}\,(u_{t-1}^+)^2 + \frac{0.0184}{2(0.3623)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.4055
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1460 | 0.0336 |
| $\rho_1$ | 0.1849 | 0.0432 |
| $\rho_2$ | 0.0812 | 0.0439 |
| $\phi_1$ | 0.9787 | 0.1048 |
| $\phi_2$ | -0.3699 | 0.0891 |
| $p_0$ | 0.1443 | 0.0321 |
| $n_0$ | 0.4055 | 0.0525 |
| $\rho_p$ | 0.7377 | 0.0172 |
| $\phi_p^+$ | 0.4074 | 0.0285 |
| $\phi_p^-$ | 0.0184 | 0.0323 |
| $\sigma_p$ | 0.3623 | 0.0408 |
| $\sigma_n$ | 0.5750 | 0.0736 |
