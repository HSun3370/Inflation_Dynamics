```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-11-10T18:23:00`
Total estimations: `2400`
Successful estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate across mean processes. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by log likelihood across mean processes.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 1 | 27 | -188.2489 | 394.4978 | 425.4405 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.9686 | 2.9686 | 2.9686 |
| $n_t$ | 0.2219 | 0.2219 | 0.2219 |
| $\sigma_t^2$ | 0.3860 | 0.3860 | 0.3860 |
| $s_t^2$ | -0.2714 | -0.2714 | -0.2714 |
| $k_t^2$ | 1.1095 | 1.1095 | 1.1095 |

Mean process:

$$
\pi_{t+1} = -0.0196 + 0.2762\,\pi_t + 0.1961\,\pi_{t-1} + 0.5339\,SPF_t + 0.1173\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.2532\,\omega_{p,t} - 0.9393\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.9686,\qquad \bar{n} = 0.2219,\\
\operatorname{Var}_t(u_t) &= (0.2532)^2\,2.9686 + (0.9393)^2\,0.2219.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | -0.0196 | 0.1149 |
| $\rho_1$ | 0.2762 | 0.0729 |
| $\rho_2$ | 0.1961 | 0.0873 |
| $\phi_1$ | 0.5339 | 0.2845 |
| $\phi_2$ | 0.1173 | 0.3180 |
| $\bar{p}$ | 2.9686 | 0.9744 |
| $\bar{n}$ | 0.2219 | 0.1434 |
| $\sigma_p$ | 0.2532 | 0.0580 |
| $\sigma_n$ | 0.9393 | 0.4453 |