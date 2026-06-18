```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-06-18T10:10:04`
Total estimations: `8000`
Successful estimations: `8000`
Eligible estimations for best-model selection: `8000`

Selection screen: successful optimizer status, finite positive BEGE parameters, documented parameter bounds, EWMA implied-variance bounds, positive conditional variance, and mean-process stationarity.
This report shows only the single likelihood-best admissible estimate across mean processes. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/Constant_BEGE/results/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by log likelihood across mean processes.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 30 | 30 | -181.6884 | 381.3768 | 411.7126 |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Standard errors: `sandwich`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.6646 | 2.6646 | 2.6646 |
| $n_t$ | 0.2821 | 0.2821 | 0.2821 |
| $\sigma_t^2$ | 0.3952 | 0.3952 | 0.3952 |
| $s_t^2$ | -0.2286 | -0.2286 | -0.2286 |
| $k_t^2$ | 0.9308 | 0.9308 | 0.9308 |

Mean process:

$$
\pi_{t+1} = 0.0041 + 0.3232\,\pi_t + 0.1633\,\pi_{t-1} + 0.4107\,SPF_t + 0.1942\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.2712\,\omega_{p,t} - 0.8404\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.6646,\qquad \bar{n} = 0.2821,\\
\operatorname{Var}_t(u_t) &= (0.2712)^2\,2.6646 + (0.8404)^2\,0.2821.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0041 | 0.1115 |
| $\rho_1$ | 0.3232 | 0.1214 |
| $\rho_2$ | 0.1633 | 0.0829 |
| $\phi_1$ | 0.4107 | 0.3014 |
| $\phi_2$ | 0.1942 | 0.2957 |
| $\bar{p}$ | 2.6646 | 0.1839 |
| $\bar{n}$ | 0.2821 | 0.1597 |
| $\sigma_p$ | 0.2712 | 0.0218 |
| $\sigma_n$ | 0.8404 | 0.3557 |