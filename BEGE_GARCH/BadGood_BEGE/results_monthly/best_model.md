```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# BadGood BEGE Best Model Summary (Monthly)

Generated: `2026-11-11T21:31:48`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BadGood_BEGE/results_monthly/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 4 | 30 | 103.7101 | -181.4203 | -122.4057 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.5340 | 3.0488 | 5.8112 |
| $n_t$ | 0.1306 | 0.5234 | 3.5438 |
| $\sigma_t^2$ | 0.0251 | 0.0411 | 0.1535 |
| $\kappa_{3,t}$ | -0.0289 | -0.0008 | 0.0026 |
| $\kappa_{4,t}$ | 0.0018 | 0.0040 | 0.0212 |

Mean process:

$$
\pi_{t+1} = 0.0287 + 0.4455\,\pi_t - 0.0371\,\pi_{t-1} + 0.3000\,SPF_t + 0.2699\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.0912\,\omega_{p,t} - 0.1729\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.3873 + 0.8350\,p_{t-1} + \frac{0.0512}{2(0.0912)^2}\,u_{t-1}^2,\\
n_t &= 0.0136 + 0.6177\,n_{t-1} + \frac{0.3806}{2(0.1729)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0287 | 0.0154 |
| $\rho_1$ | 0.4455 | 0.0405 |
| $\rho_2$ | -0.0371 | 0.0388 |
| $\phi_1$ | 0.3000 | 0.0946 |
| $\phi_2$ | 0.2699 | 0.0897 |
| $p_0$ | 0.3873 | 0.0159 |
| $n_0$ | 0.0136 | 0.0234 |
| $\rho_p$ | 0.8350 | 0.0276 |
| $\rho_n$ | 0.6177 | 0.0590 |
| $\phi_p$ | 0.0512 | 0.0258 |
| $\phi_n$ | 0.3806 | 0.0578 |
| $\sigma_p$ | 0.0912 | 0.0135 |
| $\sigma_n$ | 0.1729 | 0.0311 |
