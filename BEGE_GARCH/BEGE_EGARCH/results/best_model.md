```{raw:typst}
#set page(margin: auto)
```

# BEGE-EGARCH Best Model Summary

Generated: `2026-11-12T14:28:19`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(1,1) | 9 | 35 | -167.3416 | 360.6832 | 405.3783 | 230 | 1969Q2--2026Q3 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.2802 | 1.0267 | 5.5818 |
| $n_t$ | 0.3404 | 0.6140 | 2.1203 |
| $\sigma_t^2$ | 0.1644 | 0.2939 | 1.1183 |
| $s_t^2$ | -0.2308 | 0.0038 | 0.2225 |
| $k_t^2$ | 0.1708 | 0.2976 | 1.0491 |

Mean process:

$$
\pi_{t+1} = 0.1057 + 0.2237\,\pi_t + 0.7446\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.3703\,\omega_{p,t} - 0.4624\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
\ln p_t &= 0.0070 + 0.7997\,\ln p_{t-1} + 0.0203\,\left(|z_{t-1}| - \sqrt{2/\pi}\right) + 0.4531\,z_{t-1},\\
\ln n_t &= -0.1198 + 0.5281\,\ln n_{t-1} + 0.7112\,\left(|z_{t-1}| - \sqrt{2/\pi}\right) + 0.1336\,z_{t-1},\\
z_t &= u_t/\sigma_t,\qquad \sigma_t^2 = (0.3703)^2\,p_t + (0.4624)^2\,n_t
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1057 | 0.0271 |
| $\rho_1$ | 0.2237 | 0.0349 |
| $\phi_1$ | 0.7446 | 0.0431 |
| $\omega_p$ | 0.0070 | 0.0532 |
| $\omega_n$ | -0.1198 | 0.0883 |
| $\beta_p$ | 0.7997 | 0.0250 |
| $\beta_n$ | 0.5281 | 0.0371 |
| $\alpha_p$ | 0.0203 | 0.0811 |
| $\alpha_n$ | 0.7112 | 0.0670 |
| $\gamma_p$ | 0.4531 | 0.0683 |
| $\gamma_n$ | 0.1336 | 0.0553 |
| $\sigma_p$ | 0.3703 | 0.0682 |
| $\sigma_n$ | 0.4624 | 0.0641 |
