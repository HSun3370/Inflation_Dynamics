```{raw:typst}
#set page(margin: auto)
```

# Symmetric-p/full-n BEGE Best Model Summary

Generated: `2026-11-15T05:52:01`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows the single likelihood-best admissible estimate among Constant, ARX(1,1), ARX(2,1), ARX(2,2). The CSV files retain the complete research search. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/SymmetricP_BEGE/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 3 | 18 | -170.3622 | 368.7244 | 416.8575 | 230 | 1969Q2--2026Q3 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 1.0016 | 1.5513 | 2.5973 |
| $n_t$ | 0.0349 | 0.3847 | 5.3911 |
| $\sigma_t^2$ | 0.1285 | 0.2526 | 1.1310 |
| $\kappa_{3,t}$ | -0.6352 | 0.0522 | 0.1251 |
| $\kappa_{4,t}$ | 0.0886 | 0.1890 | 1.1149 |

Mean process:

$$
\pi_{t+1} = 0.1415 + 0.0985\,\pi_t + 0.1494\,\pi_{t-1} + 0.3216\,SPF_t + 0.3783\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.3260\,\omega_{p,t} - 0.4190\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.0292 + 0.9554\,p_{t-1} + \frac{0.0238}{2(0.3260)^2}\,u_{t-1}^2,\\
n_t &= 0.0000 + 0.3470\,n_{t-1} + \frac{0.9404}{2(0.4190)^2}\,(u_{t-1}^+)^2 + \frac{0.3656}{2(0.4190)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1415 | 0.0539 |
| $\rho_1$ | 0.0985 | 0.0649 |
| $\rho_2$ | 0.1494 | 0.0632 |
| $\phi_1$ | 0.3216 | 0.2654 |
| $\phi_2$ | 0.3783 | 0.2252 |
| $p_0$ | 0.0292 | 0.0073 |
| $n_0$ | 0.0000 | 0.0002 |
| $\rho_p$ | 0.9554 | 0.0062 |
| $\rho_n$ | 0.3470 | 0.0003 |
| $\phi_p$ | 0.0238 | 0.0027 |
| $\phi_n^+$ | 0.9404 | 0.0005 |
| $\phi_n^-$ | 0.3656 | 0.0002 |
| $\sigma_p$ | 0.3260 | 0.0430 |
| $\sigma_n$ | 0.4190 | 0.0728 |
