```{raw:typst}
#set page(margin: auto)
```

# Constant-n Full BEGE Best Model Summary

Generated: `2026-11-10T21:12:12`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

- [all estimations](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/all_estimations.csv)
- [best model with SE](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/best_loglik_with_se.csv)
- [selection diagnostics](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/selection_diagnostics.csv)
- [path quantiles](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/path_quantile_diagnostics.csv)
- [constant cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/constant.csv)
- [ARX(1,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_1_0.csv)
- [ARX(1,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_1_1.csv)
- [ARX(2,0) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_2_0.csv)
- [ARX(2,1) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_2_1.csv)
- [ARX(2,2) cleaned rows](https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/ConstantN_BEGE/results/by_mean/ARX_2_2.csv)

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| ARX(2,2) | 6 | 38 | -176.6814 | 377.3629 | 418.6198 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 4.1260 | 5.8644 | 26.8262 |
| $n_t$ | 0.0826 | 0.0826 | 0.0826 |
| $\sigma_t^2$ | 0.2224 | 0.2724 | 0.8751 |
| $s_t^2$ | -0.1924 | -0.1754 | 0.0290 |
| $k_t^2$ | 0.8027 | 0.8113 | 0.9153 |

Mean process:

$$
\pi_{t+1} = 0.1778 + 0.2268\,\pi_t + 0.1089\,\pi_{t-1} + 0.5669\,SPF_t + -0.0473\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1696\,\omega_{p,t} - 1.1209\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.8540 + 0.5334\,p_{t-1} + \frac{0.6482}{2(0.1696)^2}\,(u_{t-1}^+)^2 + \frac{0.0583}{2(0.1696)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.0826
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.1778 | 0.0713 |
| $\rho_1$ | 0.2268 | 0.0853 |
| $\rho_2$ | 0.1089 | 0.1137 |
| $\phi_1$ | 0.5669 | 0.1112 |
| $\phi_2$ | -0.0473 | 0.0027 |
| $p_0$ | 1.8540 | 0.6786 |
| $n_0$ | 0.0826 | 0.0679 |
| $\rho_p$ | 0.5334 | 0.1442 |
| $\phi_p^+$ | 0.6482 | 0.1857 |
| $\phi_p^-$ | 0.0583 | 0.0678 |
| $\sigma_p$ | 0.1696 | 0.0003 |
| $\sigma_n$ | 1.1209 | 0.7420 |
