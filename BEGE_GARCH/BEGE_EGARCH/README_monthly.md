```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# BEGE-EGARCH Best Model Summary (Monthly)

Generated: `2026-11-12T14:28:45`
Total estimations: `2400`
Converged estimations: `2400`
Eligible estimations for best-model selection: `2400`

This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage and reported in the parameter table.

CSV outputs:

```{raw:typst}
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results_monthly/by_mean/constant.csv")[constant cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results_monthly/by_mean/ARX_1_0.csv")[ARX(1,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results_monthly/by_mean/ARX_1_1.csv")[ARX(1,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results_monthly/by_mean/ARX_2_0.csv")[ARX(2,0) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results_monthly/by_mean/ARX_2_1.csv")[ARX(2,1) cleaned rows]
- #link("https://github.com/HSun3370/Inflation_Dynamics/blob/main/BEGE_GARCH/BEGE_EGARCH/results_monthly/by_mean/ARX_2_2.csv")[ARX(2,2) cleaned rows]
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | N | Sample |
|---|---:|---:|---:|---:|---:|---:|---|
| ARX(2,2) | 9 | 6 | 106.9987 | -183.9974 | -115.9036 | 692 | 1969M2--2026M9 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.9455 | 1.4297 | 4.0383 |
| $n_t$ | 0.2873 | 0.9609 | 6.6203 |
| $\sigma_t^2$ | 0.0234 | 0.0448 | 0.1647 |
| $s_t^2$ | -0.0274 | 0.0011 | 0.0100 |
| $k_t^2$ | 0.0023 | 0.0044 | 0.0168 |

Mean process:

$$
\pi_{t+1} = 0.0166 + 0.3755\,\pi_t + 0.0161\,\pi_{t-1} + 0.1448\,SPF_t + 0.4864\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.1235\,\omega_{p,t} - 0.1344\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
\ln p_t &= 0.2038 + 0.6658\,\ln p_{t-1} + 0.4608\,\left(|z_{t-1}| - \sqrt{2/\pi}\right) - 0.2327\,z_{t-1},\\
\ln n_t &= 0.0205 + 0.9292\,\ln n_{t-1} + 0.2846\,\left(|z_{t-1}| - \sqrt{2/\pi}\right) + 0.1967\,z_{t-1},\\
z_t &= u_t/\sigma_t,\qquad \sigma_t^2 = (0.1235)^2\,p_t + (0.1344)^2\,n_t
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| $c$ | 0.0166 | 0.0140 |
| $\rho_1$ | 0.3755 | 0.0420 |
| $\rho_2$ | 0.0161 | 0.0360 |
| $\phi_1$ | 0.1448 | 0.2517 |
| $\phi_2$ | 0.4864 | 0.2416 |
| $\omega_p$ | 0.2038 | 0.0479 |
| $\omega_n$ | 0.0205 | 0.0246 |
| $\beta_p$ | 0.6658 | 0.0697 |
| $\beta_n$ | 0.9292 | 0.0188 |
| $\alpha_p$ | 0.4608 | 0.0711 |
| $\alpha_n$ | 0.2846 | 0.0602 |
| $\gamma_p$ | -0.2327 | 0.0645 |
| $\gamma_n$ | 0.1967 | 0.0525 |
| $\sigma_p$ | 0.1235 | 0.0019 |
| $\sigma_n$ | 0.1344 | 0.0255 |