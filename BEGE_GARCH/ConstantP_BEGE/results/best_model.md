```{raw:typst}
#set page(margin: auto)
```

# Constant-p Full BEGE Best Model Summary

Generated: `2026-06-11T17:23:03`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7976`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 2845 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,1) | 45 | 12 | 1668.0462 | -3314.0924 | -3277.0154 | 262.8530 | 72.9883 | yes |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability restrictions: `yes`
- Shape upper-cap diagnostic: `flagged`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.4367 | 0.4367 | 0.4367 |
| $n_t$ | 157.1363 | 157.6637 | 210.5382 |
| $\sigma_t^2$ | 43.8090 | 43.9546 | 58.5487 |
| $s_t^2$ | -60.1847 | -44.8501 | -44.6972 |
| $k_t^2$ | 74.4546 | 74.6957 | 98.8647 |

Mean process:

$$
\pi_{t+1} = \underset{(0.3410)}{0.1726} + \underset{(0.0488)}{0.3690}\,\pi_t + \underset{(0.1611)}{0.1871}\,\pi_{t-1} + \underset{(0.2022)}{0.5242}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0314)}{1.0007}\,\omega_{p,t} - \underset{(0.0251)}{0.5254}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.2607)}{0.4367},\\
n_t &= \underset{(0.0053)}{9.7071} + \underset{(0.0000)}{0.9382}\,n_{t-1} + \frac{\underset{(0.0004)}{0.0133}}{2(\underset{(0.0251)}{0.5254})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0005)}{0.0366}}{2(\underset{(0.0251)}{0.5254})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.1726 | 0.3410 |
| rho_1 | 0.3690 | 0.0488 |
| rho_2 | 0.1871 | 0.1611 |
| phi_1 | 0.5242 | 0.2022 |
| p0 | 0.4367 | 0.2607 |
| n0 | 9.7071 | 0.0053 |
| rho_n | 0.9382 | 0.0000 |
| phi_n_plus | 0.0133 | 0.0004 |
| phi_n_minus | 0.0366 | 0.0005 |
| sigma_p | 1.0007 | 0.0314 |
| sigma_n | 0.5254 | 0.0251 |
