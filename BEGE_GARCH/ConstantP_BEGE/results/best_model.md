```{raw:typst}
#set page(margin: auto)
```

# Constant-p Full BEGE Best Model Summary

Generated: `2026-06-17T22:16:52`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,2) | 28 | 25 | -166.7909 | 357.5819 | 398.0296 | 15.5250 | 2.4032 | no |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability restrictions: `yes`
- Shape upper-cap diagnostic: `not flagged`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 0.4727 | 0.4727 | 0.4727 |
| $n_t$ | 0.4590 | 0.9268 | 7.3165 |
| $\sigma_t^2$ | 0.1990 | 0.2674 | 1.2023 |
| $s_t^2$ | -0.6797 | 0.0355 | 0.0878 |
| $k_t^2$ | 0.2795 | 0.3396 | 1.1602 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0000)}{0.1831} + \underset{(0.0000)}{0.1660}\,\pi_t + \underset{(0.0000)}{0.0787}\,\pi_{t-1} + \underset{(0.0000)}{0.2999}\,SPF_t + \underset{(0.0000)}{0.3076}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0000)}{0.5281}\,\omega_{p,t} - \underset{(0.0000)}{0.3825}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.0000)}{0.4727},\\
n_t &= \underset{(0.0000)}{0.2113} + \underset{(0.0000)}{0.4747}\,n_{t-1} + \frac{\underset{(0.0000)}{0.7577}}{2(\underset{(0.0000)}{0.3825})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0000)}{0.2362}}{2(\underset{(0.0000)}{0.3825})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.1831 | 0.0000 |
| rho_1 | 0.1660 | 0.0000 |
| rho_2 | 0.0787 | 0.0000 |
| phi_1 | 0.2999 | 0.0000 |
| phi_2 | 0.3076 | 0.0000 |
| p0 | 0.4727 | 0.0000 |
| n0 | 0.2113 | 0.0000 |
| rho_n | 0.4747 | 0.0000 |
| phi_n_plus | 0.7577 | 0.0000 |
| phi_n_minus | 0.2362 | 0.0000 |
| sigma_p | 0.5281 | 0.0000 |
| sigma_n | 0.3825 | 0.0000 |
