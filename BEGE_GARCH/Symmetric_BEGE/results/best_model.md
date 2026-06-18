```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-06-17T23:33:23`
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
| ARX(2,2) | 35 | 12 | -167.9012 | 359.8025 | 400.2501 | 9.5259 | 2.4576 | no |

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
| $p_t$ | 0.6347 | 1.1647 | 5.6629 |
| $n_t$ | 0.3243 | 0.5949 | 2.8917 |
| $\sigma_t^2$ | 0.1638 | 0.3005 | 1.4610 |
| $s_t^2$ | -0.2097 | -0.0432 | -0.0235 |
| $k_t^2$ | 0.1876 | 0.3441 | 1.6727 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0000)}{0.2006} + \underset{(0.0000)}{0.1978}\,\pi_t + \underset{(0.0000)}{0.2405}\,\pi_{t-1} + \underset{(0.0000)}{0.2257}\,SPF_t + \underset{(0.0000)}{0.1197}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0000)}{0.3592}\,\omega_{p,t} - \underset{(0.0000)}{0.5026}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.0000)}{0.1762} + \underset{(0.0000)}{0.6852}\,p_{t-1} + \frac{\underset{(0.0000)}{0.4128}}{2(\underset{(0.0000)}{0.3592})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0000)}{0.0809}}{2(\underset{(0.0000)}{0.3592})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.0000)}{0.0900} + \underset{(0.0000)}{0.6852}\,n_{t-1} + \frac{\underset{(0.0000)}{0.4128}}{2(\underset{(0.0000)}{0.5026})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0000)}{0.0809}}{2(\underset{(0.0000)}{0.5026})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.2006 | 0.0000 |
| rho_1 | 0.1978 | 0.0000 |
| rho_2 | 0.2405 | 0.0000 |
| phi_1 | 0.2257 | 0.0000 |
| phi_2 | 0.1197 | 0.0000 |
| p0 | 0.1762 | 0.0000 |
| n0 | 0.0900 | 0.0000 |
| rho | 0.6852 | 0.0000 |
| phi_plus | 0.4128 | 0.0000 |
| phi_minus | 0.0809 | 0.0000 |
| sigma_p | 0.3592 | 0.0000 |
| sigma_n | 0.5026 | 0.0000 |
