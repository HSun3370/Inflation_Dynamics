```{raw:typst}
#set page(margin: auto)
```

# Constant-n Full BEGE Best Model Summary

Generated: `2026-06-17T20:39:10`
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
| ARX(2,2) | 31 | 40 | -170.2704 | 364.5408 | 404.9885 | 10.3406 | 1.4912 | no |

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
| $p_t$ | 0.6223 | 1.1422 | 5.6821 |
| $n_t$ | 0.4055 | 0.4055 | 0.4055 |
| $\sigma_t^2$ | 0.2157 | 0.2840 | 0.8798 |
| $s_t^2$ | -0.0950 | -0.0455 | 0.3862 |
| $k_t^2$ | 0.3302 | 0.3840 | 0.8532 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0000)}{0.1460} + \underset{(0.0000)}{0.1849}\,\pi_t + \underset{(0.0000)}{0.0812}\,\pi_{t-1} + \underset{(0.0000)}{0.9787}\,SPF_t + \underset{(0.0000)}{-0.3699}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0000)}{0.3623}\,\omega_{p,t} - \underset{(0.0000)}{0.5750}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.0000)}{0.1443} + \underset{(0.0000)}{0.7377}\,p_{t-1} + \frac{\underset{(0.0000)}{0.4074}}{2(\underset{(0.0000)}{0.3623})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0000)}{0.0184}}{2(\underset{(0.0000)}{0.3623})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.0000)}{0.4055}
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.1460 | 0.0000 |
| rho_1 | 0.1849 | 0.0000 |
| rho_2 | 0.0812 | 0.0000 |
| phi_1 | 0.9787 | 0.0000 |
| phi_2 | -0.3699 | 0.0000 |
| p0 | 0.1443 | 0.0000 |
| n0 | 0.4055 | 0.0000 |
| rho_p | 0.7377 | 0.0000 |
| phi_p_plus | 0.4074 | 0.0000 |
| phi_p_minus | 0.0184 | 0.0000 |
| sigma_p | 0.3623 | 0.0000 |
| sigma_n | 0.5750 | 0.0000 |
