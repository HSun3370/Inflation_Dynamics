```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-06-11T15:50:34`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 79 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,1) | 97 | 23 | 246.5203 | -471.0405 | -433.9635 | 1825.7331 | 2.0443 | yes |

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
| $p_t$ | 73.6284 | 191.2480 | 628.2197 |
| $n_t$ | 48.8297 | 54.5960 | 77.8556 |
| $\sigma_t^2$ | 0.5941 | 0.7148 | 1.0726 |
| $s_t^2$ | -0.1848 | -0.1312 | -0.1195 |
| $k_t^2$ | 0.0390 | 0.0437 | 0.0627 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0371)}{0.0255} + \underset{(0.0074)}{0.2294}\,\pi_t + \underset{(0.0354)}{0.1191}\,\pi_{t-1} + \underset{(0.1255)}{0.6355}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0000)}{0.0204}\,\omega_{p,t} - \underset{(0.0125)}{0.1073}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.4662)}{4.2041} + \underset{(0.0000)}{0.8848}\,p_{t-1} + \frac{\underset{(0.0000)}{0.0347}}{2(\underset{(0.0000)}{0.0204})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0363)}{0.0757}}{2(\underset{(0.0000)}{0.0204})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.0686)}{5.4609} + \underset{(0.0000)}{0.8848}\,n_{t-1} + \frac{\underset{(0.0000)}{0.0347}}{2(\underset{(0.0125)}{0.1073})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0363)}{0.0757}}{2(\underset{(0.0125)}{0.1073})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0255 | 0.0371 |
| rho_1 | 0.2294 | 0.0074 |
| rho_2 | 0.1191 | 0.0354 |
| phi_1 | 0.6355 | 0.1255 |
| p0 | 4.2041 | 3.4662 |
| n0 | 5.4609 | 0.0686 |
| rho | 0.8848 | 0.0000 |
| phi_plus | 0.0347 | 0.0000 |
| phi_minus | 0.0757 | 0.0363 |
| sigma_p | 0.0204 | 0.0000 |
| sigma_n | 0.1073 | 0.0125 |
