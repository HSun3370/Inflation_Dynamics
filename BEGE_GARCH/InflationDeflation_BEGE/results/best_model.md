```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-06-03T14:44:25`
Total estimations: `8000`
Converged estimations: `7310`
Eligible estimations for best-model selection: `7273`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 106 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,2) | 4 | 20 | -37.4525 | 100.9050 | 144.7233 | 380.8158 | 97.5593 | yes |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability and variance restrictions: `yes`
- Shape upper-cap diagnostic: `flagged`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.0000)}{0.0018} + \underset{(0.0000)}{0.2393}\,\pi_t + \underset{(0.0000)}{0.3107}\,\pi_{t-1} + \underset{(0.0054)}{-0.4637}\,SPF_t + \underset{(0.0054)}{0.0534}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0000)}{0.1808}\,\omega_{p,t} - \underset{(0.0000)}{1.8998}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.0000)}{6.4515} + \underset{(0.0000)}{0.9551}\,p_{t-1} + \frac{\underset{(0.0000)}{0.0559}}{2(\underset{(0.0000)}{0.1808})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.0000)}{0.1348} + \underset{(0.0000)}{0.9943}\,n_{t-1} + \frac{\underset{(0.0000)}{0.0000}}{2(\underset{(0.0000)}{1.8998})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0018 | 0.0000 |
| rho_1 | 0.2393 | 0.0000 |
| rho_2 | 0.3107 | 0.0000 |
| phi_1 | -0.4637 | 0.0054 |
| phi_2 | 0.0534 | 0.0054 |
| p0 | 6.4515 | 0.0000 |
| n0 | 0.1348 | 0.0000 |
| rho_p | 0.9551 | 0.0000 |
| rho_n | 0.9943 | 0.0000 |
| phi_p_plus | 0.0559 | 0.0000 |
| phi_n_minus | 0.0000 | 0.0000 |
| sigma_p | 0.1808 | 0.0000 |
| sigma_n | 1.8998 | 0.0000 |
