```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-06-03T14:51:48`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7999`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 54 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,1) | 21 | 40 | 253.4762 | -484.9524 | -447.8754 | 271.1604 | 23.4717 | yes |

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
\pi_{t+1} = \underset{(30.1515)}{-0.0279} + \underset{(1.2296)}{0.4083}\,\pi_t + \underset{(20.4300)}{-0.0331}\,\pi_{t-1} + \underset{(26.7291)}{0.6653}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.2348)}{0.0761}\,\omega_{p,t} - \underset{(4.3916)}{0.4957}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(66.6572)}{8.1612} + \underset{(0.0000)}{0.9400}\,p_{t-1} + \frac{\underset{(0.0698)}{0.0506}}{2(\underset{(1.2348)}{0.0761})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.1648)}{0.0093}}{2(\underset{(1.2348)}{0.0761})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(6.9573)}{2.6820} + \underset{(0.0000)}{0.9400}\,n_{t-1} + \frac{\underset{(0.0698)}{0.0506}}{2(\underset{(4.3916)}{0.4957})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.1648)}{0.0093}}{2(\underset{(4.3916)}{0.4957})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.0279 | 30.1515 |
| rho_1 | 0.4083 | 1.2296 |
| rho_2 | -0.0331 | 20.4300 |
| phi_1 | 0.6653 | 26.7291 |
| p0 | 8.1612 | 66.6572 |
| n0 | 2.6820 | 6.9573 |
| rho | 0.9400 | 0.0000 |
| phi_plus | 0.0506 | 0.0698 |
| phi_minus | 0.0093 | 1.1648 |
| sigma_p | 0.0761 | 1.2348 |
| sigma_n | 0.4957 | 4.3916 |
