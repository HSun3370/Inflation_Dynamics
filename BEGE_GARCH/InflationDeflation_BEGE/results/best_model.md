```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-06-18T00:47:59`
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
| ARX(2,2) | 3 | 39 | -167.9018 | 361.8035 | 405.6218 | 47.6988 | 12.0756 | no |

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
| $p_t$ | 4.5793 | 7.2032 | 28.6664 |
| $n_t$ | 0.0584 | 0.0697 | 0.6744 |
| $\sigma_t^2$ | 0.1699 | 0.2766 | 1.0806 |
| $s_t^2$ | -1.2028 | -0.0927 | 0.0660 |
| $k_t^2$ | 0.3575 | 0.4378 | 3.8913 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0735)}{0.1246} + \underset{(0.0770)}{0.2617}\,\pi_t + \underset{(0.0767)}{0.1743}\,\pi_{t-1} + \underset{(0.3909)}{0.2915}\,SPF_t + \underset{(0.3305)}{0.1749}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0430)}{0.1486}\,\omega_{p,t} - \underset{(0.6371)}{0.9893}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.8133)}{1.2658} + \underset{(0.1217)}{0.7076}\,p_{t-1} + \frac{\underset{(0.1758)}{0.4250}}{2(\underset{(0.0430)}{0.1486})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.0512)}{0.0508} + \underset{(0.0585)}{0.1317}\,n_{t-1} + \frac{\underset{(0.9302)}{1.2626}}{2(\underset{(0.6371)}{0.9893})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.1246 | 0.0735 |
| rho_1 | 0.2617 | 0.0770 |
| rho_2 | 0.1743 | 0.0767 |
| phi_1 | 0.2915 | 0.3909 |
| phi_2 | 0.1749 | 0.3305 |
| p0 | 1.2658 | 0.8133 |
| n0 | 0.0508 | 0.0512 |
| rho_p | 0.7076 | 0.1217 |
| rho_n | 0.1317 | 0.0585 |
| phi_p_plus | 0.4250 | 0.1758 |
| phi_n_minus | 1.2626 | 0.9302 |
| sigma_p | 0.1486 | 0.0430 |
| sigma_n | 0.9893 | 0.6371 |
