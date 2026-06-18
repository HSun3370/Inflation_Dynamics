```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-06-17T23:17:55`
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
| ARX(2,2) | 13 | 33 | -165.2805 | 360.5611 | 411.1206 | 5.6241 | 2.0217 | no |

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
| $p_t$ | 0.9806 | 1.1858 | 2.3678 |
| $n_t$ | 0.0849 | 0.2556 | 2.8363 |
| $\sigma_t^2$ | 0.1834 | 0.2969 | 1.1546 |
| $s_t^2$ | -0.9826 | 0.0427 | 0.1478 |
| $k_t^2$ | 0.1925 | 0.3661 | 2.0843 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0000)}{0.1937} + \underset{(0.0000)}{0.1415}\,\pi_t + \underset{(0.0760)}{0.1380}\,\pi_{t-1} + \underset{(0.0000)}{0.2283}\,SPF_t + \underset{(0.1402)}{0.3635}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0000)}{0.3802}\,\omega_{p,t} - \underset{(0.0321)}{0.5787}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.0000)}{0.0347} + \underset{(0.0000)}{0.9561}\,p_{t-1} + \frac{\underset{(0.0002)}{0.0000}}{2(\underset{(0.0000)}{0.3802})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0002)}{0.0334}}{2(\underset{(0.0000)}{0.3802})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.0001)}{0.0295} + \underset{(0.0000)}{0.5611}\,n_{t-1} + \frac{\underset{(0.0722)}{0.7784}}{2(\underset{(0.0321)}{0.5787})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0249)}{0.0720}}{2(\underset{(0.0321)}{0.5787})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.1937 | 0.0000 |
| rho_1 | 0.1415 | 0.0000 |
| rho_2 | 0.1380 | 0.0760 |
| phi_1 | 0.2283 | 0.0000 |
| phi_2 | 0.3635 | 0.1402 |
| p0 | 0.0347 | 0.0000 |
| n0 | 0.0295 | 0.0001 |
| rho_p | 0.9561 | 0.0000 |
| rho_n | 0.5611 | 0.0000 |
| phi_p_plus | 0.0000 | 0.0002 |
| phi_p_minus | 0.0334 | 0.0002 |
| phi_n_plus | 0.7784 | 0.0722 |
| phi_n_minus | 0.0720 | 0.0249 |
| sigma_p | 0.3802 | 0.0000 |
| sigma_n | 0.5787 | 0.0321 |
