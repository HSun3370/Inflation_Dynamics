```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-06-03T14:41:38`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7999`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 104 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,2) | 16 | 1 | 41.1967 | -52.3934 | -1.8339 | 410.2149 | 30.8171 | yes |

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
\pi_{t+1} = \underset{(5.8189)}{0.0538} + \underset{(88.3952)}{0.2275}\,\pi_t + \underset{(16.7431)}{0.0073}\,\pi_{t-1} + \underset{(2328.5311)}{0.7791}\,SPF_t + \underset{(1925.4941)}{0.2126}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0190)}{0.0466}\,\omega_{p,t} - \underset{(28.6017)}{0.2734}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(996.7706)}{5.1586} + \underset{(1.1682)}{0.9061}\,p_{t-1} + \frac{\underset{(1.5762)}{0.0101}}{2(\underset{(0.0190)}{0.0466})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0000)}{0.0373}}{2(\underset{(0.0190)}{0.0466})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(40.8905)}{6.2259} + \underset{(0.3850)}{0.8302}\,n_{t-1} + \frac{\underset{(58.1956)}{0.1150}}{2(\underset{(28.6017)}{0.2734})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(31.2553)}{0.1942}}{2(\underset{(28.6017)}{0.2734})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0538 | 5.8189 |
| rho_1 | 0.2275 | 88.3952 |
| rho_2 | 0.0073 | 16.7431 |
| phi_1 | 0.7791 | 2328.5311 |
| phi_2 | 0.2126 | 1925.4941 |
| p0 | 5.1586 | 996.7706 |
| n0 | 6.2259 | 40.8905 |
| rho_p | 0.9061 | 1.1682 |
| rho_n | 0.8302 | 0.3850 |
| phi_p_plus | 0.0101 | 1.5762 |
| phi_p_minus | 0.0373 | 0.0000 |
| phi_n_plus | 0.1150 | 58.1956 |
| phi_n_minus | 0.1942 | 31.2553 |
| sigma_p | 0.0466 | 0.0190 |
| sigma_n | 0.2734 | 28.6017 |
