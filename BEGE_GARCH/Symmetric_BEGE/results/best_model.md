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
| ARX(2,1) | 21 | 40 | 253.476194 | -484.952389 | -447.875370 | 271.160360 | 23.471739 | yes |

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
\pi_{t+1} = \underset{(30.151503)}{-0.027867} + \underset{(1.229643)}{0.408254}\,\pi_t + \underset{(20.430019)}{-0.033101}\,\pi_{t-1} + \underset{(26.729075)}{0.665318}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.234846)}{0.076122}\,\omega_{p,t} - \underset{(4.391602)}{0.495743}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(66.657177)}{8.161163} + \underset{(0.000000)}{0.939951}\,p_{t-1} + \frac{\underset{(0.069776)}{0.050597}}{2(\underset{(1.234846)}{0.076122})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.164779)}{0.009308}}{2(\underset{(1.234846)}{0.076122})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(6.957336)}{2.682046} + \underset{(0.000000)}{0.939951}\,n_{t-1} + \frac{\underset{(0.069776)}{0.050597}}{2(\underset{(4.391602)}{0.495743})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.164779)}{0.009308}}{2(\underset{(4.391602)}{0.495743})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.027867 | 30.151503 |
| rho_1 | 0.408254 | 1.229643 |
| rho_2 | -0.033101 | 20.430019 |
| phi_1 | 0.665318 | 26.729075 |
| p0 | 8.161163 | 66.657177 |
| n0 | 2.682046 | 6.957336 |
| rho | 0.939951 | 0.000000 |
| phi_plus | 0.050597 | 0.069776 |
| phi_minus | 0.009308 | 1.164779 |
| sigma_p | 0.076122 | 1.234846 |
| sigma_n | 0.495743 | 4.391602 |
