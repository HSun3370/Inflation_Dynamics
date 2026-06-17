```{raw:typst}
#set page(margin: auto)
```

# Constant-n Full BEGE Best Model Summary

Generated: `2026-06-15T20:58:22`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 2 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| constant | 43 | 36 | -134.6749 | 283.3499 | 306.9443 | 33062.2733 | 3.2741 | yes |

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
| $p_t$ | 55.7815 | 1097.3843 | 15730.8885 |
| $n_t$ | 9.7141 | 9.7141 | 9.7141 |
| $\sigma_t^2$ | 0.3153 | 0.4086 | 1.7204 |
| $s_t^2$ | -0.1108 | -0.1090 | -0.0842 |
| $k_t^2$ | 0.0595 | 0.0595 | 0.0602 |

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0001)}{0.0095}\,\omega_{p,t} - \underset{(0.0957)}{0.1787}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1406.1875)}{4.5998} + \underset{(0.0000)}{0.5260}\,p_{t-1} + \frac{\underset{(19.4463)}{0.9425}}{2(\underset{(0.0001)}{0.0095})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0000)}{0.0001}}{2(\underset{(0.0001)}{0.0095})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(1.9364)}{9.7141}
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 4.5998 | 1406.1875 |
| n0 | 9.7141 | 1.9364 |
| rho_p | 0.5260 | 0.0000 |
| phi_p_plus | 0.9425 | 19.4463 |
| phi_p_minus | 0.0001 | 0.0000 |
| sigma_p | 0.0095 | 0.0001 |
| sigma_n | 0.1787 | 0.0957 |
