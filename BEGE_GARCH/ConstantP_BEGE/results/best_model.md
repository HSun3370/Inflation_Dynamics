```{raw:typst}
#set page(margin: auto)
```

# Constant-p Full BEGE Best Model Summary

Generated: `2026-06-15T20:14:14`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `8000`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 1 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(1,1) | 45 | 22 | -141.3518 | 302.7036 | 336.4100 | 211263.4978 | 11.7220 | yes |

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
| $p_t$ | 6.9081 | 6.9081 | 6.9081 |
| $n_t$ | 39.9058 | 1002.7078 | 11882.7431 |
| $\sigma_t^2$ | 0.2363 | 0.2886 | 0.8803 |
| $s_t^2$ | 0.0767 | 0.0854 | 0.0862 |
| $k_t^2$ | 0.0476 | 0.0476 | 0.0478 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0167)}{0.1361} + \underset{(0.0000)}{0.5723}\,\pi_t + \underset{(0.0367)}{0.3535}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0579)}{0.1841}\,\omega_{p,t} - \underset{(0.0000)}{0.0074}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.1342)}{6.9081},\\
n_t &= \underset{(714.8479)}{9.0000} + \underset{(0.0002)}{0.0300}\,n_{t-1} + \frac{\underset{(5.6233)}{0.5947}}{2(\underset{(0.0000)}{0.0074})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(5.7328)}{1.3420}}{2(\underset{(0.0000)}{0.0074})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.1361 | 0.0167 |
| rho_1 | 0.5723 | 0.0000 |
| phi_1 | 0.3535 | 0.0367 |
| p0 | 6.9081 | 0.1342 |
| n0 | 9.0000 | 714.8479 |
| rho_n | 0.0300 | 0.0002 |
| phi_n_plus | 0.5947 | 5.6233 |
| phi_n_minus | 1.3420 | 5.7328 |
| sigma_p | 0.1841 | 0.0579 |
| sigma_n | 0.0074 | 0.0000 |
