```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-06-11T15:52:42`
Total estimations: `8000`
Converged estimations: `7841`
Eligible estimations for best-model selection: `7841`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 109 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(1,1) | 16 | 24 | 134.0336 | -246.0672 | -208.9902 | 3024.7460 | 11.5915 | yes |

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
| $p_t$ | 54.9936 | 56.0795 | 63.4918 |
| $n_t$ | 33.5494 | 125.5921 | 692.7979 |
| $\sigma_t^2$ | 4.1811 | 4.4715 | 5.9837 |
| $s_t^2$ | 2.1477 | 2.2139 | 2.4990 |
| $k_t^2$ | 1.8035 | 1.8418 | 2.0880 |

Mean process:

$$
\pi_{t+1} = \underset{(0.1663)}{0.3789} + \underset{(0.1689)}{0.4455}\,\pi_t + \underset{(0.3632)}{1.1357}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(3.4000)}{0.2718}\,\omega_{p,t} - \underset{(0.0002)}{0.0490}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.2521)}{4.2462} + \underset{(0.0000)}{0.9213}\,p_{t-1} + \frac{\underset{(0.1315)}{0.0220}}{2(\underset{(3.4000)}{0.2718})^2}\,u_{t-1}^2,\\
n_t &= \underset{(610.1096)}{9.6817} + \underset{(0.0000)}{0.2328}\,n_{t-1} + \frac{\underset{(2.9324)}{0.6270}}{2(\underset{(0.0002)}{0.0490})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.3789 | 0.1663 |
| rho_1 | 0.4455 | 0.1689 |
| phi_1 | 1.1357 | 0.3632 |
| p0 | 4.2462 | 3.2521 |
| n0 | 9.6817 | 610.1096 |
| rho_p | 0.9213 | 0.0000 |
| rho_n | 0.2328 | 0.0000 |
| phi_p | 0.0220 | 0.1315 |
| phi_n | 0.6270 | 2.9324 |
| sigma_p | 0.2718 | 3.4000 |
| sigma_n | 0.0490 | 0.0002 |
