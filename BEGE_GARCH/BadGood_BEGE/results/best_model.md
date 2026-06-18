```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-06-17T19:37:09`
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
| ARX(2,1) | 39 | 14 | -163.5594 | 351.1188 | 391.5664 | 40.7220 | 8.9086 | no |

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
| $p_t$ | 0.4097 | 0.5799 | 1.7285 |
| $n_t$ | 0.2985 | 0.7108 | 3.8717 |
| $\sigma_t^2$ | 0.1416 | 0.2554 | 1.0320 |
| $s_t^2$ | -0.2196 | 0.0384 | 0.0683 |
| $k_t^2$ | 0.1754 | 0.2991 | 1.1456 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0000)}{0.2119} + \underset{(0.0000)}{0.2240}\,\pi_t + \underset{(0.0000)}{0.2742}\,\pi_{t-1} + \underset{(0.0000)}{0.2699}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0000)}{0.4778}\,\omega_{p,t} - \underset{(0.0000)}{0.4042}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.0000)}{0.1575} + \underset{(0.0000)}{0.5795}\,p_{t-1} + \frac{\underset{(0.0000)}{0.2143}}{2(\underset{(0.0000)}{0.4778})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.0000)}{0.1900} + \underset{(0.0000)}{0.2831}\,n_{t-1} + \frac{\underset{(0.0000)}{0.6633}}{2(\underset{(0.0000)}{0.4042})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.2119 | 0.0000 |
| rho_1 | 0.2240 | 0.0000 |
| rho_2 | 0.2742 | 0.0000 |
| phi_1 | 0.2699 | 0.0000 |
| p0 | 0.1575 | 0.0000 |
| n0 | 0.1900 | 0.0000 |
| rho_p | 0.5795 | 0.0000 |
| rho_n | 0.2831 | 0.0000 |
| phi_p | 0.2143 | 0.0000 |
| phi_n | 0.6633 | 0.0000 |
| sigma_p | 0.4778 | 0.0000 |
| sigma_n | 0.4042 | 0.0000 |
