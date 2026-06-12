```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-06-11T15:44:01`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7999`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 100 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,2) | 16 | 1 | 41.1967 | -52.3934 | -1.8339 | 410.2149 | 30.8171 | yes |

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
| $p_t$ | 63.5349 | 76.2473 | 136.7380 |
| $n_t$ | 37.3094 | 39.0378 | 89.0824 |
| $\sigma_t^2$ | 2.9288 | 3.0913 | 6.8097 |
| $s_t^2$ | -3.6258 | -1.5785 | -1.5118 |
| $k_t^2$ | 1.2521 | 1.3105 | 2.9873 |

Mean process:

$$
\pi_{t+1} = \underset{(5.8738)}{0.0538} + \underset{(88.4443)}{0.2275}\,\pi_t + \underset{(16.8605)}{0.0073}\,\pi_{t-1} + \underset{(2328.8940)}{0.7791}\,SPF_t + \underset{(1925.1570)}{0.2126}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0190)}{0.0466}\,\omega_{p,t} - \underset{(28.5820)}{0.2734}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(996.1017)}{5.1586} + \underset{(1.1699)}{0.9061}\,p_{t-1} + \frac{\underset{(1.5773)}{0.0101}}{2(\underset{(0.0190)}{0.0466})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0000)}{0.0373}}{2(\underset{(0.0190)}{0.0466})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(40.8070)}{6.2259} + \underset{(0.3844)}{0.8302}\,n_{t-1} + \frac{\underset{(58.2336)}{0.1150}}{2(\underset{(28.5820)}{0.2734})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(31.2592)}{0.1942}}{2(\underset{(28.5820)}{0.2734})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0538 | 5.8738 |
| rho_1 | 0.2275 | 88.4443 |
| rho_2 | 0.0073 | 16.8605 |
| phi_1 | 0.7791 | 2328.8940 |
| phi_2 | 0.2126 | 1925.1570 |
| p0 | 5.1586 | 996.1017 |
| n0 | 6.2259 | 40.8070 |
| rho_p | 0.9061 | 1.1699 |
| rho_n | 0.8302 | 0.3844 |
| phi_p_plus | 0.0101 | 1.5773 |
| phi_p_minus | 0.0373 | 0.0000 |
| phi_n_plus | 0.1150 | 58.2336 |
| phi_n_minus | 0.1942 | 31.2592 |
| sigma_p | 0.0466 | 0.0190 |
| sigma_n | 0.2734 | 28.5820 |
