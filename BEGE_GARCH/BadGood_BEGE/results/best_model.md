```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-06-15T20:41:40`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7998`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 150 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,1) | 45 | 25 | 4920.7089 | -9817.4177 | -9776.9701 | 157.9640 | 137.2035 | yes |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability restrictions: `yes`
- Shape upper-cap diagnostic: `not flagged`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Recursion initialization:

- Fixed $p_{\mathrm{init}}$ from Constant BEGE $\bar{p}$: `3.3171`
- Fixed $n_{\mathrm{init}}$ from Constant BEGE $\bar{n}$: `0.2281`
- Recursion intercept parameters `p0` and `n0`: `estimated`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 73.4005 | 157.5227 | 157.9092 |
| $n_t$ | 34.4913 | 37.3735 | 37.5844 |
| $\sigma_t^2$ | 119.9482 | 135.4737 | 135.9241 |
| $s_t^2$ | -450.2806 | -447.6353 | -415.7867 |
| $k_t^2$ | 2292.4773 | 2486.3708 | 2500.2918 |

Mean process:

$$
\pi_{t+1} = \underset{(13.6959)}{0.2973} + \underset{(31.5048)}{0.3673}\,\pi_t + \underset{(26.4588)}{-0.0266}\,\pi_{t-1} + \underset{(12.1536)}{1.2211}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.3331)}{0.2663}\,\omega_{p,t} - \underset{(4.1645)}{1.8239}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.0163)}{8.6687} + \underset{(0.0000)}{0.9451}\,p_{t-1} + \frac{\underset{(0.0698)}{0.0007}}{2(\underset{(1.3331)}{0.2663})^2}\,u_{t-1}^2,\\
n_t &= \underset{(52.1786)}{7.9303} + \underset{(1.4795)}{0.7874}\,n_{t-1} + \frac{\underset{(5.3891)}{0.1604}}{2(\underset{(4.1645)}{1.8239})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.2973 | 13.6959 |
| rho_1 | 0.3673 | 31.5048 |
| rho_2 | -0.0266 | 26.4588 |
| phi_1 | 1.2211 | 12.1536 |
| p0 | 8.6687 | 1.0163 |
| n0 | 7.9303 | 52.1786 |
| rho_p | 0.9451 | 0.0000 |
| rho_n | 0.7874 | 1.4795 |
| phi_p | 0.0007 | 0.0698 |
| phi_n | 0.1604 | 5.3891 |
| sigma_p | 0.2663 | 1.3331 |
| sigma_n | 1.8239 | 4.1645 |
