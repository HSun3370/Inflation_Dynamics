```{raw:typst}
#set page(margin: auto)
```

# Constant-n Full BEGE Best Model Summary

Generated: `2026-06-11T18:31:56`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7977`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 4340 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,1) | 50 | 13 | 1779.2094 | -3536.4188 | -3499.3418 | 148.4118 | 138.0274 | yes |

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
| $p_t$ | 109.3892 | 109.4497 | 124.2171 |
| $n_t$ | 0.3744 | 0.3744 | 0.3744 |
| $\sigma_t^2$ | 102.0023 | 102.0581 | 115.6912 |
| $s_t^2$ | 190.7165 | 190.8238 | 217.0218 |
| $k_t^2$ | 575.9039 | 576.2132 | 651.7285 |

Mean process:

$$
\pi_{t+1} = \underset{(1.9522)}{-0.0065} + \underset{(1.2712)}{0.3704}\,\pi_t + \underset{(0.0805)}{-0.0379}\,\pi_{t-1} + \underset{(1.5715)}{0.8958}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.1955)}{0.9608}\,\omega_{p,t} - \underset{(1.2723)}{1.6469}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.0270)}{9.4705} + \underset{(0.0001)}{0.9134}\,p_{t-1} + \frac{\underset{(0.0735)}{0.0188}}{2(\underset{(1.1955)}{0.9608})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.0760)}{0.0268}}{2(\underset{(1.1955)}{0.9608})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.4704)}{0.3744}
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.0065 | 1.9522 |
| rho_1 | 0.3704 | 1.2712 |
| rho_2 | -0.0379 | 0.0805 |
| phi_1 | 0.8958 | 1.5715 |
| p0 | 9.4705 | 0.0270 |
| n0 | 0.3744 | 0.4704 |
| rho_p | 0.9134 | 0.0001 |
| phi_p_plus | 0.0188 | 0.0735 |
| phi_p_minus | 0.0268 | 0.0760 |
| sigma_p | 0.9608 | 1.1955 |
| sigma_n | 1.6469 | 1.2723 |
