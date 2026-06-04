```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-06-03T14:41:47`
Total estimations: `8000`
Successful estimations: `7997`
Eligible estimations for best-model selection: `7997`

Selection screen: successful optimizer status, finite positive BEGE parameters, documented parameter bounds, EWMA implied-variance bounds, positive conditional variance, and mean-process stationarity. Log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

## Selected Best Model

Best admissible estimate ranked by log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,2) | 18 | 1 | -181.6884 | 381.3768 | 411.7126 | 0.3952 | no |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability and variance restrictions: `not applicable for fixed-shape Constant BEGE`
- Shape upper-cap diagnostic: `not applicable for fixed-shape Constant BEGE`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.1082)}{0.0042} + \underset{(0.1196)}{0.3232}\,\pi_t + \underset{(0.0834)}{0.1633}\,\pi_{t-1} + \underset{(0.3064)}{0.4107}\,SPF_t + \underset{(0.2891)}{0.1942}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0368)}{0.2712}\,\omega_{p,t} - \underset{(0.3484)}{0.8404}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.4671)}{2.6648},\qquad \bar{n} = \underset{(0.1606)}{0.2821},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0368)}{0.2712})^2\,\underset{(0.4671)}{2.6648} + (\underset{(0.3484)}{0.8404})^2\,\underset{(0.1606)}{0.2821}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0042 | 0.1082 |
| rho_1 | 0.3232 | 0.1196 |
| rho_2 | 0.1633 | 0.0834 |
| phi_1 | 0.4107 | 0.3064 |
| phi_2 | 0.1942 | 0.2891 |
| shape_p | 2.6648 | 0.4671 |
| shape_n | 0.2821 | 0.1606 |
| sigma_p | 0.2712 | 0.0368 |
| sigma_n | 0.8404 | 0.3484 |
