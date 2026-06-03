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
| ARX(2,2) | 18 | 1 | -181.688415 | 381.376830 | 411.712572 | 0.395222 | no |

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
\pi_{t+1} = \underset{(0.108204)}{0.004171} + \underset{(0.119603)}{0.323173}\,\pi_t + \underset{(0.083407)}{0.163315}\,\pi_{t-1} + \underset{(0.306386)}{0.410717}\,SPF_t + \underset{(0.289064)}{0.194172}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.036800)}{0.271204}\,\omega_{p,t} - \underset{(0.348411)}{0.840368}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.467080)}{2.664771},\qquad \bar{n} = \underset{(0.160632)}{0.282099},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.036800)}{0.271204})^2\,\underset{(0.467080)}{2.664771} + (\underset{(0.348411)}{0.840368})^2\,\underset{(0.160632)}{0.282099}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004171 | 0.108204 |
| rho_1 | 0.323173 | 0.119603 |
| rho_2 | 0.163315 | 0.083407 |
| phi_1 | 0.410717 | 0.306386 |
| phi_2 | 0.194172 | 0.289064 |
| shape_p | 2.664771 | 0.467080 |
| shape_n | 0.282099 | 0.160632 |
| sigma_p | 0.271204 | 0.036800 |
| sigma_n | 0.840368 | 0.348411 |
