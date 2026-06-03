```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-06-03T14:38:52`
Total estimations: `8000`
Converged estimations: `7841`
Eligible estimations for best-model selection: `7841`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 107 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(1,1) | 16 | 24 | 134.033609 | -246.067218 | -208.990199 | 3024.746008 | 11.591510 | yes |

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
\pi_{t+1} = \underset{(0.166260)}{0.378918} + \underset{(0.168886)}{0.445463}\,\pi_t + \underset{(0.363141)}{1.135712}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(3.399730)}{0.271844}\,\omega_{p,t} - \underset{(0.000202)}{0.048968}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.251517)}{4.246196} + \underset{(0.000002)}{0.921285}\,p_{t-1} + \frac{\underset{(0.131530)}{0.022016}}{2(\underset{(3.399730)}{0.271844})^2}\,u_{t-1}^2,\\
n_t &= \underset{(610.059564)}{9.681746} + \underset{(0.000010)}{0.232777}\,n_{t-1} + \frac{\underset{(2.932198)}{0.626957}}{2(\underset{(0.000202)}{0.048968})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.378918 | 0.166260 |
| rho_1 | 0.445463 | 0.168886 |
| phi_1 | 1.135712 | 0.363141 |
| p0 | 4.246196 | 3.251517 |
| n0 | 9.681746 | 610.059564 |
| rho_p | 0.921285 | 0.000002 |
| rho_n | 0.232777 | 0.000010 |
| phi_p | 0.022016 | 0.131530 |
| phi_n | 0.626957 | 2.932198 |
| sigma_p | 0.271844 | 3.399730 |
| sigma_n | 0.048968 | 0.000202 |
