```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-06-03T14:41:38`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7999`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows only the single likelihood-best admissible estimate. Standard errors are computed at the reporting stage for this selected model.

```{note}
Flagged 104 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## Selected Best Model

Best admissible estimate ranked by corrected log likelihood.

| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ARX(2,2) | 16 | 1 | 41.196715 | -52.393430 | -1.833859 | 410.214941 | 30.817130 | yes |

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
\pi_{t+1} = \underset{(5.818876)}{0.053818} + \underset{(88.395174)}{0.227540}\,\pi_t + \underset{(16.743121)}{0.007305}\,\pi_{t-1} + \underset{(2328.531061)}{0.779104}\,SPF_t + \underset{(1925.494113)}{0.212611}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.018974)}{0.046637}\,\omega_{p,t} - \underset{(28.601672)}{0.273377}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(996.770567)}{5.158554} + \underset{(1.168214)}{0.906100}\,p_{t-1} + \frac{\underset{(1.576196)}{0.010131}}{2(\underset{(0.018974)}{0.046637})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000031)}{0.037255}}{2(\underset{(0.018974)}{0.046637})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(40.890537)}{6.225894} + \underset{(0.384972)}{0.830220}\,n_{t-1} + \frac{\underset{(58.195570)}{0.114966}}{2(\underset{(28.601672)}{0.273377})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(31.255279)}{0.194239}}{2(\underset{(28.601672)}{0.273377})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.053818 | 5.818876 |
| rho_1 | 0.227540 | 88.395174 |
| rho_2 | 0.007305 | 16.743121 |
| phi_1 | 0.779104 | 2328.531061 |
| phi_2 | 0.212611 | 1925.494113 |
| p0 | 5.158554 | 996.770567 |
| n0 | 6.225894 | 40.890537 |
| rho_p | 0.906100 | 1.168214 |
| rho_n | 0.830220 | 0.384972 |
| phi_p_plus | 0.010131 | 1.576196 |
| phi_p_minus | 0.037255 | 0.000031 |
| phi_n_plus | 0.114966 | 58.195570 |
| phi_n_minus | 0.194239 | 31.255279 |
| sigma_p | 0.046637 | 0.018974 |
| sigma_n | 0.273377 | 28.601672 |
