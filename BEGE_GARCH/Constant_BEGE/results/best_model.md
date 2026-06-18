```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-06-17T18:34:22`
Total estimations: `8000`
Successful estimations: `8000`
Eligible estimations for best-model selection: `8000`

Selection screen: successful optimizer status, finite positive BEGE parameters, documented parameter bounds, EWMA implied-variance bounds, positive conditional variance, and mean-process stationarity. Log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows the likelihood-best admissible estimate for each mean process. Standard errors are computed at the reporting stage for these selected models.

## Selected Best Models by Mean Process

Best admissible estimates are ranked by log likelihood within each mean process. The marked `p_bar` and `n_bar` values are the fixed Constant BEGE shape estimates; they are also the recommended initial `p_0` and `n_0` values for follow-up dynamic-shape tests.

Initial-shape values saved to `BEGE_GARCH/Constant_BEGE/results/constant_bege_initial_shapes_by_mean.csv`.

| Mean | Overall Pick | Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Initial p_0 | Initial n_0 | Implied Var | Above -150 Diagnostic |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| constant | no | 20 | 1 | -199.8439 | 407.6878 | 421.1703 | **2.6759** | **0.1860** | 2.6759 | 0.1860 | 0.6979 | no |
| ARX(1,1) | no | 48 | 13 | -184.3962 | 382.7924 | 406.3868 | **2.6279** | **0.2811** | 2.6279 | 0.2811 | 0.3945 | no |
| ARX(2,1) | no | 31 | 26 | -181.8848 | 379.7696 | 406.7348 | **3.3171** | **0.2281** | 3.3171 | 0.2281 | 0.3982 | no |
| ARX(2,2) | yes | 30 | 30 | -181.6884 | 381.3768 | 411.7126 | **2.6646** | **0.2821** | 2.6646 | 0.2821 | 0.3952 | no |

### constant

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 20 | 1 | -199.8439 | 407.6878 | 421.1703 | **2.6759** | **0.1860** | 0.6979 | no |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability and variance restrictions: `not applicable for fixed-shape Constant BEGE`
- Shape upper-cap diagnostic: `not applicable for fixed-shape Constant BEGE`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.6759 | 2.6759 | 2.6759 |
| $n_t$ | 0.1860 | 0.1860 | 0.1860 |
| $\sigma_t^2$ | 0.6979 | 0.6979 | 0.6979 |
| $s_t^2$ | -1.3044 | -1.3044 | -1.3044 |
| $k_t^2$ | 6.9510 | 6.9510 | 6.9510 |

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0519)}{0.2983}\,\omega_{p,t} - \underset{(0.9266)}{1.5726}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.6439)}{2.6759},\qquad \bar{n} = \underset{(0.1186)}{0.1860},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0519)}{0.2983})^2\,\underset{(0.6439)}{2.6759} + (\underset{(0.9266)}{1.5726})^2\,\underset{(0.1186)}{0.1860}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.6759 | 0.6439 |
| shape_n | 0.1860 | 0.1186 |
| sigma_p | 0.2983 | 0.0519 |
| sigma_n | 1.5726 | 0.9266 |

### ARX(1,1)

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 48 | 13 | -184.3962 | 382.7924 | 406.3868 | **2.6279** | **0.2811** | 0.3945 | no |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability and variance restrictions: `not applicable for fixed-shape Constant BEGE`
- Shape upper-cap diagnostic: `not applicable for fixed-shape Constant BEGE`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.6279 | 2.6279 | 2.6279 |
| $n_t$ | 0.2811 | 0.2811 | 0.2811 |
| $\sigma_t^2$ | 0.3945 | 0.3945 | 0.3945 |
| $s_t^2$ | -0.1656 | -0.1656 | -0.1656 |
| $k_t^2$ | 0.7966 | 0.7966 | 0.7966 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0719)}{0.0560} + \underset{(0.0815)}{0.3237}\,\pi_t + \underset{(0.1149)}{0.7378}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0612)}{0.2857}\,\omega_{p,t} - \underset{(0.3443)}{0.8002}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.7678)}{2.6279},\qquad \bar{n} = \underset{(0.1684)}{0.2811},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0612)}{0.2857})^2\,\underset{(0.7678)}{2.6279} + (\underset{(0.3443)}{0.8002})^2\,\underset{(0.1684)}{0.2811}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0560 | 0.0719 |
| rho_1 | 0.3237 | 0.0815 |
| phi_1 | 0.7378 | 0.1149 |
| shape_p | 2.6279 | 0.7678 |
| shape_n | 0.2811 | 0.1684 |
| sigma_p | 0.2857 | 0.0612 |
| sigma_n | 0.8002 | 0.3443 |

### ARX(2,1)

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 31 | 26 | -181.8848 | 379.7696 | 406.7348 | **3.3171** | **0.2281** | 0.3982 | no |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability and variance restrictions: `not applicable for fixed-shape Constant BEGE`
- Shape upper-cap diagnostic: `not applicable for fixed-shape Constant BEGE`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 3.3171 | 3.3171 | 3.3171 |
| $n_t$ | 0.2281 | 0.2281 | 0.2281 |
| $\sigma_t^2$ | 0.3982 | 0.3982 | 0.3982 |
| $s_t^2$ | -0.2656 | -0.2656 | -0.2656 |
| $k_t^2$ | 1.0902 | 1.0902 | 1.0902 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0918)}{0.0392} + \underset{(0.0765)}{0.3168}\,\pi_t + \underset{(0.0982)}{0.1892}\,\pi_{t-1} + \underset{(0.1516)}{0.5392}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0710)}{0.2465}\,\omega_{p,t} - \underset{(0.4176)}{0.9284}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.4028)}{3.3171},\qquad \bar{n} = \underset{(0.1417)}{0.2281},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0710)}{0.2465})^2\,\underset{(1.4028)}{3.3171} + (\underset{(0.4176)}{0.9284})^2\,\underset{(0.1417)}{0.2281}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0392 | 0.0918 |
| rho_1 | 0.3168 | 0.0765 |
| rho_2 | 0.1892 | 0.0982 |
| phi_1 | 0.5392 | 0.1516 |
| shape_p | 3.3171 | 1.4028 |
| shape_n | 0.2281 | 0.1417 |
| sigma_p | 0.2465 | 0.0710 |
| sigma_n | 0.9284 | 0.4176 |

### ARX(2,2)

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 30 | 30 | -181.6884 | 381.3768 | 411.7126 | **2.6646** | **0.2821** | 0.3952 | no |

Selection checks:

- Optimizer convergence: `yes`
- Parameter bounds: `yes`
- BEGE stability and variance restrictions: `not applicable for fixed-shape Constant BEGE`
- Shape upper-cap diagnostic: `not applicable for fixed-shape Constant BEGE`
- Implied variance bounds: `yes`
- Mean-process stationarity: `yes`
- Selection diagnostics: `eligible`
- Standard errors: `computed`

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $p_t$ | 2.6646 | 2.6646 | 2.6646 |
| $n_t$ | 0.2821 | 0.2821 | 0.2821 |
| $\sigma_t^2$ | 0.3952 | 0.3952 | 0.3952 |
| $s_t^2$ | -0.2286 | -0.2286 | -0.2286 |
| $k_t^2$ | 0.9308 | 0.9308 | 0.9308 |

Mean process:

$$
\pi_{t+1} = \underset{(0.1115)}{0.0041} + \underset{(0.1214)}{0.3232}\,\pi_t + \underset{(0.0829)}{0.1633}\,\pi_{t-1} + \underset{(0.3014)}{0.4107}\,SPF_t + \underset{(0.2957)}{0.1942}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0218)}{0.2712}\,\omega_{p,t} - \underset{(0.3557)}{0.8404}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.1839)}{2.6646},\qquad \bar{n} = \underset{(0.1597)}{0.2821},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0218)}{0.2712})^2\,\underset{(0.1839)}{2.6646} + (\underset{(0.3557)}{0.8404})^2\,\underset{(0.1597)}{0.2821}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0041 | 0.1115 |
| rho_1 | 0.3232 | 0.1214 |
| rho_2 | 0.1633 | 0.0829 |
| phi_1 | 0.4107 | 0.3014 |
| phi_2 | 0.1942 | 0.2957 |
| shape_p | 2.6646 | 0.1839 |
| shape_n | 0.2821 | 0.1597 |
| sigma_p | 0.2712 | 0.0218 |
| sigma_n | 0.8404 | 0.3557 |
