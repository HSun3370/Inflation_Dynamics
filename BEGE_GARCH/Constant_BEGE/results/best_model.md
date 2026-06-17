```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-06-12T09:36:25`
Total estimations: `8000`
Successful estimations: `7997`
Eligible estimations for best-model selection: `7997`

Selection screen: successful optimizer status, finite positive BEGE parameters, documented parameter bounds, EWMA implied-variance bounds, positive conditional variance, and mean-process stationarity. Log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
This report shows the likelihood-best admissible estimate for each mean process. Standard errors are computed at the reporting stage for these selected models.

## Selected Best Models by Mean Process

Best admissible estimates are ranked by log likelihood within each mean process. The marked `p_bar` and `n_bar` values are the fixed Constant BEGE shape estimates; they are also the recommended initial `p_0` and `n_0` values for follow-up dynamic-shape tests.

Initial-shape values saved to `BEGE_GARCH/Constant_BEGE/results/constant_bege_initial_shapes_by_mean.csv`.

| Mean | Overall Pick | Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Initial p_0 | Initial n_0 | Implied Var | Above -150 Diagnostic |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| constant | no | 46 | 11 | -199.8439 | 407.6878 | 421.1703 | **2.6759** | **0.1860** | 2.6759 | 0.1860 | 0.6979 | no |
| ARX(1,1) | no | 35 | 31 | -184.3962 | 382.7924 | 406.3868 | **2.6277** | **0.2811** | 2.6277 | 0.2811 | 0.3945 | no |
| ARX(2,1) | no | 27 | 15 | -181.8848 | 379.7696 | 406.7348 | **3.3171** | **0.2281** | 3.3171 | 0.2281 | 0.3982 | no |
| ARX(2,2) | yes | 18 | 1 | -181.6884 | 381.3768 | 411.7126 | **2.6648** | **0.2821** | 2.6648 | 0.2821 | 0.3952 | no |

### constant

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 46 | 11 | -199.8439 | 407.6878 | 421.1703 | **2.6759** | **0.1860** | 0.6979 | no |

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
| $k_t^2$ | 6.9505 | 6.9505 | 6.9505 |

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0499)}{0.2983}\,\omega_{p,t} - \underset{(0.9321)}{1.5725}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.6149)}{2.6759},\qquad \bar{n} = \underset{(0.1191)}{0.1860},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0499)}{0.2983})^2\,\underset{(0.6149)}{2.6759} + (\underset{(0.9321)}{1.5725})^2\,\underset{(0.1191)}{0.1860}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.6759 | 0.6149 |
| shape_n | 0.1860 | 0.1191 |
| sigma_p | 0.2983 | 0.0499 |
| sigma_n | 1.5725 | 0.9321 |

### ARX(1,1)

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 35 | 31 | -184.3962 | 382.7924 | 406.3868 | **2.6277** | **0.2811** | 0.3945 | no |

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
| $p_t$ | 2.6277 | 2.6277 | 2.6277 |
| $n_t$ | 0.2811 | 0.2811 | 0.2811 |
| $\sigma_t^2$ | 0.3945 | 0.3945 | 0.3945 |
| $s_t^2$ | -0.1656 | -0.1656 | -0.1656 |
| $k_t^2$ | 0.7968 | 0.7968 | 0.7968 |

Mean process:

$$
\pi_{t+1} = \underset{(0.0712)}{0.0560} + \underset{(0.0811)}{0.3237}\,\pi_t + \underset{(0.1129)}{0.7378}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0563)}{0.2857}\,\omega_{p,t} - \underset{(0.3230)}{0.8003}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.6892)}{2.6277},\qquad \bar{n} = \underset{(0.1534)}{0.2811},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0563)}{0.2857})^2\,\underset{(0.6892)}{2.6277} + (\underset{(0.3230)}{0.8003})^2\,\underset{(0.1534)}{0.2811}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0560 | 0.0712 |
| rho_1 | 0.3237 | 0.0811 |
| phi_1 | 0.7378 | 0.1129 |
| shape_p | 2.6277 | 0.6892 |
| shape_n | 0.2811 | 0.1534 |
| sigma_p | 0.2857 | 0.0563 |
| sigma_n | 0.8003 | 0.3230 |

### ARX(2,1)

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 27 | 15 | -181.8848 | 379.7696 | 406.7348 | **3.3171** | **0.2281** | 0.3982 | no |

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
\pi_{t+1} = \underset{(0.0969)}{0.0392} + \underset{(0.0780)}{0.3168}\,\pi_t + \underset{(0.0983)}{0.1891}\,\pi_{t-1} + \underset{(0.1621)}{0.5392}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.1257)}{0.2465}\,\omega_{p,t} - \underset{(0.4207)}{0.9284}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(2.6133)}{3.3171},\qquad \bar{n} = \underset{(0.1425)}{0.2281},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.1257)}{0.2465})^2\,\underset{(2.6133)}{3.3171} + (\underset{(0.4207)}{0.9284})^2\,\underset{(0.1425)}{0.2281}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0392 | 0.0969 |
| rho_1 | 0.3168 | 0.0780 |
| rho_2 | 0.1891 | 0.0983 |
| phi_1 | 0.5392 | 0.1621 |
| shape_p | 3.3171 | 2.6133 |
| shape_n | 0.2281 | 0.1425 |
| sigma_p | 0.2465 | 0.1257 |
| sigma_n | 0.9284 | 0.4207 |

### ARX(2,2)

| Seed | Draw | LogLik | AIC | BIC | **p_bar** | **n_bar** | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 18 | 1 | -181.6884 | 381.3768 | 411.7126 | **2.6648** | **0.2821** | 0.3952 | no |

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
| $p_t$ | 2.6648 | 2.6648 | 2.6648 |
| $n_t$ | 0.2821 | 0.2821 | 0.2821 |
| $\sigma_t^2$ | 0.3952 | 0.3952 | 0.3952 |
| $s_t^2$ | -0.2285 | -0.2285 | -0.2285 |
| $k_t^2$ | 0.9307 | 0.9307 | 0.9307 |

Mean process:

$$
\pi_{t+1} = \underset{(0.1079)}{0.0042} + \underset{(0.1196)}{0.3232}\,\pi_t + \underset{(0.0834)}{0.1633}\,\pi_{t-1} + \underset{(0.3070)}{0.4107}\,SPF_t + \underset{(0.2935)}{0.1942}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.0338)}{0.2712}\,\omega_{p,t} - \underset{(0.3428)}{0.8404}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.4091)}{2.6648},\qquad \bar{n} = \underset{(0.1586)}{0.2821},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.0338)}{0.2712})^2\,\underset{(0.4091)}{2.6648} + (\underset{(0.3428)}{0.8404})^2\,\underset{(0.1586)}{0.2821}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.0042 | 0.1079 |
| rho_1 | 0.3232 | 0.1196 |
| rho_2 | 0.1633 | 0.0834 |
| phi_1 | 0.4107 | 0.3070 |
| phi_2 | 0.1942 | 0.2935 |
| shape_p | 2.6648 | 0.4091 |
| shape_n | 0.2821 | 0.1586 |
| sigma_p | 0.2712 | 0.0338 |
| sigma_n | 0.8404 | 0.3428 |
