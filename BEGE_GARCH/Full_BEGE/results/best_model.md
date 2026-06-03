```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-06-03T14:04:19`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7999`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `5` admissible estimates by corrected log likelihood.

```{note}
Flagged 104 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## constant

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 18 | 2 | 14.824079 | -9.648157 | 24.058223 | 175.813702 | 3.099039 | yes |
| 2 | 29 | 3 | -76.661083 | 173.322166 | 207.028547 | 641.332527 | 4.029585 | yes |
| 3 | 22 | 25 | -90.207296 | 200.414592 | 234.120972 | 311.596535 | 2.881161 | yes |
| 4 | 5 | 25 | -92.043366 | 204.086731 | 237.793111 | 2721.157655 | 2.792048 | yes |
| 5 | 15 | 32 | -102.920283 | 225.840566 | 259.546946 | 638.183734 | 12.687129 | yes |

### Rank 1: Seed 18, Draw 2

- LogLik: `14.824079`; AIC: `-9.648157`; BIC: `24.058223`
- Max shape path: `175.813702`; max implied variance: `3.099039`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.028983\,\omega_{p,t} - 0.234080\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 10.000000 + 0.943122\,p_{t-1} + \frac{0.000000}{2(0.028983)^2}\,(u_{t-1}^+)^2 + \frac{0.000000}{2(0.028983)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.053033 + 0.529879\,n_{t-1} + \frac{0.933640}{2(0.234080)^2}\,(u_{t-1}^+)^2 + \frac{0.000000}{2(0.234080)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 10.000000 |
| n0 | 0.053033 |
| rho_p | 0.943122 |
| rho_n | 0.529879 |
| phi_p_plus | 0.000000 |
| phi_p_minus | 0.000000 |
| phi_n_plus | 0.933640 |
| phi_n_minus | 0.000000 |
| sigma_p | 0.028983 |
| sigma_n | 0.234080 |

### Rank 2: Seed 29, Draw 3

- LogLik: `-76.661083`; AIC: `173.322166`; BIC: `207.028547`
- Max shape path: `641.332527`; max implied variance: `4.029585`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.048652\,\omega_{p,t} - 0.261142\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.140974 + 0.844859\,p_{t-1} + \frac{0.306113}{2(0.048652)^2}\,(u_{t-1}^+)^2 + \frac{0.000012}{2(0.048652)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.581006 + 0.364141\,n_{t-1} + \frac{0.746948}{2(0.261142)^2}\,(u_{t-1}^+)^2 + \frac{0.434048}{2(0.261142)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 0.140974 |
| n0 | 0.581006 |
| rho_p | 0.844859 |
| rho_n | 0.364141 |
| phi_p_plus | 0.306113 |
| phi_p_minus | 0.000012 |
| phi_n_plus | 0.746948 |
| phi_n_minus | 0.434048 |
| sigma_p | 0.048652 |
| sigma_n | 0.261142 |

### Rank 3: Seed 22, Draw 25

- LogLik: `-90.207296`; AIC: `200.414592`; BIC: `234.120972`
- Max shape path: `311.596535`; max implied variance: `2.881161`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.031390\,\omega_{p,t} - 0.172840\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.995125 + 0.892478\,p_{t-1} + \frac{0.038456}{2(0.031390)^2}\,(u_{t-1}^+)^2 + \frac{0.000002}{2(0.031390)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.536247 + 0.471176\,n_{t-1} + \frac{0.847206}{2(0.172840)^2}\,(u_{t-1}^+)^2 + \frac{0.161942}{2(0.172840)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 9.995125 |
| n0 | 0.536247 |
| rho_p | 0.892478 |
| rho_n | 0.471176 |
| phi_p_plus | 0.038456 |
| phi_p_minus | 0.000002 |
| phi_n_plus | 0.847206 |
| phi_n_minus | 0.161942 |
| sigma_p | 0.031390 |
| sigma_n | 0.172840 |

### Rank 4: Seed 5, Draw 25

- LogLik: `-92.043366`; AIC: `204.086731`; BIC: `237.793111`
- Max shape path: `2721.157655`; max implied variance: `2.792048`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.025280\,\omega_{p,t} - 0.090473\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.145144 + 0.404763\,p_{t-1} + \frac{0.604697}{2(0.025280)^2}\,(u_{t-1}^+)^2 + \frac{0.124047}{2(0.025280)^2}\,(u_{t-1}^-)^2,\\
n_t &= 2.575203 + 0.897088\,n_{t-1} + \frac{0.147715}{2(0.090473)^2}\,(u_{t-1}^+)^2 + \frac{0.005656}{2(0.090473)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 9.145144 |
| n0 | 2.575203 |
| rho_p | 0.404763 |
| rho_n | 0.897088 |
| phi_p_plus | 0.604697 |
| phi_p_minus | 0.124047 |
| phi_n_plus | 0.147715 |
| phi_n_minus | 0.005656 |
| sigma_p | 0.025280 |
| sigma_n | 0.090473 |

### Rank 5: Seed 15, Draw 32

- LogLik: `-102.920283`; AIC: `225.840566`; BIC: `259.546946`
- Max shape path: `638.183734`; max implied variance: `12.687129`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.030250\,\omega_{p,t} - 0.140995\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.075629 + 0.730210\,p_{t-1} + \frac{0.060665}{2(0.030250)^2}\,(u_{t-1}^+)^2 + \frac{0.000509}{2(0.030250)^2}\,(u_{t-1}^-)^2,\\
n_t &= 5.729896 + 0.385460\,n_{t-1} + \frac{1.185744}{2(0.140995)^2}\,(u_{t-1}^+)^2 + \frac{0.025378}{2(0.140995)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 0.075629 |
| n0 | 5.729896 |
| rho_p | 0.730210 |
| rho_n | 0.385460 |
| phi_p_plus | 0.060665 |
| phi_p_minus | 0.000509 |
| phi_n_plus | 1.185744 |
| phi_n_minus | 0.025378 |
| sigma_p | 0.030250 |
| sigma_n | 0.140995 |

## ARX(1,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 21 | 2 | -26.869083 | 79.738167 | 123.556461 | 5898.089507 | 3.686220 | yes |
| 2 | 50 | 28 | -67.411628 | 160.823255 | 204.641549 | 430.611824 | 3.165198 | yes |
| 3 | 4 | 38 | -71.555372 | 169.110743 | 212.929037 | 277.843395 | 4.234636 | yes |
| 4 | 41 | 5 | -74.562727 | 175.125454 | 218.943748 | 286.807713 | 2.029689 | yes |
| 5 | 10 | 11 | -75.633998 | 177.267996 | 221.086291 | 4739.724944 | 6.276822 | yes |

### Rank 1: Seed 21, Draw 2

- LogLik: `-26.869083`; AIC: `79.738167`; BIC: `123.556461`
- Max shape path: `5898.089507`; max implied variance: `3.686220`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.022180 + 0.168108\,\pi_t + 1.166054\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.018556\,\omega_{p,t} - 0.172765\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.168162 + 0.403220\,p_{t-1} + \frac{0.716565}{2(0.018556)^2}\,(u_{t-1}^+)^2 + \frac{0.221073}{2(0.018556)^2}\,(u_{t-1}^-)^2,\\
n_t &= 6.388150 + 0.734837\,n_{t-1} + \frac{0.234816}{2(0.172765)^2}\,(u_{t-1}^+)^2 + \frac{0.093477}{2(0.172765)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.022180 |
| rho_1 | 0.168108 |
| phi_1 | 1.166054 |
| p0 | 6.168162 |
| n0 | 6.388150 |
| rho_p | 0.403220 |
| rho_n | 0.734837 |
| phi_p_plus | 0.716565 |
| phi_p_minus | 0.221073 |
| phi_n_plus | 0.234816 |
| phi_n_minus | 0.093477 |
| sigma_p | 0.018556 |
| sigma_n | 0.172765 |

### Rank 2: Seed 50, Draw 28

- LogLik: `-67.411628`; AIC: `160.823255`; BIC: `204.641549`
- Max shape path: `430.611824`; max implied variance: `3.165198`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.037310 + 0.229352\,\pi_t + 0.826295\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.044675\,\omega_{p,t} - 0.198032\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 5.562551 + 0.845754\,p_{t-1} + \frac{0.248931}{2(0.044675)^2}\,(u_{t-1}^+)^2 + \frac{0.000058}{2(0.044675)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.364403 + 0.254848\,n_{t-1} + \frac{1.095792}{2(0.198032)^2}\,(u_{t-1}^+)^2 + \frac{0.330711}{2(0.198032)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.037310 |
| rho_1 | 0.229352 |
| phi_1 | 0.826295 |
| p0 | 5.562551 |
| n0 | 0.364403 |
| rho_p | 0.845754 |
| rho_n | 0.254848 |
| phi_p_plus | 0.248931 |
| phi_p_minus | 0.000058 |
| phi_n_plus | 1.095792 |
| phi_n_minus | 0.330711 |
| sigma_p | 0.044675 |
| sigma_n | 0.198032 |

### Rank 3: Seed 4, Draw 38

- LogLik: `-71.555372`; AIC: `169.110743`; BIC: `212.929037`
- Max shape path: `277.843395`; max implied variance: `4.234636`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.507740 + 0.175004\,\pi_t + 0.195945\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.039269\,\omega_{p,t} - 0.156981\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 4.340422 + 0.834358\,p_{t-1} + \frac{0.044745}{2(0.039269)^2}\,(u_{t-1}^+)^2 + \frac{0.001392}{2(0.039269)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.865892 + 0.318641\,n_{t-1} + \frac{0.865555}{2(0.156981)^2}\,(u_{t-1}^+)^2 + \frac{0.372185}{2(0.156981)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.507740 |
| rho_1 | 0.175004 |
| phi_1 | 0.195945 |
| p0 | 4.340422 |
| n0 | 0.865892 |
| rho_p | 0.834358 |
| rho_n | 0.318641 |
| phi_p_plus | 0.044745 |
| phi_p_minus | 0.001392 |
| phi_n_plus | 0.865555 |
| phi_n_minus | 0.372185 |
| sigma_p | 0.039269 |
| sigma_n | 0.156981 |

### Rank 4: Seed 41, Draw 5

- LogLik: `-74.562727`; AIC: `175.125454`; BIC: `218.943748`
- Max shape path: `286.807713`; max implied variance: `2.029689`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.043453 + 0.254010\,\pi_t + 0.723205\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.042148\,\omega_{p,t} - 0.156414\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.006826 + 0.931167\,p_{t-1} + \frac{0.081020}{2(0.042148)^2}\,(u_{t-1}^+)^2 + \frac{0.032743}{2(0.042148)^2}\,(u_{t-1}^-)^2,\\
n_t &= 1.076626 + 0.393064\,n_{t-1} + \frac{0.740828}{2(0.156414)^2}\,(u_{t-1}^+)^2 + \frac{0.089400}{2(0.156414)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.043453 |
| rho_1 | 0.254010 |
| phi_1 | 0.723205 |
| p0 | 1.006826 |
| n0 | 1.076626 |
| rho_p | 0.931167 |
| rho_n | 0.393064 |
| phi_p_plus | 0.081020 |
| phi_p_minus | 0.032743 |
| phi_n_plus | 0.740828 |
| phi_n_minus | 0.089400 |
| sigma_p | 0.042148 |
| sigma_n | 0.156414 |

### Rank 5: Seed 10, Draw 11

- LogLik: `-75.633998`; AIC: `177.267996`; BIC: `221.086291`
- Max shape path: `4739.724944`; max implied variance: `6.276822`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.266273 + 0.257758\,\pi_t + 1.179998\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.027389\,\omega_{p,t} - 0.192021\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.678423 + 0.446215\,p_{t-1} + \frac{0.497974}{2(0.027389)^2}\,(u_{t-1}^+)^2 + \frac{0.335539}{2(0.027389)^2}\,(u_{t-1}^-)^2,\\
n_t &= 2.764703 + 0.880577\,n_{t-1} + \frac{0.044749}{2(0.192021)^2}\,(u_{t-1}^+)^2 + \frac{0.158151}{2(0.192021)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.266273 |
| rho_1 | 0.257758 |
| phi_1 | 1.179998 |
| p0 | 0.678423 |
| n0 | 2.764703 |
| rho_p | 0.446215 |
| rho_n | 0.880577 |
| phi_p_plus | 0.497974 |
| phi_p_minus | 0.335539 |
| phi_n_plus | 0.044749 |
| phi_n_minus | 0.158151 |
| sigma_p | 0.027389 |
| sigma_n | 0.192021 |

## ARX(2,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 38 | 14 | -34.178772 | 96.357544 | 143.546476 | 2985.005187 | 5.417105 | yes |
| 2 | 49 | 17 | -41.426513 | 110.853025 | 158.041958 | 179.662537 | 11.302464 | yes |
| 3 | 43 | 5 | -58.453182 | 144.906365 | 192.095297 | 16259.129152 | 12.495088 | yes |
| 4 | 16 | 11 | -67.753408 | 163.506816 | 210.695748 | 356.139521 | 10.249964 | yes |
| 5 | 20 | 33 | -81.948089 | 191.896178 | 239.085110 | 321.567669 | 8.449473 | yes |

### Rank 1: Seed 38, Draw 14

- LogLik: `-34.178772`; AIC: `96.357544`; BIC: `143.546476`
- Max shape path: `2985.005187`; max implied variance: `5.417105`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.153688 + 0.276691\,\pi_t + 0.059441\,\pi_{t-1} + 0.721519\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.232353\,\omega_{p,t} - 0.029955\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 4.775107 + 0.869433\,p_{t-1} + \frac{0.031620}{2(0.232353)^2}\,(u_{t-1}^+)^2 + \frac{0.075491}{2(0.232353)^2}\,(u_{t-1}^-)^2,\\
n_t &= 8.255813 + 0.018547\,n_{t-1} + \frac{0.722367}{2(0.029955)^2}\,(u_{t-1}^+)^2 + \frac{0.289966}{2(0.029955)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.153688 |
| rho_1 | 0.276691 |
| rho_2 | 0.059441 |
| phi_1 | 0.721519 |
| p0 | 4.775107 |
| n0 | 8.255813 |
| rho_p | 0.869433 |
| rho_n | 0.018547 |
| phi_p_plus | 0.031620 |
| phi_p_minus | 0.075491 |
| phi_n_plus | 0.722367 |
| phi_n_minus | 0.289966 |
| sigma_p | 0.232353 |
| sigma_n | 0.029955 |

### Rank 2: Seed 49, Draw 17

- LogLik: `-41.426513`; AIC: `110.853025`; BIC: `158.041958`
- Max shape path: `179.662537`; max implied variance: `11.302464`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.063907 + 0.244734\,\pi_t + 0.300020\,\pi_{t-1} + 0.426482\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.317553\,\omega_{p,t} - 0.089397\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.630865 + 0.130807\,p_{t-1} + \frac{0.148960}{2(0.317553)^2}\,(u_{t-1}^+)^2 + \frac{0.984365}{2(0.317553)^2}\,(u_{t-1}^-)^2,\\
n_t &= 4.140626 + 0.551281\,n_{t-1} + \frac{0.435596}{2(0.089397)^2}\,(u_{t-1}^+)^2 + \frac{0.129471}{2(0.089397)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.063907 |
| rho_1 | 0.244734 |
| rho_2 | 0.300020 |
| phi_1 | 0.426482 |
| p0 | 0.630865 |
| n0 | 4.140626 |
| rho_p | 0.130807 |
| rho_n | 0.551281 |
| phi_p_plus | 0.148960 |
| phi_p_minus | 0.984365 |
| phi_n_plus | 0.435596 |
| phi_n_minus | 0.129471 |
| sigma_p | 0.317553 |
| sigma_n | 0.089397 |

### Rank 3: Seed 43, Draw 5

- LogLik: `-58.453182`; AIC: `144.906365`; BIC: `192.095297`
- Max shape path: `16259.129152`; max implied variance: `12.495088`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.154859 + 0.479165\,\pi_t + 0.072775\,\pi_{t-1} + 0.282024\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.086016\,\omega_{p,t} - 0.026151\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 7.827147 + 0.343571\,p_{t-1} + \frac{0.871722}{2(0.086016)^2}\,(u_{t-1}^+)^2 + \frac{0.139319}{2(0.086016)^2}\,(u_{t-1}^-)^2,\\
n_t &= 7.770777 + 0.149366\,n_{t-1} + \frac{0.242042}{2(0.026151)^2}\,(u_{t-1}^+)^2 + \frac{1.262432}{2(0.026151)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.154859 |
| rho_1 | 0.479165 |
| rho_2 | 0.072775 |
| phi_1 | 0.282024 |
| p0 | 7.827147 |
| n0 | 7.770777 |
| rho_p | 0.343571 |
| rho_n | 0.149366 |
| phi_p_plus | 0.871722 |
| phi_p_minus | 0.139319 |
| phi_n_plus | 0.242042 |
| phi_n_minus | 1.262432 |
| sigma_p | 0.086016 |
| sigma_n | 0.026151 |

### Rank 4: Seed 16, Draw 11

- LogLik: `-67.753408`; AIC: `163.506816`; BIC: `210.695748`
- Max shape path: `356.139521`; max implied variance: `10.249964`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.124004 + 0.331481\,\pi_t + 0.093285\,\pi_{t-1} + 0.744715\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.046725\,\omega_{p,t} - 0.434081\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 7.390186 + 0.940718\,p_{t-1} + \frac{0.067441}{2(0.046725)^2}\,(u_{t-1}^+)^2 + \frac{0.009622}{2(0.046725)^2}\,(u_{t-1}^-)^2,\\
n_t &= 3.366784 + 0.886815\,n_{t-1} + \frac{0.037033}{2(0.434081)^2}\,(u_{t-1}^+)^2 + \frac{0.055391}{2(0.434081)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.124004 |
| rho_1 | 0.331481 |
| rho_2 | 0.093285 |
| phi_1 | 0.744715 |
| p0 | 7.390186 |
| n0 | 3.366784 |
| rho_p | 0.940718 |
| rho_n | 0.886815 |
| phi_p_plus | 0.067441 |
| phi_p_minus | 0.009622 |
| phi_n_plus | 0.037033 |
| phi_n_minus | 0.055391 |
| sigma_p | 0.046725 |
| sigma_n | 0.434081 |

### Rank 5: Seed 20, Draw 33

- LogLik: `-81.948089`; AIC: `191.896178`; BIC: `239.085110`
- Max shape path: `321.567669`; max implied variance: `8.449473`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.071501 + 0.323503\,\pi_t + -0.033146\,\pi_{t-1} + 0.672888\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.416666\,\omega_{p,t} - 0.072792\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.490167 + 0.011027\,p_{t-1} + \frac{0.254305}{2(0.416666)^2}\,(u_{t-1}^+)^2 + \frac{0.924932}{2(0.416666)^2}\,(u_{t-1}^-)^2,\\
n_t &= 3.538118 + 0.540642\,n_{t-1} + \frac{0.719493}{2(0.072792)^2}\,(u_{t-1}^+)^2 + \frac{0.075552}{2(0.072792)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.071501 |
| rho_1 | 0.323503 |
| rho_2 | -0.033146 |
| phi_1 | 0.672888 |
| p0 | 0.490167 |
| n0 | 3.538118 |
| rho_p | 0.011027 |
| rho_n | 0.540642 |
| phi_p_plus | 0.254305 |
| phi_p_minus | 0.924932 |
| phi_n_plus | 0.719493 |
| phi_n_minus | 0.075552 |
| sigma_p | 0.416666 |
| sigma_n | 0.072792 |

## ARX(2,2)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 16 | 1 | 41.196715 | -52.393430 | -1.833859 | 410.214941 | 30.817130 | yes |
| 2 | 35 | 33 | -40.726117 | 111.452235 | 162.011805 | 190.820496 | 12.919577 | yes |
| 3 | 32 | 35 | -56.969410 | 143.938821 | 194.498391 | 217.922111 | 12.060259 | yes |
| 4 | 30 | 3 | -62.989225 | 155.978449 | 206.538020 | 3634.909745 | 9.092887 | yes |
| 5 | 13 | 9 | -69.859170 | 169.718341 | 220.277911 | 1187.874141 | 128.229298 | yes |

### Rank 1: Seed 16, Draw 1

- LogLik: `41.196715`; AIC: `-52.393430`; BIC: `-1.833859`
- Max shape path: `410.214941`; max implied variance: `30.817130`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.053818 + 0.227540\,\pi_t + 0.007305\,\pi_{t-1} + 0.779104\,SPF_t + 0.212611\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.046637\,\omega_{p,t} - 0.273377\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 5.158554 + 0.906100\,p_{t-1} + \frac{0.010131}{2(0.046637)^2}\,(u_{t-1}^+)^2 + \frac{0.037255}{2(0.046637)^2}\,(u_{t-1}^-)^2,\\
n_t &= 6.225894 + 0.830220\,n_{t-1} + \frac{0.114966}{2(0.273377)^2}\,(u_{t-1}^+)^2 + \frac{0.194239}{2(0.273377)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.053818 |
| rho_1 | 0.227540 |
| rho_2 | 0.007305 |
| phi_1 | 0.779104 |
| phi_2 | 0.212611 |
| p0 | 5.158554 |
| n0 | 6.225894 |
| rho_p | 0.906100 |
| rho_n | 0.830220 |
| phi_p_plus | 0.010131 |
| phi_p_minus | 0.037255 |
| phi_n_plus | 0.114966 |
| phi_n_minus | 0.194239 |
| sigma_p | 0.046637 |
| sigma_n | 0.273377 |

### Rank 2: Seed 35, Draw 33

- LogLik: `-40.726117`; AIC: `111.452235`; BIC: `162.011805`
- Max shape path: `190.820496`; max implied variance: `12.919577`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.067941 + 0.378207\,\pi_t + 0.085487\,\pi_{t-1} + 1.210219\,SPF_t + 0.816326\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.075710\,\omega_{p,t} - 0.329627\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.353878 + 0.887637\,p_{t-1} + \frac{0.006194}{2(0.075710)^2}\,(u_{t-1}^+)^2 + \frac{0.020211}{2(0.075710)^2}\,(u_{t-1}^-)^2,\\
n_t &= 4.852463 + 0.873405\,n_{t-1} + \frac{0.108534}{2(0.329627)^2}\,(u_{t-1}^+)^2 + \frac{0.061793}{2(0.329627)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.067941 |
| rho_1 | 0.378207 |
| rho_2 | 0.085487 |
| phi_1 | 1.210219 |
| phi_2 | 0.816326 |
| p0 | 3.353878 |
| n0 | 4.852463 |
| rho_p | 0.887637 |
| rho_n | 0.873405 |
| phi_p_plus | 0.006194 |
| phi_p_minus | 0.020211 |
| phi_n_plus | 0.108534 |
| phi_n_minus | 0.061793 |
| sigma_p | 0.075710 |
| sigma_n | 0.329627 |

### Rank 3: Seed 32, Draw 35

- LogLik: `-56.969410`; AIC: `143.938821`; BIC: `194.498391`
- Max shape path: `217.922111`; max implied variance: `12.060259`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.097252 + 0.287809\,\pi_t + 0.169059\,\pi_{t-1} + 0.226085\,SPF_t + 0.276628\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.073715\,\omega_{p,t} - 0.496103\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 5.694250 + 0.593508\,p_{t-1} + \frac{0.506733}{2(0.073715)^2}\,(u_{t-1}^+)^2 + \frac{0.077264}{2(0.073715)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.244397 + 0.006813\,n_{t-1} + \frac{0.285996}{2(0.496103)^2}\,(u_{t-1}^+)^2 + \frac{1.190769}{2(0.496103)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.097252 |
| rho_1 | 0.287809 |
| rho_2 | 0.169059 |
| phi_1 | 0.226085 |
| phi_2 | 0.276628 |
| p0 | 5.694250 |
| n0 | 0.244397 |
| rho_p | 0.593508 |
| rho_n | 0.006813 |
| phi_p_plus | 0.506733 |
| phi_p_minus | 0.077264 |
| phi_n_plus | 0.285996 |
| phi_n_minus | 1.190769 |
| sigma_p | 0.073715 |
| sigma_n | 0.496103 |

### Rank 4: Seed 30, Draw 3

- LogLik: `-62.989225`; AIC: `155.978449`; BIC: `206.538020`
- Max shape path: `3634.909745`; max implied variance: `9.092887`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.458494 + -0.187749\,\pi_t + 0.023497\,\pi_{t-1} + -0.244087\,SPF_t + 1.385269\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.190439\,\omega_{p,t} - 0.046954\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.035957 + 0.373410\,p_{t-1} + \frac{1.031564}{2(0.190439)^2}\,(u_{t-1}^+)^2 + \frac{0.084976}{2(0.190439)^2}\,(u_{t-1}^-)^2,\\
n_t &= 6.982887 + 0.204765\,n_{t-1} + \frac{0.178274}{2(0.046954)^2}\,(u_{t-1}^+)^2 + \frac{0.800888}{2(0.046954)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.458494 |
| rho_1 | -0.187749 |
| rho_2 | 0.023497 |
| phi_1 | -0.244087 |
| phi_2 | 1.385269 |
| p0 | 3.035957 |
| n0 | 6.982887 |
| rho_p | 0.373410 |
| rho_n | 0.204765 |
| phi_p_plus | 1.031564 |
| phi_p_minus | 0.084976 |
| phi_n_plus | 0.178274 |
| phi_n_minus | 0.800888 |
| sigma_p | 0.190439 |
| sigma_n | 0.046954 |

### Rank 5: Seed 13, Draw 9

- LogLik: `-69.859170`; AIC: `169.718341`; BIC: `220.277911`
- Max shape path: `1187.874141`; max implied variance: `128.229298`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.084565 + 0.339129\,\pi_t + -0.009413\,\pi_{t-1} + 0.682671\,SPF_t + -0.212126\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.044647\,\omega_{p,t} - 0.328365\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 5.834275 + 0.833422\,p_{t-1} + \frac{0.082322}{2(0.044647)^2}\,(u_{t-1}^+)^2 + \frac{0.094527}{2(0.044647)^2}\,(u_{t-1}^-)^2,\\
n_t &= 6.618129 + 0.803619\,n_{t-1} + \frac{0.179316}{2(0.328365)^2}\,(u_{t-1}^+)^2 + \frac{0.202302}{2(0.328365)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.084565 |
| rho_1 | 0.339129 |
| rho_2 | -0.009413 |
| phi_1 | 0.682671 |
| phi_2 | -0.212126 |
| p0 | 5.834275 |
| n0 | 6.618129 |
| rho_p | 0.833422 |
| rho_n | 0.803619 |
| phi_p_plus | 0.082322 |
| phi_p_minus | 0.094527 |
| phi_n_plus | 0.179316 |
| phi_n_minus | 0.202302 |
| sigma_p | 0.044647 |
| sigma_n | 0.328365 |
