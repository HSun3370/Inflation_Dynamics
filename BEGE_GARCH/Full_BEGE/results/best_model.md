```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-06-02T10:03:14`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7895`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are treated as implausible and excluded.

```{warning}
Excluded 104 estimate(s) with corrected log likelihood above `-150`.
```

## Global Best by AIC

- Mean type: `constant`
- Seed / draw: `3` / `1`
- AIC: `326.833506`
- BIC: `360.539886`
- LogLik: `-153.416753`
- Max shape: `535.161718`

## Global Best by BIC

- Mean type: `constant`
- Seed / draw: `3` / `1`
- AIC: `326.833506`
- BIC: `360.539886`
- LogLik: `-153.416753`
- Max shape: `535.161718`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 3 | 1 | 326.833506 | 360.539886 | -153.416753 | 535.161718 | 0.074248 |
| ARX(1,1) | 35 | 11 | 331.695671 | 375.513965 | -152.847835 | 6496.083791 | 0.031822 |
| ARX(2,1) | 50 | 6 | 328.579887 | 375.768819 | -150.289943 | 16692.715206 | 0.015083 |
| ARX(2,2) | 25 | 26 | 331.724617 | 382.284187 | -150.862308 | 5089.377463 | 0.029278 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 3 | 1 | 326.833506 | 360.539886 | -153.416753 | 535.161718 | 0.074248 |
| ARX(1,1) | 35 | 11 | 331.695671 | 375.513965 | -152.847835 | 6496.083791 | 0.031822 |
| ARX(2,1) | 50 | 6 | 328.579887 | 375.768819 | -150.289943 | 16692.715206 | 0.015083 |
| ARX(2,2) | 25 | 26 | 331.724617 | 382.284187 | -150.862308 | 5089.377463 | 0.029278 |

## Parameter Estimates From Best AIC Fits

### constant

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.464501 | 37590.490564 |
| n0 | 8.552625 | 17625.573573 |
| rho_p | 0.037769 | 4070.170037 |
| rho_n | 0.418121 | 0.328423 |
| phi_p_plus | 0.134657 | 44375.384068 |
| phi_p_minus | 0.555989 | 0.327991 |
| phi_n_plus | 0.994069 | 0.328897 |
| phi_n_minus | 0.048366 | 0.330186 |
| sigma_p | 0.318520 | 0.327834 |
| sigma_n | 0.074248 | 0.332946 |

### ARX(1,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.013398 | 0.043291 |
| rho_1 | 0.259900 | 0.029896 |
| phi_1 | 0.617780 | 0.019969 |
| p0 | 8.699231 | 2.527833 |
| n0 | 2.338771 | 0.044368 |
| rho_p | 0.320819 | 0.000000 |
| rho_n | 0.777366 | 0.001973 |
| phi_p_plus | 0.503152 | 0.001896 |
| phi_p_minus | 0.827996 | 0.094628 |
| phi_n_plus | 0.267358 | 0.043251 |
| phi_n_minus | 0.006907 | 0.259838 |
| sigma_p | 0.031822 | 0.000001 |
| sigma_n | 0.086111 | 0.000002 |

### ARX(2,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.090350 | 53.247340 |
| rho_1 | 0.259204 | 119.947697 |
| rho_2 | 0.183225 | 100.195288 |
| phi_1 | 0.677700 | 220.152216 |
| p0 | 7.101834 | 395.008027 |
| n0 | 4.736314 | 491.383449 |
| rho_p | 0.213334 | 7.423446 |
| rho_n | 0.461366 | 17.040112 |
| phi_p_plus | 0.154040 | 0.916699 |
| phi_p_minus | 0.420709 | 33.453530 |
| phi_n_plus | 0.677840 | 360.503090 |
| phi_n_minus | 0.038529 | 458.620466 |
| sigma_p | 0.015083 | 0.000009 |
| sigma_n | 0.187338 | 0.428736 |

### ARX(2,2)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.028382 | 0.519774 |
| rho_1 | 0.365045 | 0.312706 |
| rho_2 | 0.252178 | 0.408417 |
| phi_1 | -0.056993 | 4.590155 |
| phi_2 | 0.374529 | 4.104036 |
| p0 | 8.555608 | 6.307115 |
| n0 | 4.528795 | 1.281725 |
| rho_p | 0.373223 | 0.051952 |
| rho_n | 0.500570 | 0.056816 |
| phi_p_plus | 0.401637 | 0.008126 |
| phi_p_minus | 0.456211 | 0.086074 |
| phi_n_plus | 0.428306 | 0.226887 |
| phi_n_minus | 0.208402 | 0.301885 |
| sigma_p | 0.029278 | 0.000010 |
| sigma_n | 0.134996 | 0.018681 |
