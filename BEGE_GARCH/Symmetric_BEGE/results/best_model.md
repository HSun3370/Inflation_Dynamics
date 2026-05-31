```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-05-31T16:32:31`
Total estimations: `4004`
Converged estimations: `4004`
Eligible estimations for best-model selection: `4004`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, and documented parameter/stability/unconditional-variance constraints.

## Global Best by AIC

- Mean type: `ARX(2,1)`
- Seed / draw: `33` / `3`
- AIC: `336.821887`
- BIC: `373.898906`
- LogLik: `-157.410944`
- Max shape: `321.028674`

## Global Best by BIC

- Mean type: `constant`
- Seed / draw: `80` / `4`
- AIC: `348.225798`
- BIC: `371.820264`
- LogLik: `-167.112899`
- Max shape: `509.938159`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 80 | 4 | 348.225798 | 371.820264 | -167.112899 | 509.938159 | 0.046945 |
| ARX(1,1) | 75 | 7 | 360.128702 | 393.835082 | -170.064351 | 39.977562 | 0.184266 |
| ARX(2,1) | 33 | 3 | 336.821887 | 373.898906 | -157.410944 | 321.028674 | 0.055808 |
| ARX(2,2) | 59 | 1 | 361.653630 | 402.101286 | -168.826815 | 35.324447 | 0.177234 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 80 | 4 | 348.225798 | 371.820264 | -167.112899 | 509.938159 | 0.046945 |
| ARX(1,1) | 75 | 7 | 360.128702 | 393.835082 | -170.064351 | 39.977562 | 0.184266 |
| ARX(2,1) | 33 | 3 | 336.821887 | 373.898906 | -157.410944 | 321.028674 | 0.055808 |
| ARX(2,2) | 59 | 1 | 361.653630 | 402.101286 | -168.826815 | 35.324447 | 0.177234 |

## Parameter Estimates From Best AIC Fits

### constant

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 6.009042 | 1944.188830 |
| n0 | 6.566534 | 1049.350091 |
| rho | 0.632466 | 0.141303 |
| phi_plus | 0.301054 | 16.107963 |
| phi_minus | 0.118481 | 46.852710 |
| sigma_p | 0.046945 | 0.000392 |
| sigma_n | 0.063736 | 0.000816 |

### ARX(1,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.127718 | 0.084455 |
| rho_1 | 0.226108 | 0.081926 |
| phi_1 | 0.713765 | 0.117150 |
| p0 | 1.685365 | 0.492642 |
| n0 | 0.050587 | 0.049623 |
| rho | 0.518141 | 0.159624 |
| phi_plus | 0.557483 | 0.193902 |
| phi_minus | 0.044598 | 0.079540 |
| sigma_p | 0.184266 | 0.049650 |
| sigma_n | 0.646891 | 0.375694 |

### ARX(2,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.146472 | 0.055409 |
| rho_1 | 0.538570 | 0.012069 |
| rho_2 | 0.203855 | 0.005827 |
| phi_1 | -0.192917 | 0.009124 |
| p0 | 1.761364 | 0.004706 |
| n0 | 6.492780 | 0.482157 |
| rho | 0.908778 | 0.000005 |
| phi_plus | 0.051370 | 0.000009 |
| phi_minus | 0.063027 | 0.002754 |
| sigma_p | 0.663130 | 0.007542 |
| sigma_n | 0.055808 | 0.000005 |

### ARX(2,2)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.118775 | 0.099447 |
| rho_1 | 0.222018 | 0.085452 |
| rho_2 | 0.119321 | 0.086342 |
| phi_1 | 0.458211 | 0.314800 |
| phi_2 | 0.136363 | 0.282144 |
| p0 | 1.489645 | 1.153155 |
| n0 | 0.040169 | 0.039032 |
| rho | 0.583471 | 0.268069 |
| phi_plus | 0.465367 | 0.303378 |
| phi_minus | 0.049596 | 0.090857 |
| sigma_p | 0.177234 | 0.056448 |
| sigma_n | 0.730911 | 0.395504 |
