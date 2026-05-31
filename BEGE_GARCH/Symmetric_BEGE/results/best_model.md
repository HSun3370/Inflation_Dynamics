```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-05-31T10:14:49`
Total estimations: `4004`
Converged estimations: `4004`
Eligible estimations for best-model selection: `3978`

Selection screen: finite AIC/BIC/log-likelihood, successful optimizer status, and `max(p_t, n_t) < 200`.

## Global Best by AIC

- Mean type: `ARX(2,1)`
- Seed / draw: `39` / `8`
- AIC: `358.770737`
- BIC: `395.847756`
- LogLik: `-168.385369`
- Max shape: `10.561242`

## Global Best by BIC

- Mean type: `constant`
- Seed / draw: `19` / `7`
- AIC: `366.246483`
- BIC: `389.840949`
- LogLik: `-176.123241`
- Max shape: `112.717098`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 19 | 7 | 366.246483 | 389.840949 | -176.123241 | 112.717098 | 0.139871 |
| ARX(1,1) | 75 | 7 | 360.128702 | 393.835082 | -170.064351 | 39.977562 | 0.184266 |
| ARX(2,1) | 39 | 8 | 358.770737 | 395.847756 | -168.385369 | 10.561242 | 0.361281 |
| ARX(2,2) | 59 | 1 | 361.653630 | 402.101286 | -168.826815 | 35.324447 | 0.177234 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 19 | 7 | 366.246483 | 389.840949 | -176.123241 | 112.717098 | 0.139871 |
| ARX(1,1) | 75 | 7 | 360.128702 | 393.835082 | -170.064351 | 39.977562 | 0.184266 |
| ARX(2,1) | 39 | 8 | 358.770737 | 395.847756 | -168.385369 | 10.561242 | 0.361281 |
| ARX(2,2) | 59 | 1 | 361.653630 | 402.101286 | -168.826815 | 35.324447 | 0.177234 |

## Parameter Estimates From Eligible Best AIC Fits

### constant

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 2.877574 | NA |
| n0 | 0.035896 | NA |
| rho | 0.423340 | NA |
| phi_plus | 0.727509 | NA |
| phi_minus | 0.134980 | NA |
| sigma_p | 0.139871 | NA |
| sigma_n | 1.496023 | NA |

### ARX(1,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.127718 | NA |
| rho_1 | 0.226108 | NA |
| phi_1 | 0.713765 | NA |
| p0 | 1.685365 | NA |
| n0 | 0.050587 | NA |
| rho | 0.518141 | NA |
| phi_plus | 0.557483 | NA |
| phi_minus | 0.044598 | NA |
| sigma_p | 0.184266 | NA |
| sigma_n | 0.646891 | NA |

### ARX(2,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.212112 | NA |
| rho_1 | 0.191574 | NA |
| rho_2 | 0.248838 | NA |
| phi_1 | 0.308879 | NA |
| p0 | 0.195926 | NA |
| n0 | 0.109561 | NA |
| rho | 0.627380 | NA |
| phi_plus | 0.468479 | NA |
| phi_minus | 0.096435 | NA |
| sigma_p | 0.361281 | NA |
| sigma_n | 0.482695 | NA |

### ARX(2,2)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.118775 | NA |
| rho_1 | 0.222018 | NA |
| rho_2 | 0.119321 | NA |
| phi_1 | 0.458211 | NA |
| phi_2 | 0.136363 | NA |
| p0 | 1.489645 | NA |
| n0 | 0.040169 | NA |
| rho | 0.583471 | NA |
| phi_plus | 0.465367 | NA |
| phi_minus | 0.049596 | NA |
| sigma_p | 0.177234 | NA |
| sigma_n | 0.730911 | NA |
