```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-05-31T10:14:49`
Total estimations: `4020`
Converged estimations: `3917`
Eligible estimations for best-model selection: `3886`

Selection screen: finite AIC/BIC/log-likelihood, successful optimizer status, and `max(p_t, n_t) < 200`.

## Global Best by AIC

- Mean type: `constant`
- Seed / draw: `100` / `2`
- AIC: `187.972717`
- BIC: `214.937821`
- LogLik: `-85.986358`
- Max shape: `186.829178`

## Global Best by BIC

- Mean type: `constant`
- Seed / draw: `100` / `2`
- AIC: `187.972717`
- BIC: `214.937821`
- LogLik: `-85.986358`
- Max shape: `186.829178`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 100 | 2 | 187.972717 | 214.937821 | -85.986358 | 186.829178 | 0.072113 |
| ARX(1,1) | 72 | 8 | 284.980641 | 322.057659 | -131.490320 | 176.720681 | 0.046657 |
| ARX(2,1) | 51 | 8 | 279.139340 | 319.586996 | -127.569670 | 197.372013 | 0.131779 |
| ARX(2,2) | 34 | 2 | 291.884502 | 335.702796 | -132.942251 | 146.783773 | 0.073678 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 100 | 2 | 187.972717 | 214.937821 | -85.986358 | 186.829178 | 0.072113 |
| ARX(1,1) | 72 | 8 | 284.980641 | 322.057659 | -131.490320 | 176.720681 | 0.046657 |
| ARX(2,1) | 51 | 8 | 279.139340 | 319.586996 | -127.569670 | 197.372013 | 0.131779 |
| ARX(2,2) | 34 | 2 | 291.884502 | 335.702796 | -132.942251 | 146.783773 | 0.073678 |

## Parameter Estimates From Eligible Best AIC Fits

### constant

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 5.671685 | NA |
| n0 | 0.005003 | NA |
| rho_p | 0.861405 | NA |
| rho_n | 0.117878 | NA |
| phi_p | 0.079471 | NA |
| phi_n | 0.881953 | NA |
| sigma_p | 0.072113 | NA |
| sigma_n | 0.287229 | NA |

### ARX(1,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.103599 | NA |
| rho_1 | 0.382466 | NA |
| phi_1 | 0.997476 | NA |
| p0 | 1.370311 | NA |
| n0 | 3.788525 | NA |
| rho_p | 0.799942 | NA |
| rho_n | 0.596036 | NA |
| phi_p | 0.038658 | NA |
| phi_n | 0.371670 | NA |
| sigma_p | 0.046657 | NA |
| sigma_n | 0.240144 | NA |

### ARX(2,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.095943 | NA |
| rho_1 | 0.259777 | NA |
| rho_2 | -0.017626 | NA |
| phi_1 | 0.797796 | NA |
| p0 | 1.340577 | NA |
| n0 | 6.667036 | NA |
| rho_p | 0.734316 | NA |
| rho_n | 0.905106 | NA |
| phi_p | 0.252627 | NA |
| phi_n | 0.061115 | NA |
| sigma_p | 0.510072 | NA |
| sigma_n | 0.131779 | NA |

### ARX(2,2)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.057000 | NA |
| rho_1 | 0.373407 | NA |
| rho_2 | 0.188337 | NA |
| phi_1 | -0.330196 | NA |
| phi_2 | 0.745016 | NA |
| p0 | 5.273562 | NA |
| n0 | 3.478176 | NA |
| rho_p | 0.727260 | NA |
| rho_n | 0.080459 | NA |
| phi_p | 0.070550 | NA |
| phi_n | 0.430884 | NA |
| sigma_p | 0.073678 | NA |
| sigma_n | 0.183330 | NA |
