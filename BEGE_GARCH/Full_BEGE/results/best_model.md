```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-05-31T10:14:49`
Total estimations: `4004`
Converged estimations: `4004`
Eligible estimations for best-model selection: `3983`

Selection screen: finite AIC/BIC/log-likelihood, successful optimizer status, and `max(p_t, n_t) < 200`.

## Global Best by AIC

- Mean type: `ARX(2,1)`
- Seed / draw: `97` / `1`
- AIC: `196.337067`
- BIC: `243.525999`
- LogLik: `-84.168534`
- Max shape: `187.521810`

## Global Best by BIC

- Mean type: `ARX(2,1)`
- Seed / draw: `97` / `1`
- AIC: `196.337067`
- BIC: `243.525999`
- LogLik: `-84.168534`
- Max shape: `187.521810`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 13 | 8 | 356.665194 | 390.371574 | -168.332597 | 150.961940 | 0.065994 |
| ARX(1,1) | 77 | 7 | 299.071118 | 342.889413 | -136.535559 | 187.845399 | 0.033859 |
| ARX(2,1) | 97 | 1 | 196.337067 | 243.525999 | -84.168534 | 187.521810 | 0.096691 |
| ARX(2,2) | 79 | 8 | 261.512546 | 312.072116 | -115.756273 | 174.496722 | 0.087963 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 13 | 8 | 356.665194 | 390.371574 | -168.332597 | 150.961940 | 0.065994 |
| ARX(1,1) | 77 | 7 | 299.071118 | 342.889413 | -136.535559 | 187.845399 | 0.033859 |
| ARX(2,1) | 97 | 1 | 196.337067 | 243.525999 | -84.168534 | 187.521810 | 0.096691 |
| ARX(2,2) | 79 | 8 | 261.512546 | 312.072116 | -115.756273 | 174.496722 | 0.087963 |

## Parameter Estimates From Eligible Best AIC Fits

### constant

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 5.725354 | NA |
| n0 | 1.619786 | NA |
| rho_p | 0.722368 | NA |
| rho_n | 0.408315 | NA |
| phi_p_plus | 0.145191 | NA |
| phi_p_minus | 0.005300 | NA |
| phi_n_plus | 0.929807 | NA |
| phi_n_minus | 0.114666 | NA |
| sigma_p | 0.065994 | NA |
| sigma_n | 0.136878 | NA |

### ARX(1,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.057023 | NA |
| rho_1 | 0.291693 | NA |
| phi_1 | 0.731579 | NA |
| p0 | 3.485764 | NA |
| n0 | 0.995942 | NA |
| rho_p | 0.047565 | NA |
| rho_n | 0.969135 | NA |
| phi_p_plus | 0.833180 | NA |
| phi_p_minus | 0.011776 | NA |
| phi_n_plus | 0.027313 | NA |
| phi_n_minus | 0.006666 | NA |
| sigma_p | 0.164450 | NA |
| sigma_n | 0.033859 | NA |

### ARX(2,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.112878 | NA |
| rho_1 | 0.219899 | NA |
| rho_2 | 0.123548 | NA |
| phi_1 | 0.569001 | NA |
| p0 | 1.214025 | NA |
| n0 | 3.434311 | NA |
| rho_p | 0.252028 | NA |
| rho_n | 0.552227 | NA |
| phi_p_plus | 0.267015 | NA |
| phi_p_minus | 0.822841 | NA |
| phi_n_plus | 0.272032 | NA |
| phi_n_minus | 0.179901 | NA |
| sigma_p | 0.201014 | NA |
| sigma_n | 0.096691 | NA |

### ARX(2,2)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.131670 | NA |
| rho_1 | 0.135261 | NA |
| rho_2 | 0.076096 | NA |
| phi_1 | 1.026076 | NA |
| phi_2 | -0.248071 | NA |
| p0 | 0.054830 | NA |
| n0 | 6.010088 | NA |
| rho_p | 0.114365 | NA |
| rho_n | 0.665189 | NA |
| phi_p_plus | 0.502739 | NA |
| phi_p_minus | 0.825129 | NA |
| phi_n_plus | 0.314290 | NA |
| phi_n_minus | 0.122241 | NA |
| sigma_p | 0.268209 | NA |
| sigma_n | 0.087963 | NA |
