```{raw:typst}
#set page(margin: auto)
```

# Full BEGE Best Model Summary

Generated: `2026-05-31T16:29:30`
Total estimations: `4004`
Converged estimations: `4004`
Eligible estimations for best-model selection: `4004`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, and documented parameter/stability/unconditional-variance constraints.

## Global Best by AIC

- Mean type: `ARX(2,2)`
- Seed / draw: `7` / `8`
- AIC: `-6.730114`
- BIC: `43.829457`
- LogLik: `18.365057`
- Max shape: `745.568795`

## Global Best by BIC

- Mean type: `ARX(2,2)`
- Seed / draw: `7` / `8`
- AIC: `-6.730114`
- BIC: `43.829457`
- LogLik: `18.365057`
- Max shape: `745.568795`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 90 | 10 | 107.459755 | 141.166135 | -43.729878 | 360.905485 | 0.073864 |
| ARX(1,1) | 3 | 10 | 275.192039 | 319.010334 | -124.596020 | 226.890218 | 0.078104 |
| ARX(2,1) | 97 | 1 | 196.337067 | 243.525999 | -84.168534 | 187.521810 | 0.096691 |
| ARX(2,2) | 7 | 8 | -6.730114 | 43.829457 | 18.365057 | 745.568795 | 0.053275 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 90 | 10 | 107.459755 | 141.166135 | -43.729878 | 360.905485 | 0.073864 |
| ARX(1,1) | 3 | 10 | 275.192039 | 319.010334 | -124.596020 | 226.890218 | 0.078104 |
| ARX(2,1) | 97 | 1 | 196.337067 | 243.525999 | -84.168534 | 187.521810 | 0.096691 |
| ARX(2,2) | 7 | 8 | -6.730114 | 43.829457 | 18.365057 | 745.568795 | 0.053275 |

## Parameter Estimates From Best AIC Fits

### constant

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 1.020568 | 642.603477 |
| n0 | 2.629179 | 1930.516048 |
| rho_p | 0.000140 | 467.719148 |
| rho_n | 0.284566 | 22.302143 |
| phi_p_plus | 0.060659 | 960.348389 |
| phi_p_minus | 0.999000 | 12.869956 |
| phi_n_plus | 0.732789 | 227.103516 |
| phi_n_minus | 0.118977 | 0.000416 |
| sigma_p | 0.299363 | 0.000409 |
| sigma_n | 0.073864 | 0.000412 |

### ARX(1,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.108817 | 3.723452 |
| rho_1 | 0.401823 | 8.700547 |
| phi_1 | 0.647492 | 16.436002 |
| p0 | 6.027018 | 7.410728 |
| n0 | 4.401859 | 5.342318 |
| rho_p | 0.222627 | 0.538293 |
| rho_n | 0.230459 | 0.157217 |
| phi_p_plus | 0.512699 | 0.211133 |
| phi_p_minus | 0.112869 | 0.000041 |
| phi_n_plus | 0.127898 | 3.001225 |
| phi_n_minus | 0.769526 | 2.173598 |
| sigma_p | 0.078104 | 0.000000 |
| sigma_n | 0.255866 | 0.386044 |

### ARX(2,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.112878 | 0.000491 |
| rho_1 | 0.219899 | 0.472785 |
| rho_2 | 0.123548 | 0.000479 |
| phi_1 | 0.569001 | 0.473256 |
| p0 | 1.214025 | 3309.782984 |
| n0 | 3.434311 | 1434.096076 |
| rho_p | 0.252028 | 1434.100350 |
| rho_n | 0.552227 | 0.000000 |
| phi_p_plus | 0.267015 | 9411.917053 |
| phi_p_minus | 0.822841 | 0.000475 |
| phi_n_plus | 0.272032 | 0.000000 |
| phi_n_minus | 0.179901 | 0.000497 |
| sigma_p | 0.201014 | 0.000473 |
| sigma_n | 0.096691 | 0.000528 |

### ARX(2,2)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.017255 | 60.223484 |
| rho_1 | 0.001986 | 15.600399 |
| rho_2 | 0.059976 | 32.980582 |
| phi_1 | 0.213953 | 116.842533 |
| phi_2 | 1.913764 | 105.692778 |
| p0 | 5.608108 | 3927.875724 |
| n0 | 6.078626 | 2153.442460 |
| rho_p | 0.228366 | 3.189194 |
| rho_n | 0.636047 | 40.359291 |
| phi_p_plus | 0.324595 | 2413.806029 |
| phi_p_minus | 0.183621 | 5.177028 |
| phi_n_plus | 0.063376 | 79.566406 |
| phi_n_minus | 0.505966 | 0.321695 |
| sigma_p | 0.053275 | 0.001003 |
| sigma_n | 0.296416 | 3.459597 |
