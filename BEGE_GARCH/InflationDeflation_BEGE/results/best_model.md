```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-05-31T16:27:33`
Total estimations: `702`
Converged estimations: `73`
Eligible estimations for best-model selection: `7`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, and documented parameter/stability/unconditional-variance constraints.

## Global Best by AIC

- Mean type: `constant`
- Seed / draw: `54` / `2`
- AIC: `446.798303`
- BIC: `473.763407`
- LogLik: `-215.399151`
- Max shape: `62.877897`

## Global Best by BIC

- Mean type: `constant`
- Seed / draw: `54` / `2`
- AIC: `446.798303`
- BIC: `473.763407`
- LogLik: `-215.399151`
- Max shape: `62.877897`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 54 | 2 | 446.798303 | 473.763407 | -215.399151 | 62.877897 | 0.213622 |
| ARX(1,1) | 92 | 2 | 510.492518 | 547.569537 | -244.246259 | 19369.989267 | 0.009492 |
| ARX(2,2) | 55 | 1 | 553.766438 | 597.584732 | -263.883219 | 43.635811 | 0.004287 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 54 | 2 | 446.798303 | 473.763407 | -215.399151 | 62.877897 | 0.213622 |
| ARX(1,1) | 92 | 2 | 510.492518 | 547.569537 | -244.246259 | 19369.989267 | 0.009492 |
| ARX(2,2) | 55 | 1 | 553.766438 | 597.584732 | -263.883219 | 43.635811 | 0.004287 |

## Parameter Estimates From Best AIC Fits

### constant

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 8.506759 | 1.583037 |
| n0 | 0.005000 | 0.005161 |
| rho_p | 0.000010 | 0.259706 |
| rho_n | 0.000010 | 1.087155 |
| phi_p_plus | 1.050974 | 1.168389 |
| phi_n_minus | 0.000010 | 0.039698 |
| sigma_p | 0.213622 | 0.073689 |
| sigma_n | 0.463986 | 0.250562 |

### ARX(1,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.006195 | 0.083565 |
| rho_1 | 0.226001 | 0.161346 |
| phi_1 | 0.678940 | 0.183477 |
| p0 | 6.714397 | 17.234445 |
| n0 | 1.728725 | 0.426538 |
| rho_p | 0.313153 | 0.129167 |
| rho_n | 0.013862 | 0.234585 |
| phi_p_plus | 0.715508 | 0.098501 |
| phi_n_minus | 0.880822 | 1.096451 |
| sigma_p | 0.009492 | 0.001731 |
| sigma_n | 0.564762 | 0.237184 |

### ARX(2,2)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.055768 | 0.134788 |
| rho_1 | 0.231099 | 0.126895 |
| rho_2 | 0.333519 | 0.088086 |
| phi_1 | 0.791746 | 0.384160 |
| phi_2 | -0.193110 | 0.307943 |
| p0 | 1.555467 | 0.310893 |
| n0 | 2.111224 | 0.506625 |
| rho_p | 0.000010 | 0.546156 |
| rho_n | 0.288606 | 0.131727 |
| phi_p_plus | 0.000010 | 0.108792 |
| phi_n_minus | 1.252742 | 0.292665 |
| sigma_p | 0.004287 | 0.004926 |
| sigma_n | 0.572866 | 0.144670 |
