```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-05-31T16:30:31`
Total estimations: `4020`
Converged estimations: `3917`
Eligible estimations for best-model selection: `3917`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, and documented parameter/stability/unconditional-variance constraints.

## Global Best by AIC

- Mean type: `ARX(2,1)`
- Seed / draw: `91` / `3`
- AIC: `75.377840`
- BIC: `115.825497`
- LogLik: `-25.688920`
- Max shape: `395.826549`

## Global Best by BIC

- Mean type: `ARX(2,1)`
- Seed / draw: `91` / `3`
- AIC: `75.377840`
- BIC: `115.825497`
- LogLik: `-25.688920`
- Max shape: `395.826549`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 100 | 2 | 187.972717 | 214.937821 | -85.986358 | 186.829178 | 0.072113 |
| ARX(1,1) | 72 | 4 | 186.399065 | 223.476083 | -82.199533 | 204.131893 | 0.057609 |
| ARX(2,1) | 91 | 3 | 75.377840 | 115.825497 | -25.688920 | 395.826549 | 0.035492 |
| ARX(2,2) | 34 | 2 | 291.884502 | 335.702796 | -132.942251 | 146.783773 | 0.073678 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 100 | 2 | 187.972717 | 214.937821 | -85.986358 | 186.829178 | 0.072113 |
| ARX(1,1) | 72 | 4 | 186.399065 | 223.476083 | -82.199533 | 204.131893 | 0.057609 |
| ARX(2,1) | 91 | 3 | 75.377840 | 115.825497 | -25.688920 | 395.826549 | 0.035492 |
| ARX(2,2) | 34 | 2 | 291.884502 | 335.702796 | -132.942251 | 146.783773 | 0.073678 |

## Parameter Estimates From Best AIC Fits

### constant

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 5.671685 | 0.968909 |
| n0 | 0.005003 | 0.026126 |
| rho_p | 0.861405 | 0.002157 |
| rho_n | 0.117878 | 0.052076 |
| phi_p | 0.079471 | 0.000003 |
| phi_n | 0.881953 | 0.051794 |
| sigma_p | 0.072113 | 0.000005 |
| sigma_n | 0.287229 | 0.002115 |

### ARX(1,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.134875 | 0.000316 |
| rho_1 | 0.481958 | 0.000006 |
| phi_1 | 1.067438 | 179.057216 |
| p0 | 7.225714 | 975.382214 |
| n0 | 6.429995 | 179.058119 |
| rho_p | 0.077997 | 179.065758 |
| rho_n | 0.955600 | 0.000380 |
| phi_p | 0.611996 | 0.000002 |
| phi_n | 0.012901 | 0.000239 |
| sigma_p | 0.313328 | 0.000321 |
| sigma_n | 0.057609 | 0.000283 |

### ARX(2,1)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.057319 | 6099.087934 |
| rho_1 | 0.274752 | 6893.332675 |
| rho_2 | 0.236062 | 1692.649286 |
| phi_1 | 0.431935 | 8882.934120 |
| p0 | 0.195254 | 0.057180 |
| n0 | 5.063837 | 822793.344188 |
| rho_p | 0.998972 | 0.057781 |
| rho_n | 0.171948 | 10697.837383 |
| phi_p | 0.000010 | 0.000055 |
| phi_n | 0.635029 | 7535.413191 |
| sigma_p | 0.035492 | 0.057074 |
| sigma_n | 0.124464 | 0.000745 |

### ARX(2,2)

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.057000 | 0.131283 |
| rho_1 | 0.373407 | 0.739969 |
| rho_2 | 0.188337 | 0.054558 |
| phi_1 | -0.330196 | 0.793610 |
| phi_2 | 0.745016 | 0.077710 |
| p0 | 5.273562 | 0.514440 |
| n0 | 3.478176 | 2.063162 |
| rho_p | 0.727260 | 0.053210 |
| rho_n | 0.080459 | 0.360179 |
| phi_p | 0.070550 | 0.000002 |
| phi_n | 0.430884 | 0.054449 |
| sigma_p | 0.073678 | 0.000010 |
| sigma_n | 0.183330 | 0.000006 |
