```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-05-31T10:14:48`
Total saved estimations: `702`
Converged estimations: `73`
Eligible estimations for best-model selection: `5`

SEs are skipped during Slurm estimation jobs and computed only for the eligible best AIC fit in each mean process.

Selection screen: finite AIC/BIC/log-likelihood, successful optimizer status, and `max(p_t, n_t) < 200`.

## Global Best by AIC

- Mean type: `constant`
- Seed / draw: `4` / `1`
- AIC: `399.916891`
- BIC: `426.881995`
- LogLik: `-191.958446`
- Max shape: `5.709619`

## Global Best by BIC

- Mean type: `constant`
- Seed / draw: `4` / `1`
- AIC: `399.916891`
- BIC: `426.881995`
- LogLik: `-191.958446`
- Max shape: `5.709619`

## Eligible Best by Mean Type (AIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 4 | 1 | 399.916891 | 426.881995 | -191.958446 | 5.709619 | 0.720500 |
| ARX(2,2) | 55 | 1 | 558.093343 | 601.911637 | -266.046671 | 43.635811 | 0.004287 |

## Eligible Best by Mean Type (BIC)

| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 4 | 1 | 399.916891 | 426.881995 | -191.958446 | 5.709619 | 0.720500 |
| ARX(2,2) | 55 | 1 | 558.093343 | 601.911637 | -266.046671 | 43.635811 | 0.004287 |

## Parameter Estimates From Eligible Best AIC Fits

### constant

SE status: `computed`

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.480263 | 0.402930 |
| n0 | 0.328460 | 0.650853 |
| rho_p | 0.232389 | 1.707181 |
| rho_n | 0.001467 | 2.422704 |
| phi_p_plus | 0.532692 | 0.398849 |
| phi_n_minus | 0.347119 | 2.431084 |
| sigma_p | 0.816125 | 1.450743 |
| sigma_n | 0.720500 | 0.509619 |

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
