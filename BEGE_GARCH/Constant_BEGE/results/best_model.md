```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-05-28T19:48:01`
Total estimations: `20000`
Successful estimations: `19986`

```{warning}
Excluded 1 successful estimate(s) from best-model selection because `shape_p + shape_n` is numerically an integer, a known unstable point for the SciPy hyperu BEGE-density evaluation. Top excluded row: `constant`, seed `30`, draw `32`, reported LogLik `355.392806`.
```

## Best by Mean Type (Log-Likelihood)

| Mean Type | Seed | Draw | LogLik | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| constant | 1 | 41 | -199.843879 | 407.687759 | 421.170311 |
| ARX(1,1) | 50 | 48 | -184.396177 | 382.792353 | 406.386819 |
| ARX(2,1) | 33 | 65 | -181.884824 | 379.769649 | 406.734753 |
| ARX(2,2) | 23 | 92 | -181.688414 | 381.376828 | 411.712571 |

## Parameter Estimates From Best Log-Likelihood Fits

### constant

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675920 | 0.208030 |
| shape_n | 0.185967 | 0.109333 |
| sigma_p | 0.298260 | 0.086151 |
| sigma_n | 1.572583 | 0.062739 |

### ARX(1,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056014 | 0.004172 |
| rho_1 | 0.323706 | 0.007503 |
| phi_1 | 0.737787 | 0.014056 |
| shape_p | 2.627875 | 0.008489 |
| shape_n | 0.281123 | 0.010973 |
| sigma_p | 0.285666 | 0.006623 |
| sigma_n | 0.800204 | 0.003558 |

### ARX(2,1)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039254 | 0.014593 |
| rho_1 | 0.316821 | 0.076697 |
| rho_2 | 0.189143 | 0.046505 |
| phi_1 | 0.539235 | 0.027847 |
| shape_p | 3.317147 | 0.037952 |
| shape_n | 0.228113 | 0.064524 |
| sigma_p | 0.246532 | 0.053103 |
| sigma_n | 0.928310 | 0.026420 |

### ARX(2,2)

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004200 | 0.003184 |
| rho_1 | 0.323229 | 0.002223 |
| rho_2 | 0.163364 | 0.002537 |
| phi_1 | 0.410644 | 0.002267 |
| phi_2 | 0.194111 | 0.001594 |
| shape_p | 2.665345 | 0.002129 |
| shape_n | 0.282140 | 0.002203 |
| sigma_p | 0.271156 | 0.003024 |
| sigma_n | 0.840201 | 0.004885 |
