# Best Model Report

Selection uses all estimated models. Prefer optimizer-successful models when available.

## Best by AIC
- Model ID: `M117`
- Distribution: `mix_normal`
- Mean process: `ARX_1_1`
- Volatility: `EGARCH(2,1)`
- nobs: `213`
- Log-likelihood: `-165.214755`
- AIC: `354.429511`
- BIC: `394.765017`
- Persistence proxy: `0.7036005295949085`
- Implied initial variance: `0.3429669004783652`
- Variance status: `proxy_from_log_variance`
- Optimizer success: `True`
- Convergence flag: `0`

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| Const | 0.086147 | 0.013153 | 6.549350 | 0.000000 |
| Inflation[1] | 0.212415 | 0.028890 | 7.352610 | 0.000000 |
| SPF | 0.733637 | 0.001867 | 392.882038 | 0.000000 |
| alpha[1] | 0.574915 | 0.148268 | 3.877535 | 0.000106 |
| alpha[2] | -0.000396 | 0.212310 | -0.001865 | 0.998512 |
| beta[1] | 0.703601 | 0.103625 | 6.789845 | 0.000000 |
| gamma[1] | 0.097084 | 0.099626 | 0.974491 | 0.329813 |
| gamma[2] | 0.304117 | 0.095088 | 3.198259 | 0.001383 |
| mu_1 | 0.045566 | 0.048813 | 0.933473 | 0.350576 |
| omega | -0.317183 | 0.153359 | -2.068242 | 0.038617 |
| p_1 | 0.935227 | 0.039821 | 23.485813 | 0.000000 |
| sigma_1_sq | 0.637190 | 0.140299 | 4.541648 | 0.000006 |

## Best by BIC
- Model ID: `M050`
- Distribution: `studentst`
- Mean process: `Constant_anchor`
- Volatility: `GJR-GARCH(1,1)`
- nobs: `213`
- Log-likelihood: `-176.678323`
- AIC: `363.356646`
- BIC: `380.163107`
- Persistence proxy: `0.9076690025361234`
- Implied initial variance: `0.6943493866192694`
- Variance status: `exact_under_symmetry`
- Optimizer success: `True`
- Convergence flag: `0`

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| alpha[1] | 0.794841 | 0.259461 | 3.063429 | 0.002188 |
| beta[1] | 0.434574 | 0.117454 | 3.699951 | 0.000216 |
| gamma[1] | -0.643492 | 0.236470 | -2.721239 | 0.006504 |
| nu | 4.694320 | 1.501941 | 3.125503 | 0.001775 |
| omega | 0.064110 | 0.025792 | 2.485645 | 0.012932 |

## Notes on Initialization
- `GARCH`: uses exact unconditional variance `omega / (1 - sum(alpha) - sum(beta))` when stationary.
- `GJR-GARCH`: uses `omega / (1 - sum(alpha) - 0.5*sum(gamma) - sum(beta))` under symmetric innovations.
- `EGARCH`: uses proxy `exp(omega / (1 - sum(beta)))` from log-variance recursion.
- In `arch`, `backcast` sets recursion start; it is not automatically replaced by model-implied unconditional variance during optimization.

