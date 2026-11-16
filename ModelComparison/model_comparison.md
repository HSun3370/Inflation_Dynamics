```{raw:typst}
#set page(margin: auto)
```

# Model Comparison

This section ranks the admissible estimates of all model classes at each frequency. All models at a given frequency use the same effective sample and the residual definition $u_t=\pi_t-\hat{\pi}_t$, so their log likelihoods are comparable. Each information criterion counts every estimated parameter $k$, including variance, distribution, and transition parameters: $\mathrm{AIC}=2k-2\,\mathrm{LogLik}$ and $\mathrm{BIC}=k\ln N-2\,\mathrm{LogLik}$; lower values are better. OLS denotes the mean processes with constant Gaussian innovations. Only estimates that passed the admissibility checks of their class enter the ranking: GARCH-family fallback estimates, unverified regime-switching estimates, and BEGE estimates that failed the collection checks are excluded. For BEGE, the best eligible estimate of each specification and mean process enters.

The Gaussian regime-switching likelihood is unbounded because no variance floor is imposed, so the regime-switching estimates are verified local maxima rather than global maxima. The number of starts that reached each listed regime-switching optimum indicates how reliably the search finds it.

## Quarterly Comparison

The quarterly comparison uses the effective sample 1969Q2--2026Q3 ($N=230$) and covers 283 admissible estimates (6 OLS, 205 GARCH family, 12 Regime switching, 60 BEGE).

### Top 5 Models by AIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | BIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | BEGE-EGARCH, ARX(1,1) | BEGE | 13 | -167.3416 | 360.6832 | 405.3783 | 38 |
| 2 | BEGE-EGARCH, ARX(2,1) | BEGE | 14 | -167.6026 | 363.2053 | 411.3384 | 77 |
| 3 | ARX(1,1), K=3, switching AR and variance | Regime switching | 16 | -167.3355 | 366.6710 | 421.6803 | 162 |
| 4 | GJR-GARCH(2,1), ARX(2,1), Student's $t$ | GARCH family | 11 | -172.3922 | 366.7843 | 404.6032 | 31 |
| 5 | GJR-GARCH(2,1), ARX(1,1), Student's $t$ | GARCH family | 10 | -173.4834 | 366.9668 | 401.3476 | 16 |

### Top 5 Models by BIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | AIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | GARCH(2,1), Constant, Student's $t$ | GARCH family | 5 | -184.4270 | 378.8540 | 396.0444 | 108 |
| 2 | EGARCH(1,1), Constant, Student's $t$ | GARCH family | 5 | -184.4356 | 378.8711 | 396.0615 | 109 |
| 3 | GARCH(1,1), Constant, Student's $t$ | GARCH family | 4 | -187.4076 | 382.8151 | 396.5674 | 128 |
| 4 | GJR-GARCH(1,1), Constant, Student's $t$ | GARCH family | 5 | -184.7523 | 379.5047 | 396.6951 | 113 |
| 5 | EGARCH(1,1), Constant, Gaussian mixture | GARCH family | 7 | -179.6023 | 373.2045 | 397.2711 | 72 |

### Best Estimate in Each Class

| Class | Criterion | Best estimate in class | Value | Overall rank |
|---|---|---|---:|---:|
| OLS | AIC | ARX(1,1), Gaussian | 441.5382 | 276 of 283 |
|  | BIC | ARX(1,1), Gaussian | 455.2905 | 267 of 283 |
| GARCH family | AIC | GJR-GARCH(2,1), ARX(2,1), Student's $t$ | 366.7843 | 4 of 283 |
|  | BIC | GARCH(2,1), Constant, Student's $t$ | 396.0444 | 1 of 283 |
| Regime switching | AIC | ARX(1,1), K=3, switching AR and variance | 366.6710 | 3 of 283 |
|  | BIC | ARX(1,1), K=2, switching variance | 403.8779 | 30 of 283 |
| BEGE | AIC | BEGE-EGARCH, ARX(1,1) | 360.6832 | 1 of 283 |
|  | BIC | Constant-p Full BEGE, Constant | 399.0462 | 9 of 283 |

Starts that reached the listed regime-switching optimum, with a log likelihood within $10^{-3}$ of it: ARX(1,1), K=3, switching AR and variance: 2,754 of 4,771 valid starts. For reference, the listed GARCH-family optima were reached by 2,861 to 4,340 of 5,000 starts.

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

## Monthly Comparison

The monthly comparison uses the effective sample 1969M2--2026M9 ($N=692$) and covers 295 admissible estimates (6 OLS, 216 GARCH family, 13 Regime switching, 60 BEGE).

### Top 5 Models by AIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | BIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | ARX(2,1), K=3, switching AR and variance | Regime switching | 19 | 116.7792 | -195.5584 | -109.3062 | 106 |
| 2 | ARX(2,2), K=3, switching AR and variance | Regime switching | 20 | 117.6247 | -195.2495 | -104.4578 | 111 |
| 3 | Positive-only-p/full-n BEGE, ARX(1,1) | BEGE | 12 | 106.7471 | -189.4943 | -135.0193 | 22 |
| 4 | Positive-only-p/full-n BEGE, ARX(2,1) | BEGE | 13 | 107.5095 | -189.0190 | -130.0044 | 42 |
| 5 | Symmetric BEGE, ARX(1,1) | BEGE | 10 | 104.2258 | -188.4515 | -143.0557 | 6 |

### Top 5 Models by BIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | AIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | GARCH(1,1), ARX(1,1), Student's $t$ | GARCH family | 7 | 97.7611 | -181.5223 | -149.7452 | 32 |
| 2 | GJR-GARCH(1,1), ARX(1,1), Student's $t$ | GARCH family | 8 | 100.1943 | -184.3886 | -148.0719 | 20 |
| 3 | GARCH(1,1), ARX(2,1), Student's $t$ | GARCH family | 8 | 97.9755 | -179.9510 | -143.6343 | 44 |
| 4 | GARCH(2,1), ARX(1,1), Student's $t$ | GARCH family | 8 | 97.8253 | -179.6507 | -143.3340 | 49 |
| 5 | GARCH(1,2), ARX(1,1), Student's $t$ | GARCH family | 8 | 97.7611 | -179.5223 | -143.2056 | 51 |

### Best Estimate in Each Class

| Class | Criterion | Best estimate in class | Value | Overall rank |
|---|---|---|---:|---:|
| OLS | AIC | ARX(1,1), Gaussian | 12.8393 | 289 of 295 |
|  | BIC | ARX(1,1), Gaussian | 30.9976 | 289 of 295 |
| GARCH family | AIC | GJR-GARCH(1,1), ARX(1,1), Gaussian mixture | -184.4273 | 19 of 295 |
|  | BIC | GARCH(1,1), ARX(1,1), Student's $t$ | -149.7452 | 1 of 295 |
| Regime switching | AIC | ARX(2,1), K=3, switching AR and variance | -195.5584 | 1 of 295 |
|  | BIC | ARX(1,1), K=2, switching variance | -134.1354 | 26 of 295 |
| BEGE | AIC | Positive-only-p/full-n BEGE, ARX(1,1) | -189.4943 | 3 of 295 |
|  | BIC | Symmetric BEGE, ARX(1,1) | -143.0557 | 6 of 295 |

Starts that reached the listed regime-switching optimum, with a log likelihood within $10^{-3}$ of it: ARX(2,1), K=3, switching AR and variance: 3,021 of 4,896 valid starts; ARX(2,2), K=3, switching AR and variance: 3,111 of 4,894 valid starts. For reference, the listed GARCH-family optima were reached by 4,090 to 4,209 of 5,000 starts.
