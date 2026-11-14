```{raw:typst}
#set page(margin: auto)
```

# Model Comparison

This section ranks the admissible estimates of all model classes at each frequency. All models at a given frequency use the same effective sample and the residual definition $u_t=\pi_t-\hat{\pi}_t$, so their log likelihoods are comparable. Each information criterion counts every estimated parameter $k$, including variance, distribution, and transition parameters: $\mathrm{AIC}=2k-2\,\mathrm{LogLik}$ and $\mathrm{BIC}=k\ln N-2\,\mathrm{LogLik}$; lower values are better. OLS denotes the mean processes with constant Gaussian innovations. Only estimates that passed the admissibility checks of their class enter the ranking: GARCH-family fallback estimates, unverified regime-switching estimates, and BEGE estimates that failed the collection checks are excluded. For BEGE, the best eligible estimate of each specification and mean process enters.

The Gaussian regime-switching likelihood is unbounded because no variance floor is imposed, so the regime-switching estimates are verified local maxima rather than global maxima. The number of starts that reached each listed regime-switching optimum indicates how reliably the search finds it.

## Quarterly Comparison

The quarterly comparison uses the effective sample 1969Q2--2026Q3 ($N=230$) and covers 272 admissible estimates (6 OLS, 205 GARCH family, 13 Regime switching, 48 BEGE).

### Top 5 Models by AIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | BIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | BEGE-EGARCH, ARX(1,1) | BEGE | 13 | -167.3416 | 360.6832 | 405.3783 | 37 |
| 2 | BEGE-EGARCH, ARX(2,1) | BEGE | 14 | -167.6026 | 363.2053 | 411.3384 | 73 |
| 3 | GJR-GARCH(2,1), ARX(2,1), Student's $t$ | GARCH family | 11 | -172.3922 | 366.7843 | 404.6032 | 31 |
| 4 | GJR-GARCH(2,1), ARX(1,1), Student's $t$ | GARCH family | 10 | -173.4834 | 366.9668 | 401.3476 | 16 |
| 5 | Constant-p Full BEGE, ARX(1,1) | BEGE | 10 | -173.5656 | 367.1311 | 401.5119 | 19 |

### Top 5 Models by BIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | AIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | GARCH(2,1), Constant, Student's $t$ | GARCH family | 5 | -184.4270 | 378.8540 | 396.0444 | 99 |
| 2 | EGARCH(1,1), Constant, Student's $t$ | GARCH family | 5 | -184.4356 | 378.8711 | 396.0615 | 100 |
| 3 | GARCH(1,1), Constant, Student's $t$ | GARCH family | 4 | -187.4076 | 382.8151 | 396.5674 | 119 |
| 4 | GJR-GARCH(1,1), Constant, Student's $t$ | GARCH family | 5 | -184.7523 | 379.5047 | 396.6951 | 104 |
| 5 | EGARCH(1,1), Constant, Gaussian mixture | GARCH family | 7 | -179.6023 | 373.2045 | 397.2711 | 65 |

### Best Estimate in Each Class

| Class | Criterion | Best estimate in class | Value | Overall rank |
|---|---|---|---:|---:|
| OLS | AIC | ARX(1,1), Gaussian | 441.5382 | 265 of 272 |
|  | BIC | ARX(1,1), Gaussian | 455.2905 | 256 of 272 |
| GARCH family | AIC | GJR-GARCH(2,1), ARX(2,1), Student's $t$ | 366.7843 | 3 of 272 |
|  | BIC | GARCH(2,1), Constant, Student's $t$ | 396.0444 | 1 of 272 |
| Regime switching | AIC | ARX(1,1), K=2, switching AR and variance | 377.6354 | 95 of 272 |
|  | BIC | ARX(1,1), K=2, switching variance | 403.8779 | 30 of 272 |
| BEGE | AIC | BEGE-EGARCH, ARX(1,1) | 360.6832 | 1 of 272 |
|  | BIC | Constant-p Full BEGE, Constant | 399.0462 | 9 of 272 |

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

## Monthly Comparison

The monthly comparison uses the effective sample 1969M2--2026M9 ($N=692$) and covers 281 admissible estimates (6 OLS, 216 GARCH family, 11 Regime switching, 48 BEGE).

### Top 5 Models by AIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | BIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Symmetric BEGE, ARX(1,1) | BEGE | 10 | 104.2258 | -188.4515 | -143.0557 | 6 |
| 2 | Symmetric BEGE, ARX(2,1) | BEGE | 11 | 105.1415 | -188.2830 | -138.3476 | 13 |
| 3 | Full BEGE, ARX(1,1) | BEGE | 13 | 106.7472 | -187.4944 | -128.4798 | 45 |
| 4 | Full BEGE, ARX(2,1) | BEGE | 14 | 107.5103 | -187.0207 | -123.4665 | 64 |
| 5 | Symmetric BEGE, ARX(2,2) | BEGE | 12 | 105.3633 | -186.7265 | -132.2515 | 32 |

### Top 5 Models by BIC

| Rank | Model | Class | $k$ | LogLik | AIC | BIC | AIC rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | GARCH(1,1), ARX(1,1), Student's $t$ | GARCH family | 7 | 97.7611 | -181.5223 | -149.7452 | 24 |
| 2 | GJR-GARCH(1,1), ARX(1,1), Student's $t$ | GARCH family | 8 | 100.1943 | -184.3886 | -148.0719 | 13 |
| 3 | GARCH(1,1), ARX(2,1), Student's $t$ | GARCH family | 8 | 97.9755 | -179.9510 | -143.6343 | 36 |
| 4 | GARCH(2,1), ARX(1,1), Student's $t$ | GARCH family | 8 | 97.8253 | -179.6507 | -143.3340 | 41 |
| 5 | GARCH(1,2), ARX(1,1), Student's $t$ | GARCH family | 8 | 97.7611 | -179.5223 | -143.2056 | 43 |

### Best Estimate in Each Class

| Class | Criterion | Best estimate in class | Value | Overall rank |
|---|---|---|---:|---:|
| OLS | AIC | ARX(1,1), Gaussian | 12.8393 | 275 of 281 |
|  | BIC | ARX(1,1), Gaussian | 30.9976 | 275 of 281 |
| GARCH family | AIC | GJR-GARCH(1,1), ARX(1,1), Gaussian mixture | -184.4273 | 12 of 281 |
|  | BIC | GARCH(1,1), ARX(1,1), Student's $t$ | -149.7452 | 1 of 281 |
| Regime switching | AIC | ARX(1,1), K=3, switching variance | -176.7965 | 64 of 281 |
|  | BIC | ARX(1,1), K=2, switching variance | -134.1354 | 25 of 281 |
| BEGE | AIC | Symmetric BEGE, ARX(1,1) | -188.4515 | 1 of 281 |
|  | BIC | Symmetric BEGE, ARX(1,1) | -143.0557 | 6 of 281 |
