```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```


# Model Selection Report

Effective sample: **1969-02--2026-09** (**692 observations**, `Aggregate_CPI_inflation_Monthly.pkl`, data vintage 2026-10-20).

## Parameter Ranges and Random Starts

All models are estimated with `arch` 7.0.0, which maximizes the likelihood by SLSQP subject to the bounds and linear constraints in the second column of the table below. `arch` imposes no bounds on mean-process coefficients or on the EGARCH coefficients $\alpha_i$ and $\gamma_j$. If a supplied starting vector violates any bound or constraint, `arch` silently replaces the whole vector with its own default start. Each model combination uses 5000 starts. Start 1 is the `arch` default (Gaussian mixture: the Student's $t$ estimates plus $(p_1,\mu_1,\sigma_1^2)=(0.3,0.1,1)$ when that vector is feasible, otherwise the `arch` default); starts 2--5000 are random draws from the third column. Every random draw lies inside `arch`'s feasible set and inside the stationary region (volatility persistence and AR roots below $1-10^{-6}$ in modulus), so `arch` uses every random start as drawn.

$\bar v$ denotes the mean squared residual at `arch`'s mean starting values (OLS residuals for ARX means, $\pi_t-SPF_{t-1}$ for the Constant mean): Constant 0.0779, ARX(1,0) 0.0646, ARX(1,1) 0.0590, ARX(2,0) 0.0642, ARX(2,1) 0.0589, ARX(2,2) 0.0588.

| Parameter | `arch` bound / constraint | Random-start draw |
|---|---|---|
| **Mean process** | | |
| $\rho_1$ in ARX(1,·) | $\lvert\rho_1\rvert<1-10^{-6}$ | $U(-1,1)$ (AR stationarity) |
| $(\rho_1,\rho_2)$ in ARX(2,·) | $\rho_2>-1$, $\rho_1+\rho_2<1$, $\rho_2-\rho_1<1$; both roots of $r^2-\rho_1r-\rho_2=0$ satisfy $\lvert r\rvert<1-10^{-6}$ | uniform on the stationarity triangle $\rho_2>-1$, $\lvert\rho_1\rvert<1-\rho_2$ |
| $c$, $\phi_1$, $\phi_2$ | none | uniform on $\hat b(\rho)\pm 4\,\widehat{SE}(\rho)$, the OLS fit of $\pi_t-\rho_1\pi_{t-1}-\rho_2\pi_{t-2}$ on a constant and the SPF terms given the drawn $\rho$ |
| **GARCH** | | |
| $\omega$ | $[10^{-8}\bar v,\ 10\bar v]$ | log-uniform on the bound |
| $\alpha_i$, $\beta_k$ | $\alpha_i,\beta_k\in[0,1]$; $\sum\alpha_i+\sum\beta_k\le1$ | uniform on $\{\alpha,\beta\ge0,\ \sum\alpha_i+\sum\beta_k<1\}$ |
| **GJR-GARCH** | | |
| $\omega$ | $[10^{-8}\bar v,\ 10\bar v]$ | log-uniform on the bound |
| $\alpha_i$ (on $(u^+)^2$) | $[0,1]$ | uniform on $\{\alpha_i\in[0,1],\ \gamma_i\ge0,\ \beta_k\ge0,\ \tfrac12\sum\alpha_i+\tfrac12\sum\gamma_i+\sum\beta_k<1\}$ |
| $\gamma_i$ (on $(u^-)^2$) | `arch` estimates $\gamma_i-\alpha_i\in[-1,2]$ subject to $\gamma_i\ge0$ | joint draw above |
| $\beta_k$ | $[0,1]$; $\tfrac12\sum\alpha_i+\tfrac12\sum\gamma_i+\sum\beta_k\le1$ | joint draw above |
| **EGARCH** | | |
| $\omega$ | $[\ln\bar v-\ln10^4,\ \ln\bar v+\ln10^4]$ | uniform on the bound |
| $\alpha_i$ | none | $U(-2,2)$ |
| $\gamma_j$ | none | $U(-2,2)$ |
| $\beta_k$ | $[0,q]$; $\sum\beta_k\le1$ | uniform on $\{\beta\ge0,\ \sum\beta_k<1\}$ |
| **Student's $t$** | | |
| $\nu$ | $[2.05,\ 500]$ | log-uniform on the bound |
| **Gaussian mixture** | | |
| $p_1$ | $[10^{-4},\ 1-10^{-4}]$ | uniform on the bound |
| $\mu_1$ | $[-50,\ 50]$ | uniform on the values that allow $\sigma_2^2\ge10^{-6}$ |
| $\sigma_1^2$ | $[10^{-6},\ 100]$; $\sigma_2^2>0$ through a likelihood penalty | uniform on $[10^{-6},\ \min(100,\bar\sigma_1^2)]$, where $\bar\sigma_1^2$ gives $\sigma_2^2=10^{-6}$ |

Notes:

- *Uniform on a region* means a uniform draw over the whole region: Dirichlet(1,…,1) weights for the volatility simplices, rejection sampling for the AR triangle and for `arch`'s caps. *Log-uniform* draws are uniform in the logarithm of the parameter. The AR restrictions apply to random starts and final selection; the optimizer itself imposes no bounds on the AR coefficients.
- Compared with the stationarity conditions in the README, `arch`'s feasible set is the same for GARCH. For GJR-GARCH, `arch` also caps $\alpha_i\le1$, although stationarity alone allows $\alpha_i<2$ when $\gamma_i$ is small. For EGARCH, `arch` requires $\beta_k\ge0$, under which $\sum\beta_k<1$ is the stationarity condition; $\omega$ is also bounded. The mixture bounds come from this repository's `MixNormal` class. $\alpha_i$ and $\gamma_j$ in EGARCH are unbounded in `arch`; the reported EGARCH estimates have $\max|\alpha_i|=0.6378$ and $\max|\gamma_j|=0.2049$, well inside the draw range $[-2,2]$.
- Selection: for each combination the reported estimate has the highest log likelihood among *admissible* starts, which must pass five checks. (1) SLSQP reports success and `arch` returns convergence flag 0. (2) The first-order (KKT) conditions hold: the numerical gradient of the log likelihood, net of non-negative multipliers on the active bounds and constraints, is at most 0.005$\cdot T$ in absolute value. When a start that would become the best fails this check, SLSQP is restarted from its own solution up to 5 times, because SLSQP sometimes stops just short of a genuine maximum. Points on extremely steep ridges, where SLSQP reports success with gradients of $10^1$--$10^6$, do not move and stay rejected. (3) Volatility is stationary. (4) The AR roots lie inside the unit circle. (5) For the Gaussian mixture, the component variances satisfy $\min(\sigma_1^2,\sigma_2^2)/\max(\sigma_1^2,\sigma_2^2)\ge0.01$. This Hathaway-type restriction keeps the mixture likelihood bounded; without it the best fits place a component with variance $10^{-6}$ on one or two observations. If no start passes, the best converged start that satisfies the first-order conditions is reported and flagged in the CSV `selection_status` column. `best_loglik_hits` in the summary CSV counts the admissible starts that came within 0.01 of the reported log likelihood, and `params_at_arch_bound` lists parameters of the reported estimate that sit on an `arch` bound.
- In 2 of 216 combinations, a converged fit that satisfies the first-order conditions but fails stationarity or the mixture rule has a higher log likelihood than the reported estimate (largest difference 2.5353). For these combinations the likelihood peaks on the stationarity boundary or at a degenerate mixture, and the reported estimate is the best admissible local maximum. The CSV column `best_inadmissible_loglikelihood` records the higher value.
- In this run, 64 of 216 reported estimates have at least one parameter on an `arch` bound; parameters on a bound, with counts: beta (51), alpha (13). Standard errors of estimates on a bound or an active constraint (for example $\beta_k=0$, or a GJR $\gamma_i=0$) are not asymptotically normal and should be read with caution.

## Model Selection

For each model combination, the reported row uses the highest log-likelihood estimate that passes all admissibility checks (convergence, first-order conditions, volatility and mean stationarity, and the mixture variance ratio) across 5000 starts. If no start passes all checks, the reported row uses the best converged start that satisfies the first-order conditions and is flagged in the CSV `selection_status` column.

**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH AIC | GJR Mean | GJR Vol | GJR AIC | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(1,1) | (1,1) | -133.1821 | ARX(1,1) | (2,1) | -136.9738 | ARX(1,1) | (2,1) | -129.6969 |
| Student's $t$ | ARX(1,1) | (1,1) | -181.5223 | ARX(1,1) | (1,1) | -184.3886 | ARX(1,1) | (1,1) | -176.6642 |
| Gaussian mixture | ARX(1,1) | (1,1) | -181.2549 | ARX(1,1) | (1,1) | <span style="color:red">-184.4273</span> | ARX(1,1) | (1,1) | -178.7365 |

**Table 3: Model selection by BIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH BIC | GJR Mean | GJR Vol | GJR BIC | EGARCH Mean | EGARCH Vol | EGARCH BIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(1,1) | (1,1) | -105.9446 | ARX(1,1) | (1,1) | -102.6268 | ARX(1,1) | (2,1) | -88.8407 |
| Student's $t$ | ARX(1,1) | (1,1) | <span style="color:red">-149.7452</span> | ARX(1,1) | (1,1) | -148.0719 | ARX(1,1) | (1,1) | -140.3475 |
| Gaussian mixture | ARX(1,1) | (1,1) | -140.3986 | ARX(1,1) | (1,1) | -139.0314 | ARX(1,1) | (1,1) | -133.3407 |

Table 4 repeats the BIC selection over the five ARX mean processes only; the Constant mean is excluded from the candidates. Because Table 3 selects no Constant-mean model, Table 4 coincides with Table 3.

**Table 4: Model selection by BIC, excluding the Constant mean: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH BIC | GJR Mean | GJR Vol | GJR BIC | EGARCH Mean | EGARCH Vol | EGARCH BIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(1,1) | (1,1) | -105.9446 | ARX(1,1) | (1,1) | -102.6268 | ARX(1,1) | (2,1) | -88.8407 |
| Student's $t$ | ARX(1,1) | (1,1) | <span style="color:red">-149.7452</span> | ARX(1,1) | (1,1) | -148.0719 | ARX(1,1) | (1,1) | -140.3475 |
| Gaussian mixture | ARX(1,1) | (1,1) | -140.3986 | ARX(1,1) | (1,1) | -139.0314 | ARX(1,1) | (1,1) | -133.3407 |

## All Model Results

| Distribution | Mean Process | Volatility Process | Best LogLik | AIC | BIC |
|---|---|---|---:|---:|---:|
| Normal | Constant | GARCH(1,1) | 19.6528 | -33.3056 | -19.6868 |
| Normal | Constant | GARCH(1,2) | 21.0226 | -34.0452 | -15.8868 |
| Normal | Constant | GARCH(2,1) | 19.6528 | -31.3056 | -13.1472 |
| Normal | Constant | GARCH(2,2) | 21.0226 | -32.0452 | -9.3472 |
| Normal | Constant | GJR-GARCH(1,1) | 19.9204 | -31.8408 | -13.6824 |
| Normal | Constant | GJR-GARCH(1,2) | 21.4918 | -32.9837 | -10.2858 |
| Normal | Constant | GJR-GARCH(2,1) | 20.2213 | -28.4425 | -1.2050 |
| Normal | Constant | GJR-GARCH(2,2) | 24.0727 | -34.1455 | -2.3684 |
| Normal | Constant | EGARCH(1,1) | 14.9629 | -21.9258 | -3.7675 |
| Normal | Constant | EGARCH(1,2) | 17.2697 | -24.5394 | -1.8415 |
| Normal | Constant | EGARCH(2,1) | 23.7179 | -35.4359 | -8.1983 |
| Normal | Constant | EGARCH(2,2) | 23.7179 | -33.4359 | -1.6588 |
| Normal | ARX(1,0) | GARCH(1,1) | 38.1432 | -66.2865 | -43.5885 |
| Normal | ARX(1,0) | GARCH(1,2) | 38.1668 | -64.3336 | -37.0961 |
| Normal | ARX(1,0) | GARCH(2,1) | 38.1432 | -64.2865 | -37.0489 |
| Normal | ARX(1,0) | GARCH(2,2) | 38.5666 | -63.1332 | -31.3561 |
| Normal | ARX(1,0) | GJR-GARCH(1,1) | 39.8135 | -67.6269 | -40.3894 |
| Normal | ARX(1,0) | GJR-GARCH(1,2) | 39.8149 | -65.6298 | -33.8527 |
| Normal | ARX(1,0) | GJR-GARCH(2,1) | 42.0697 | -68.1393 | -31.8227 |
| Normal | ARX(1,0) | GJR-GARCH(2,2) | 42.4833 | -66.9665 | -26.1103 |
| Normal | ARX(1,0) | EGARCH(1,1) | 33.6339 | -55.2677 | -28.0302 |
| Normal | ARX(1,0) | EGARCH(1,2) | 33.6796 | -53.3593 | -21.5822 |
| Normal | ARX(1,0) | EGARCH(2,1) | 38.7599 | -61.5199 | -25.2032 |
| Normal | ARX(1,0) | EGARCH(2,2) | 38.7599 | -59.5199 | -18.6636 |
| Normal | ARX(1,1) | GARCH(1,1) | 72.5910 | -133.1821 | -105.9446 |
| Normal | ARX(1,1) | GARCH(1,2) | 72.5910 | -131.1821 | -99.4050 |
| Normal | ARX(1,1) | GARCH(2,1) | 72.7163 | -131.4326 | -99.6555 |
| Normal | ARX(1,1) | GARCH(2,2) | 72.7332 | -129.4663 | -93.1496 |
| Normal | ARX(1,1) | GJR-GARCH(1,1) | 74.2020 | -134.4039 | -102.6268 |
| Normal | ARX(1,1) | GJR-GARCH(1,2) | 74.2020 | -132.4039 | -96.0873 |
| Normal | ARX(1,1) | GJR-GARCH(2,1) | 77.4869 | -136.9738 | -96.1175 |
| Normal | ARX(1,1) | GJR-GARCH(2,2) | 77.8230 | -135.6460 | -90.2501 |
| Normal | ARX(1,1) | EGARCH(1,1) | 67.1366 | -120.2732 | -88.4961 |
| Normal | ARX(1,1) | EGARCH(1,2) | 67.1494 | -118.2988 | -81.9821 |
| Normal | ARX(1,1) | EGARCH(2,1) | 73.8485 | -129.6969 | -88.8407 |
| Normal | ARX(1,1) | EGARCH(2,2) | 73.8485 | -127.6969 | -82.3011 |
| Normal | ARX(2,0) | GARCH(1,1) | 40.0210 | -68.0419 | -40.8044 |
| Normal | ARX(2,0) | GARCH(1,2) | 40.0314 | -66.0629 | -34.2858 |
| Normal | ARX(2,0) | GARCH(2,1) | 40.0210 | -66.0419 | -34.2648 |
| Normal | ARX(2,0) | GARCH(2,2) | 40.4451 | -64.8902 | -28.5735 |
| Normal | ARX(2,0) | GJR-GARCH(1,1) | 41.1519 | -68.3038 | -36.5267 |
| Normal | ARX(2,0) | GJR-GARCH(1,2) | 41.1527 | -66.3054 | -29.9887 |
| Normal | ARX(2,0) | GJR-GARCH(2,1) | 43.3291 | -68.6582 | -27.8019 |
| Normal | ARX(2,0) | GJR-GARCH(2,2) | 43.5202 | -67.0405 | -21.6446 |
| Normal | ARX(2,0) | EGARCH(1,1) | 35.2128 | -56.4255 | -24.6484 |
| Normal | ARX(2,0) | EGARCH(1,2) | 35.2454 | -54.4908 | -18.1742 |
| Normal | ARX(2,0) | EGARCH(2,1) | 39.5562 | -61.1124 | -20.2562 |
| Normal | ARX(2,0) | EGARCH(2,2) | 39.5562 | -59.1124 | -13.7166 |
| Normal | ARX(2,1) | GARCH(1,1) | 73.0070 | -132.0140 | -100.2369 |
| Normal | ARX(2,1) | GARCH(1,2) | 73.0070 | -130.0140 | -93.6973 |
| Normal | ARX(2,1) | GARCH(2,1) | 73.1873 | -130.3745 | -94.0578 |
| Normal | ARX(2,1) | GARCH(2,2) | 73.2155 | -128.4311 | -87.5748 |
| Normal | ARX(2,1) | GJR-GARCH(1,1) | 74.8172 | -133.6343 | -97.3176 |
| Normal | ARX(2,1) | GJR-GARCH(1,2) | 74.8172 | -131.6343 | -90.7780 |
| Normal | ARX(2,1) | GJR-GARCH(2,1) | 78.1335 | -136.2671 | -90.8712 |
| Normal | ARX(2,1) | GJR-GARCH(2,2) | 78.6035 | -135.2071 | -85.2717 |
| Normal | ARX(2,1) | EGARCH(1,1) | 67.7831 | -119.5661 | -83.2494 |
| Normal | ARX(2,1) | EGARCH(1,2) | 67.8093 | -117.6187 | -76.7624 |
| Normal | ARX(2,1) | EGARCH(2,1) | 74.5878 | -129.1757 | -83.7798 |
| Normal | ARX(2,1) | EGARCH(2,2) | 74.5878 | -127.1757 | -77.2402 |
| Normal | ARX(2,2) | GARCH(1,1) | 73.2449 | -130.4898 | -94.1731 |
| Normal | ARX(2,2) | GARCH(1,2) | 73.2449 | -128.4898 | -87.6335 |
| Normal | ARX(2,2) | GARCH(2,1) | 73.4491 | -128.8982 | -88.0419 |
| Normal | ARX(2,2) | GARCH(2,2) | 73.4808 | -126.9616 | -81.5657 |
| Normal | ARX(2,2) | GJR-GARCH(1,1) | 75.0642 | -132.1285 | -91.2722 |
| Normal | ARX(2,2) | GJR-GARCH(1,2) | 75.0642 | -130.1285 | -84.7326 |
| Normal | ARX(2,2) | GJR-GARCH(2,1) | 78.6767 | -135.3535 | -85.4180 |
| Normal | ARX(2,2) | GJR-GARCH(2,2) | 79.2383 | -134.4766 | -80.0016 |
| Normal | ARX(2,2) | EGARCH(1,1) | 67.9456 | -117.8913 | -77.0350 |
| Normal | ARX(2,2) | EGARCH(1,2) | 67.9676 | -115.9353 | -70.5394 |
| Normal | ARX(2,2) | EGARCH(2,1) | 75.1030 | -128.2061 | -78.2706 |
| Normal | ARX(2,2) | EGARCH(2,2) | 75.1030 | -126.2061 | -71.7310 |
| Student's $t$ | Constant | GARCH(1,1) | 37.5723 | -67.1445 | -48.9862 |
| Student's $t$ | Constant | GARCH(1,2) | 38.5614 | -67.1228 | -44.4249 |
| Student's $t$ | Constant | GARCH(2,1) | 37.5723 | -65.1445 | -42.4466 |
| Student's $t$ | Constant | GARCH(2,2) | 38.5614 | -65.1228 | -37.8853 |
| Student's $t$ | Constant | GJR-GARCH(1,1) | 38.6006 | -67.2011 | -44.5032 |
| Student's $t$ | Constant | GJR-GARCH(1,2) | 39.8392 | -67.6785 | -40.4410 |
| Student's $t$ | Constant | GJR-GARCH(2,1) | 38.6614 | -63.3227 | -31.5456 |
| Student's $t$ | Constant | GJR-GARCH(2,2) | 40.8526 | -65.7052 | -29.3885 |
| Student's $t$ | Constant | EGARCH(1,1) | 35.4115 | -60.8229 | -38.1250 |
| Student's $t$ | Constant | EGARCH(1,2) | 37.1046 | -62.2091 | -34.9716 |
| Student's $t$ | Constant | EGARCH(2,1) | 41.4660 | -68.9320 | -37.1549 |
| Student's $t$ | Constant | EGARCH(2,2) | 41.4660 | -66.9320 | -30.6153 |
| Student's $t$ | ARX(1,0) | GARCH(1,1) | 61.1154 | -110.2309 | -82.9933 |
| Student's $t$ | ARX(1,0) | GARCH(1,2) | 61.1456 | -108.2911 | -76.5140 |
| Student's $t$ | ARX(1,0) | GARCH(2,1) | 61.1154 | -108.2309 | -76.4538 |
| Student's $t$ | ARX(1,0) | GARCH(2,2) | 61.4602 | -106.9205 | -70.6038 |
| Student's $t$ | ARX(1,0) | GJR-GARCH(1,1) | 64.0988 | -114.1975 | -82.4204 |
| Student's $t$ | ARX(1,0) | GJR-GARCH(1,2) | 64.1014 | -112.2027 | -75.8861 |
| Student's $t$ | ARX(1,0) | GJR-GARCH(2,1) | 64.3622 | -110.7244 | -69.8682 |
| Student's $t$ | ARX(1,0) | GJR-GARCH(2,2) | 64.8382 | -109.6763 | -64.2805 |
| Student's $t$ | ARX(1,0) | EGARCH(1,1) | 60.0210 | -106.0420 | -74.2649 |
| Student's $t$ | ARX(1,0) | EGARCH(1,2) | 60.0445 | -104.0890 | -67.7723 |
| Student's $t$ | ARX(1,0) | EGARCH(2,1) | 61.1600 | -104.3199 | -63.4636 |
| Student's $t$ | ARX(1,0) | EGARCH(2,2) | 62.8130 | -105.6261 | -60.2302 |
| Student's $t$ | ARX(1,1) | GARCH(1,1) | 97.7611 | -181.5223 | -149.7452 |
| Student's $t$ | ARX(1,1) | GARCH(1,2) | 97.7611 | -179.5223 | -143.2056 |
| Student's $t$ | ARX(1,1) | GARCH(2,1) | 97.8253 | -179.6507 | -143.3340 |
| Student's $t$ | ARX(1,1) | GARCH(2,2) | 97.8253 | -177.6507 | -136.7944 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | 100.1943 | -184.3886 | -148.0719 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,2) | 100.1943 | -182.3886 | -141.5323 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,1) | 100.7451 | -181.4902 | -136.0943 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,2) | 101.0645 | -180.1291 | -130.1936 |
| Student's $t$ | ARX(1,1) | EGARCH(1,1) | 96.3321 | -176.6642 | -140.3475 |
| Student's $t$ | ARX(1,1) | EGARCH(1,2) | 96.3432 | -174.6863 | -133.8300 |
| Student's $t$ | ARX(1,1) | EGARCH(2,1) | 97.7896 | -175.5792 | -130.1833 |
| Student's $t$ | ARX(1,1) | EGARCH(2,2) | 98.8160 | -175.6321 | -125.6966 |
| Student's $t$ | ARX(2,0) | GARCH(1,1) | 63.7151 | -113.4301 | -81.6530 |
| Student's $t$ | ARX(2,0) | GARCH(1,2) | 63.7164 | -111.4328 | -75.1161 |
| Student's $t$ | ARX(2,0) | GARCH(2,1) | 63.7151 | -111.4301 | -75.1134 |
| Student's $t$ | ARX(2,0) | GARCH(2,2) | 63.9064 | -109.8128 | -68.9565 |
| Student's $t$ | ARX(2,0) | GJR-GARCH(1,1) | 65.5332 | -115.0663 | -78.7496 |
| Student's $t$ | ARX(2,0) | GJR-GARCH(1,2) | 65.5340 | -113.0681 | -72.2118 |
| Student's $t$ | ARX(2,0) | GJR-GARCH(2,1) | 65.7262 | -111.4524 | -66.0565 |
| Student's $t$ | ARX(2,0) | GJR-GARCH(2,2) | 65.9993 | -109.9985 | -60.0631 |
| Student's $t$ | ARX(2,0) | EGARCH(1,1) | 61.2110 | -106.4220 | -70.1053 |
| Student's $t$ | ARX(2,0) | EGARCH(1,2) | 61.2386 | -104.4772 | -63.6209 |
| Student's $t$ | ARX(2,0) | EGARCH(2,1) | 62.0960 | -104.1920 | -58.7962 |
| Student's $t$ | ARX(2,0) | EGARCH(2,2) | 63.7914 | -105.5827 | -55.6473 |
| Student's $t$ | ARX(2,1) | GARCH(1,1) | 97.9755 | -179.9510 | -143.6343 |
| Student's $t$ | ARX(2,1) | GARCH(1,2) | 97.9755 | -177.9510 | -137.0947 |
| Student's $t$ | ARX(2,1) | GARCH(2,1) | 98.0343 | -178.0687 | -137.2124 |
| Student's $t$ | ARX(2,1) | GARCH(2,2) | 98.0343 | -176.0687 | -130.6728 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,1) | 100.7475 | -183.4950 | -142.6388 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,2) | 100.7475 | -181.4950 | -136.0992 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,1) | 101.3312 | -180.6625 | -130.7270 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,2) | 101.8068 | -179.6135 | -125.1385 |
| Student's $t$ | ARX(2,1) | EGARCH(1,1) | 96.9145 | -175.8291 | -134.9728 |
| Student's $t$ | ARX(2,1) | EGARCH(1,2) | 96.9385 | -173.8769 | -128.4811 |
| Student's $t$ | ARX(2,1) | EGARCH(2,1) | 98.6009 | -175.2018 | -125.2663 |
| Student's $t$ | ARX(2,1) | EGARCH(2,2) | 99.5521 | -175.1043 | -120.6292 |
| Student's $t$ | ARX(2,2) | GARCH(1,1) | 98.3232 | -178.6463 | -137.7901 |
| Student's $t$ | ARX(2,2) | GARCH(1,2) | 98.3232 | -176.6463 | -131.2505 |
| Student's $t$ | ARX(2,2) | GARCH(2,1) | 98.3965 | -176.7929 | -131.3971 |
| Student's $t$ | ARX(2,2) | GARCH(2,2) | 98.3965 | -174.7930 | -124.8575 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,1) | 101.1346 | -182.2693 | -136.8734 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,2) | 101.1346 | -180.2693 | -130.3338 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,1) | 101.8440 | -179.6879 | -125.2129 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,2) | 102.3657 | -178.7315 | -119.7169 |
| Student's $t$ | ARX(2,2) | EGARCH(1,1) | 97.2669 | -174.5338 | -129.1379 |
| Student's $t$ | ARX(2,2) | EGARCH(1,2) | 97.2838 | -172.5675 | -122.6321 |
| Student's $t$ | ARX(2,2) | EGARCH(2,1) | 99.0761 | -174.1522 | -119.6772 |
| Student's $t$ | ARX(2,2) | EGARCH(2,2) | 100.0842 | -174.1684 | -115.1538 |
| Gaussian mixture | Constant | GARCH(1,1) | 39.4976 | -66.9951 | -39.7576 |
| Gaussian mixture | Constant | GARCH(1,2) | 40.9181 | -67.8363 | -36.0592 |
| Gaussian mixture | Constant | GARCH(2,1) | 39.4976 | -64.9951 | -33.2180 |
| Gaussian mixture | Constant | GARCH(2,2) | 40.9181 | -65.8363 | -29.5196 |
| Gaussian mixture | Constant | GJR-GARCH(1,1) | 42.1330 | -70.2660 | -38.4889 |
| Gaussian mixture | Constant | GJR-GARCH(1,2) | 43.7196 | -71.4392 | -35.1225 |
| Gaussian mixture | Constant | GJR-GARCH(2,1) | 42.1330 | -66.2661 | -25.4098 |
| Gaussian mixture | Constant | GJR-GARCH(2,2) | 43.8628 | -67.7257 | -22.3298 |
| Gaussian mixture | Constant | EGARCH(1,1) | 40.1126 | -66.2252 | -34.4481 |
| Gaussian mixture | Constant | EGARCH(1,2) | 41.8257 | -67.6514 | -31.3347 |
| Gaussian mixture | Constant | EGARCH(2,1) | 44.0248 | -70.0495 | -29.1933 |
| Gaussian mixture | Constant | EGARCH(2,2) | 44.0248 | -68.0495 | -22.6537 |
| Gaussian mixture | ARX(1,0) | GARCH(1,1) | 62.3967 | -108.7934 | -72.4767 |
| Gaussian mixture | ARX(1,0) | GARCH(1,2) | 62.4921 | -106.9843 | -66.1280 |
| Gaussian mixture | ARX(1,0) | GARCH(2,1) | 62.3967 | -106.7934 | -65.9372 |
| Gaussian mixture | ARX(1,0) | GARCH(2,2) | 62.8621 | -105.7242 | -60.3283 |
| Gaussian mixture | ARX(1,0) | GJR-GARCH(1,1) | 65.5185 | -113.0369 | -72.1807 |
| Gaussian mixture | ARX(1,0) | GJR-GARCH(1,2) | 65.5185 | -111.0369 | -65.6411 |
| Gaussian mixture | ARX(1,0) | GJR-GARCH(2,1) | 65.8710 | -109.7421 | -59.8066 |
| Gaussian mixture | ARX(1,0) | GJR-GARCH(2,2) | 66.4016 | -108.8032 | -54.3282 |
| Gaussian mixture | ARX(1,0) | EGARCH(1,1) | 61.8165 | -105.6331 | -64.7768 |
| Gaussian mixture | ARX(1,0) | EGARCH(1,2) | 61.8165 | -103.6331 | -58.2372 |
| Gaussian mixture | ARX(1,0) | EGARCH(2,1) | 63.0514 | -104.1029 | -54.1674 |
| Gaussian mixture | ARX(1,0) | EGARCH(2,2) | 64.2339 | -104.4677 | -49.9927 |
| Gaussian mixture | ARX(1,1) | GARCH(1,1) | 99.6274 | -181.2549 | -140.3986 |
| Gaussian mixture | ARX(1,1) | GARCH(1,2) | 99.6274 | -179.2549 | -133.8590 |
| Gaussian mixture | ARX(1,1) | GARCH(2,1) | 99.6583 | -179.3166 | -133.9208 |
| Gaussian mixture | ARX(1,1) | GARCH(2,2) | 99.6732 | -177.3463 | -127.4109 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,1) | 102.2136 | -184.4273 | -139.0314 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,2) | 102.2136 | -182.4273 | -132.4918 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,1) | 102.7267 | -181.4534 | -126.9784 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,2) | 102.9598 | -179.9196 | -120.9050 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,1) | 99.3683 | -178.7365 | -133.3407 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,2) | 99.3683 | -176.7365 | -126.8011 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,1) | 100.5121 | -177.0242 | -122.5492 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,2) | 101.0997 | -176.1993 | -117.1847 |
| Gaussian mixture | ARX(2,0) | GARCH(1,1) | 65.3123 | -112.6247 | -71.7684 |
| Gaussian mixture | ARX(2,0) | GARCH(1,2) | 65.3263 | -110.6525 | -65.2567 |
| Gaussian mixture | ARX(2,0) | GARCH(2,1) | 65.3123 | -110.6247 | -65.2288 |
| Gaussian mixture | ARX(2,0) | GARCH(2,2) | 65.5856 | -109.1712 | -59.2358 |
| Gaussian mixture | ARX(2,0) | GJR-GARCH(1,1) | 66.6826 | -113.3653 | -67.9694 |
| Gaussian mixture | ARX(2,0) | GJR-GARCH(1,2) | 66.6891 | -111.3782 | -61.4427 |
| Gaussian mixture | ARX(2,0) | GJR-GARCH(2,1) | 66.9840 | -109.9680 | -55.4929 |
| Gaussian mixture | ARX(2,0) | GJR-GARCH(2,2) | 67.4107 | -108.8215 | -49.8069 |
| Gaussian mixture | ARX(2,0) | EGARCH(1,1) | 62.7156 | -105.4312 | -60.0353 |
| Gaussian mixture | ARX(2,0) | EGARCH(1,2) | 62.7218 | -103.4435 | -53.5081 |
| Gaussian mixture | ARX(2,0) | EGARCH(2,1) | 63.9110 | -103.8220 | -49.3470 |
| Gaussian mixture | ARX(2,0) | EGARCH(2,2) | 65.2566 | -104.5131 | -45.4985 |
| Gaussian mixture | ARX(2,1) | GARCH(1,1) | 99.9402 | -179.8804 | -134.4845 |
| Gaussian mixture | ARX(2,1) | GARCH(1,2) | 99.9402 | -177.8804 | -127.9449 |
| Gaussian mixture | ARX(2,1) | GARCH(2,1) | 99.9697 | -177.9393 | -128.0039 |
| Gaussian mixture | ARX(2,1) | GARCH(2,2) | 99.9802 | -175.9604 | -121.4853 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,1) | 103.0966 | -184.1932 | -134.2577 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,2) | 103.0966 | -182.1932 | -127.7181 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,1) | 103.6097 | -181.2194 | -122.2048 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,2) | 103.9211 | -179.8422 | -116.2880 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,1) | 100.3623 | -178.7245 | -128.7891 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,2) | 100.3623 | -176.7245 | -122.2495 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,1) | 101.5850 | -177.1700 | -118.1554 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,2) | 102.0046 | -176.0093 | -112.4551 |
| Gaussian mixture | ARX(2,2) | GARCH(1,1) | 100.1470 | -178.2941 | -128.3586 |
| Gaussian mixture | ARX(2,2) | GARCH(1,2) | 100.1470 | -176.2941 | -121.8190 |
| Gaussian mixture | ARX(2,2) | GARCH(2,1) | 100.1789 | -176.3579 | -121.8829 |
| Gaussian mixture | ARX(2,2) | GARCH(2,2) | 100.1938 | -174.3876 | -115.3730 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,1) | 103.4022 | -182.8044 | -128.3294 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,2) | 103.4022 | -180.8044 | -121.7898 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,1) | 104.0077 | -180.0155 | -116.4612 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,2) | 104.3842 | -178.7684 | -110.6746 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,1) | 100.6554 | -177.3108 | -122.8357 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,2) | 100.6554 | -175.3108 | -116.2961 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,1) | 101.9673 | -175.9346 | -112.3804 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,2) | 102.4488 | -174.8975 | -106.8037 |

## Best Models by Criterion and Volatility Family

For each criterion and each volatility family, the selected model is the best across mean-process choices, orders, and distributions using stable & successful fits.
The main result tables contain 216 reported model combinations.

### AIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M061` | Student's $t$ | ARX(1,1) | GARCH(1,1) | 97.7611 | -181.5223 | -149.7452 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $\sigma_t^2$ | 0.0231 | 0.0419 | 0.1696 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0141 +0.4398\,\pi_t +0.5770\,SPF_t
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0060 +0.2544\,u_{t-1}^2 +0.6693\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.9892 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0141 | 0.0180 | 0.7828 | 0.4337 |
| $\rho_1$ | 0.4398 | 0.0452 | 9.7379 | 0.0000 |
| $\phi_1$ | 0.5770 | 0.0761 | 7.5839 | 0.0000 |
| $\omega$ | 0.0060 | 0.0022 | 2.6762 | 0.0074 |
| $\alpha_1$ | 0.2544 | 0.0665 | 3.8256 | 0.0001 |
| $\beta_1$ | 0.6693 | 0.0679 | 9.8562 | 0.0000 |
| $\nu$ | 4.9892 | 0.8263 | 6.0380 | 0.0000 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M110` | Gaussian mixture | ARX(1,1) | GJR-GARCH(1,1) | 102.2136 | -184.4273 | -139.0314 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $\sigma_t^2$ | 0.0224 | 0.0412 | 0.1683 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0206 +0.4274\,\pi_t +0.5784\,SPF_t
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0049 +0.2932\,(u_{t-1}^+)^2 +0.1221\,(u_{t-1}^-)^2 +0.7211\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $p_1$ | 0.1810 |
| $\mu_1$ | -0.0520 |
| $\sigma_1^2$ | 3.0052 |
| $p_2$ | 0.8190 |
| $\mu_2$ | 0.0115 |
| $\sigma_2^2$ | 0.5560 |

For GJR-GARCH, `arch` estimates the indicator coefficient $\gamma_i-\alpha_i$ (row $\gamma_i-\alpha_i$). The negative-shock coefficient of the README specification is $\gamma_i = \alpha_i + (\gamma_i-\alpha_i)$ (row $\gamma_i$); its standard error uses the delta method on the same robust covariance matrix, $\operatorname{Var}(\hat\gamma_i) = \operatorname{Var}(\hat\alpha_i) + \operatorname{Var}(\widehat{\gamma_i-\alpha_i}) + 2\operatorname{Cov}(\hat\alpha_i, \widehat{\gamma_i-\alpha_i})$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0206 | 0.0186 | 1.1050 | 0.2692 |
| $\rho_1$ | 0.4274 | 0.0429 | 9.9551 | 0.0000 |
| $\phi_1$ | 0.5784 | 0.0739 | 7.8286 | 0.0000 |
| $\omega$ | 0.0049 | 0.0019 | 2.5434 | 0.0110 |
| $\alpha_1$ | 0.2932 | 0.0709 | 4.1367 | 0.0000 |
| $\gamma_1-\alpha_1$ | -0.1711 | 0.0674 | -2.5410 | 0.0111 |
| $\gamma_1$ | 0.1221 | 0.0629 | 1.9407 | 0.0523 |
| $\beta_1$ | 0.7211 | 0.0711 | 10.1453 | 0.0000 |
| $p_1$ | 0.1810 | 0.0662 | 2.7357 | 0.0062 |
| $\mu_1$ | -0.0520 | 0.1802 | -0.2888 | 0.7728 |
| $\sigma_1^2$ | 3.0052 | 0.6970 | 4.3118 | 0.0000 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M111` | Gaussian mixture | ARX(1,1) | EGARCH(1,1) | 99.3683 | -178.7365 | -133.3407 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $\sigma_t^2$ | 0.0213 | 0.0454 | 0.1454 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0171 +0.4241\,\pi_t +0.6006\,SPF_t
$$

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.2692 +0.3475\,(|z_{t-1}|-\sqrt{2/\pi}) +0.0680\,z_{t-1} +0.9039\,\ln\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $p_1$ | 0.8328 |
| $\mu_1$ | 0.0096 |
| $\sigma_1^2$ | 0.5493 |
| $p_2$ | 0.1672 |
| $\mu_2$ | -0.0480 |
| $\sigma_2^2$ | 3.2426 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0171 | 0.0016 | 10.8808 | 0.0000 |
| $\rho_1$ | 0.4241 | 0.0111 | 38.1959 | 0.0000 |
| $\phi_1$ | 0.6006 | 0.0116 | 51.7610 | 0.0000 |
| $\omega$ | -0.2692 | 0.1223 | -2.2016 | 0.0277 |
| $\alpha_1$ | 0.3475 | 0.0863 | 4.0252 | 0.0001 |
| $\gamma_1$ | 0.0680 | 0.0328 | 2.0714 | 0.0383 |
| $\beta_1$ | 0.9039 | 0.0412 | 21.9564 | 0.0000 |
| $p_1$ | 0.8328 | 0.0532 | 15.6688 | 0.0000 |
| $\mu_1$ | 0.0096 | 0.0330 | 0.2916 | 0.7706 |
| $\sigma_1^2$ | 0.5493 | 0.0654 | 8.3988 | 0.0000 |

### BIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M061` | Student's $t$ | ARX(1,1) | GARCH(1,1) | 97.7611 | -181.5223 | -149.7452 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $\sigma_t^2$ | 0.0231 | 0.0419 | 0.1696 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0141 +0.4398\,\pi_t +0.5770\,SPF_t
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0060 +0.2544\,u_{t-1}^2 +0.6693\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.9892 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0141 | 0.0180 | 0.7828 | 0.4337 |
| $\rho_1$ | 0.4398 | 0.0452 | 9.7379 | 0.0000 |
| $\phi_1$ | 0.5770 | 0.0761 | 7.5839 | 0.0000 |
| $\omega$ | 0.0060 | 0.0022 | 2.6762 | 0.0074 |
| $\alpha_1$ | 0.2544 | 0.0665 | 3.8256 | 0.0001 |
| $\beta_1$ | 0.6693 | 0.0679 | 9.8562 | 0.0000 |
| $\nu$ | 4.9892 | 0.8263 | 6.0380 | 0.0000 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M062` | Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | 100.1943 | -184.3886 | -148.0719 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $\sigma_t^2$ | 0.0224 | 0.0417 | 0.1667 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0214 +0.4302\,\pi_t +0.5699\,SPF_t
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0053 +0.3145\,(u_{t-1}^+)^2 +0.1458\,(u_{t-1}^-)^2 +0.6995\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.9530 |

For GJR-GARCH, `arch` estimates the indicator coefficient $\gamma_i-\alpha_i$ (row $\gamma_i-\alpha_i$). The negative-shock coefficient of the README specification is $\gamma_i = \alpha_i + (\gamma_i-\alpha_i)$ (row $\gamma_i$); its standard error uses the delta method on the same robust covariance matrix, $\operatorname{Var}(\hat\gamma_i) = \operatorname{Var}(\hat\alpha_i) + \operatorname{Var}(\widehat{\gamma_i-\alpha_i}) + 2\operatorname{Cov}(\hat\alpha_i, \widehat{\gamma_i-\alpha_i})$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0214 | 0.0181 | 1.1817 | 0.2373 |
| $\rho_1$ | 0.4302 | 0.0435 | 9.8960 | 0.0000 |
| $\phi_1$ | 0.5699 | 0.0739 | 7.7097 | 0.0000 |
| $\omega$ | 0.0053 | 0.0018 | 2.8957 | 0.0038 |
| $\alpha_1$ | 0.3145 | 0.0747 | 4.2125 | 0.0000 |
| $\gamma_1-\alpha_1$ | -0.1687 | 0.0690 | -2.4462 | 0.0144 |
| $\gamma_1$ | 0.1458 | 0.0718 | 2.0305 | 0.0423 |
| $\beta_1$ | 0.6995 | 0.0644 | 10.8698 | 0.0000 |
| $\nu$ | 4.9530 | 0.8237 | 6.0127 | 0.0000 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M063` | Student's $t$ | ARX(1,1) | EGARCH(1,1) | 96.3321 | -176.6642 | -140.3475 |

Empirical path quantiles:

| Series | 5% | Median | 95% |
|---|---:|---:|---:|
| $\sigma_t^2$ | 0.0217 | 0.0457 | 0.1428 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0175 +0.4290\,\pi_t +0.5919\,SPF_t
$$

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.3052 +0.3805\,(|z_{t-1}|-\sqrt{2/\pi}) +0.0628\,z_{t-1} +0.8904\,\ln\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.7151 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0175 | 0.0164 | 1.0683 | 0.2854 |
| $\rho_1$ | 0.4290 | 0.0424 | 10.1097 | 0.0000 |
| $\phi_1$ | 0.5919 | 0.0682 | 8.6724 | 0.0000 |
| $\omega$ | -0.3052 | 0.1433 | -2.1295 | 0.0332 |
| $\alpha_1$ | 0.3805 | 0.1171 | 3.2493 | 0.0012 |
| $\gamma_1$ | 0.0628 | 0.0401 | 1.5680 | 0.1169 |
| $\beta_1$ | 0.8904 | 0.0493 | 18.0473 | 0.0000 |
| $\nu$ | 4.7151 | 0.7452 | 6.3272 | 0.0000 |


