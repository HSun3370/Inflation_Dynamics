```{raw:typst}
#set page(margin: auto)
```

# Model Selection Report

Effective sample: **1969Q2--2026Q3** (**230 observations**, `Aggregate_CPI_inflation_Quarterly.pkl`, data vintage 2026-10-20).

## Parameter Ranges and Random Starts

All models are estimated with `arch` 7.0.0, which maximizes the likelihood by SLSQP subject to the bounds and linear constraints in the second column of the table below. `arch` imposes no bounds on mean-process coefficients or on the EGARCH coefficients $\alpha_i$ and $\gamma_j$. If a supplied starting vector violates any bound or constraint, `arch` silently replaces the whole vector with its own default start. Each model combination uses 5000 starts. Start 1 is the `arch` default (Gaussian mixture: the Student's $t$ estimates plus $(p_1,\mu_1,\sigma_1^2)=(0.3,0.1,1)$ when that vector is feasible, otherwise the `arch` default); starts 2--5000 are random draws from the third column. Every random draw lies inside `arch`'s feasible set and inside the stationary region (volatility persistence and AR roots below $1-10^{-6}$ in modulus), so `arch` uses every random start as drawn.

$\bar v$ denotes the mean squared residual at `arch`'s mean starting values (OLS residuals for ARX means, $\pi_t-SPF_t$ for the Constant mean): Constant 0.4419, ARX(1,1) 0.3856, ARX(2,1) 0.3834, ARX(2,2) 0.3832.

| Parameter | `arch` bound / constraint | Random-start draw |
|---|---|---|
| **Mean process** | | |
| $\rho_1$ in ARX(1,·) | none | $U(-1,1)$ (AR stationarity) |
| $(\rho_1,\rho_2)$ in ARX(2,·) | none | uniform on the stationarity triangle $\rho_2>-1$, $\lvert\rho_1\rvert<1-\rho_2$ |
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

- *Uniform on a region* means a uniform draw over the whole region: Dirichlet(1,…,1) weights for the volatility simplices, rejection sampling for the AR triangle and for `arch`'s caps. *Log-uniform* draws are uniform in the logarithm of the parameter.
- Compared with the stationarity conditions in the README, `arch`'s feasible set is the same for GARCH. For GJR-GARCH, `arch` also caps $\alpha_i\le1$, although stationarity alone allows $\alpha_i<2$ when $\gamma_i$ is small. For EGARCH, `arch` requires $\beta_k\ge0$, under which $\sum\beta_k<1$ is the stationarity condition; $\omega$ is also bounded. The mixture bounds come from this repository's `MixNormal` class. $\alpha_i$ and $\gamma_j$ in EGARCH are unbounded in `arch`; the reported EGARCH estimates have $\max|\alpha_i|=0.8035$ and $\max|\gamma_j|=0.4358$, well inside the draw range $[-2,2]$.
- Selection: for each combination the reported estimate has the highest log likelihood among *admissible* starts, which must pass five checks. (1) SLSQP reports success and `arch` returns convergence flag 0. (2) The first-order (KKT) conditions hold: the numerical gradient of the log likelihood, net of non-negative multipliers on the active bounds and constraints, is at most 0.005$\cdot T$ in absolute value. When a start that would become the best fails this check, SLSQP is restarted from its own solution up to 5 times, because SLSQP sometimes stops just short of a genuine maximum. Points on extremely steep ridges, where SLSQP reports success with gradients of $10^1$--$10^6$, do not move and stay rejected. (3) Volatility is stationary. (4) The AR roots lie inside the unit circle. (5) For the Gaussian mixture, the component variances satisfy $\min(\sigma_1^2,\sigma_2^2)/\max(\sigma_1^2,\sigma_2^2)\ge0.01$. This Hathaway-type restriction keeps the mixture likelihood bounded; without it the best fits place a component with variance $10^{-6}$ on one or two observations. If no start passes, the best converged start that satisfies the first-order conditions is reported and flagged in the CSV `selection_status` column. `best_loglik_hits` in the summary CSV counts the admissible starts that came within 0.01 of the reported log likelihood, and `params_at_arch_bound` lists parameters of the reported estimate that sit on an `arch` bound.
- In 35 of 144 combinations, a converged fit that satisfies the first-order conditions but fails stationarity or the mixture rule has a higher log likelihood than the reported estimate (largest difference 20.8965). For these combinations the likelihood peaks on the stationarity boundary or at a degenerate mixture, and the reported estimate is the best admissible local maximum. The CSV column `best_inadmissible_loglikelihood` records the higher value.
- In this run, 83 of 144 reported estimates have at least one parameter on an `arch` bound; parameters on a bound, with counts: beta (92), alpha (3). Standard errors of estimates on a bound or an active constraint (for example $\beta_k=0$, or a GJR $\gamma_i=0$) are not asymptotically normal and should be read with caution.

## Model Selection

For each model combination, the reported row uses the highest log-likelihood estimate that passes all admissibility checks (convergence, first-order conditions, volatility and mean stationarity, and the mixture variance ratio) across 5000 starts. If no start passes all checks, the reported row uses the best converged start that satisfies the first-order conditions and is flagged in the CSV `selection_status` column.

**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**[^garch-aic-sample-note]

| Distribution | GARCH Mean | GARCH Vol | GARCH AIC | GJR Mean | GJR Vol | GJR AIC | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | Constant | (2,1) | 398.6487 | ARX(1,1) | (2,1) | 383.4043 | ARX(1,1) | (2,1) | 387.0269 |
| Student's $t$ | ARX(2,1) | (2,1) | 372.7757 | ARX(2,1) | (2,1) | <span style="color:red">366.7843</span> | ARX(1,1) | (2,1) | 368.2259 |
| Gaussian mixture | ARX(2,1) | (2,1) | 374.8428 | ARX(2,1) | (2,1) | 367.8431 | ARX(1,1) | (2,1) | 367.3223 |

[^garch-aic-sample-note]: These numbers differ from earlier GARCH tables because earlier code used inconsistent effective sample sizes across model settings due to a sample-trimming error. The correction is discussed in the [Data Summary effective-sample section](../../DataSummary/README.md#effective-sample). The current estimates use a common effective sample with **230 observations**.

**Table 3: Model selection by BIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH BIC | GJR Mean | GJR Vol | GJR BIC | EGARCH Mean | EGARCH Vol | EGARCH BIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | Constant | (2,1) | 412.4011 | Constant | (2,1) | 408.4957 | Constant | (2,1) | 411.0244 |
| Student's $t$ | Constant | (2,1) | <span style="color:red">396.0444</span> | Constant | (1,1) | 396.6951 | Constant | (1,1) | 396.0615 |
| Gaussian mixture | ARX(1,1) | (1,1) | 410.4860 | ARX(1,1) | (1,1) | 406.2031 | Constant | (1,1) | 397.2711 |

## All Model Results

| Distribution | Mean Process | Volatility Process | Best LogLik | AIC | BIC |
|---|---|---|---:|---:|---:|
| Normal | Constant | GARCH(1,1) | -205.0243 | 416.0486 | 426.3628 |
| Normal | Constant | GARCH(1,2) | -205.0243 | 418.0486 | 431.8009 |
| Normal | Constant | GARCH(2,1) | -195.3244 | 398.6487 | 412.4011 |
| Normal | Constant | GARCH(2,2) | -195.3244 | 400.6487 | 417.8391 |
| Normal | Constant | GJR-GARCH(1,1) | -197.4180 | 402.8360 | 416.5883 |
| Normal | Constant | GJR-GARCH(1,2) | -197.4180 | 404.8360 | 422.0263 |
| Normal | Constant | GJR-GARCH(2,1) | -187.9336 | 387.8672 | 408.4957 |
| Normal | Constant | GJR-GARCH(2,2) | -187.9336 | 389.8672 | 413.9337 |
| Normal | Constant | EGARCH(1,1) | -196.1155 | 400.2311 | 413.9834 |
| Normal | Constant | EGARCH(1,2) | -196.1155 | 402.2311 | 419.4215 |
| Normal | Constant | EGARCH(2,1) | -189.1979 | 390.3959 | 411.0244 |
| Normal | Constant | EGARCH(2,2) | -189.1979 | 392.3959 | 416.4624 |
| Normal | ARX(1,1) | GARCH(1,1) | -198.8654 | 409.7308 | 430.3593 |
| Normal | ARX(1,1) | GARCH(1,2) | -198.8654 | 411.7308 | 435.7974 |
| Normal | ARX(1,1) | GARCH(2,1) | -192.4791 | 398.9581 | 423.0247 |
| Normal | ARX(1,1) | GARCH(2,2) | -192.4791 | 400.9581 | 428.4628 |
| Normal | ARX(1,1) | GJR-GARCH(1,1) | -190.7751 | 395.5502 | 419.6167 |
| Normal | ARX(1,1) | GJR-GARCH(1,2) | -190.7751 | 397.5502 | 425.0548 |
| Normal | ARX(1,1) | GJR-GARCH(2,1) | -182.7022 | 383.4043 | 414.3471 |
| Normal | ARX(1,1) | GJR-GARCH(2,2) | -182.7022 | 385.4043 | 419.7851 |
| Normal | ARX(1,1) | EGARCH(1,1) | -190.9195 | 395.8390 | 419.9056 |
| Normal | ARX(1,1) | EGARCH(1,2) | -190.9195 | 397.8390 | 425.3437 |
| Normal | ARX(1,1) | EGARCH(2,1) | -184.5134 | 387.0269 | 417.9696 |
| Normal | ARX(1,1) | EGARCH(2,2) | -184.5134 | 389.0269 | 423.4077 |
| Normal | ARX(2,1) | GARCH(1,1) | -196.5713 | 407.1426 | 431.2092 |
| Normal | ARX(2,1) | GARCH(1,2) | -196.5713 | 409.1426 | 436.6473 |
| Normal | ARX(2,1) | GARCH(2,1) | -191.4640 | 398.9281 | 426.4327 |
| Normal | ARX(2,1) | GARCH(2,2) | -191.4640 | 400.9281 | 431.8708 |
| Normal | ARX(2,1) | GJR-GARCH(1,1) | -190.7002 | 397.4005 | 424.9051 |
| Normal | ARX(2,1) | GJR-GARCH(1,2) | -190.7002 | 399.4005 | 430.3432 |
| Normal | ARX(2,1) | GJR-GARCH(2,1) | -181.7997 | 383.5993 | 417.9801 |
| Normal | ARX(2,1) | GJR-GARCH(2,2) | -181.7997 | 385.5993 | 423.4182 |
| Normal | ARX(2,1) | EGARCH(1,1) | -190.7499 | 397.4998 | 425.0045 |
| Normal | ARX(2,1) | EGARCH(1,2) | -190.7499 | 399.4998 | 430.4426 |
| Normal | ARX(2,1) | EGARCH(2,1) | -184.1546 | 388.3092 | 422.6900 |
| Normal | ARX(2,1) | EGARCH(2,2) | -184.1546 | 390.3092 | 428.1281 |
| Normal | ARX(2,2) | GARCH(1,1) | -196.5596 | 409.1191 | 436.6238 |
| Normal | ARX(2,2) | GARCH(1,2) | -196.5596 | 411.1191 | 442.0619 |
| Normal | ARX(2,2) | GARCH(2,1) | -191.4622 | 400.9244 | 431.8672 |
| Normal | ARX(2,2) | GARCH(2,2) | -191.4622 | 402.9244 | 437.3052 |
| Normal | ARX(2,2) | GJR-GARCH(1,1) | -190.6858 | 399.3716 | 430.3143 |
| Normal | ARX(2,2) | GJR-GARCH(1,2) | -190.6858 | 401.3716 | 435.7524 |
| Normal | ARX(2,2) | GJR-GARCH(2,1) | -181.7659 | 385.5318 | 423.3507 |
| Normal | ARX(2,2) | GJR-GARCH(2,2) | -181.7659 | 387.5318 | 428.7887 |
| Normal | ARX(2,2) | EGARCH(1,1) | -190.7003 | 399.4006 | 430.3433 |
| Normal | ARX(2,2) | EGARCH(1,2) | -190.7003 | 401.4006 | 435.7814 |
| Normal | ARX(2,2) | EGARCH(2,1) | -183.9822 | 389.9643 | 427.7832 |
| Normal | ARX(2,2) | EGARCH(2,2) | -183.9822 | 391.9643 | 433.2213 |
| Student's $t$ | Constant | GARCH(1,1) | -187.4076 | 382.8151 | 396.5674 |
| Student's $t$ | Constant | GARCH(1,2) | -187.4076 | 384.8151 | 402.0055 |
| Student's $t$ | Constant | GARCH(2,1) | -184.4270 | 378.8540 | 396.0444 |
| Student's $t$ | Constant | GARCH(2,2) | -184.4270 | 380.8540 | 401.4825 |
| Student's $t$ | Constant | GJR-GARCH(1,1) | -184.7523 | 379.5047 | 396.6951 |
| Student's $t$ | Constant | GJR-GARCH(1,2) | -184.7523 | 381.5047 | 402.1331 |
| Student's $t$ | Constant | GJR-GARCH(2,1) | -179.7242 | 373.4483 | 397.5149 |
| Student's $t$ | Constant | GJR-GARCH(2,2) | -179.7242 | 375.4483 | 402.9530 |
| Student's $t$ | Constant | EGARCH(1,1) | -184.4356 | 378.8711 | 396.0615 |
| Student's $t$ | Constant | EGARCH(1,2) | -184.4356 | 380.8711 | 401.4996 |
| Student's $t$ | Constant | EGARCH(2,1) | -181.1143 | 376.2285 | 400.2951 |
| Student's $t$ | Constant | EGARCH(2,2) | -181.1143 | 378.2285 | 405.7332 |
| Student's $t$ | ARX(1,1) | GARCH(1,1) | -180.9397 | 375.8794 | 399.9459 |
| Student's $t$ | ARX(1,1) | GARCH(1,2) | -180.9397 | 377.8794 | 405.3840 |
| Student's $t$ | ARX(1,1) | GARCH(2,1) | -179.1243 | 374.2485 | 401.7531 |
| Student's $t$ | ARX(1,1) | GARCH(2,2) | -179.1243 | 376.2485 | 407.1912 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | -177.1939 | 370.3878 | 397.8925 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,2) | -177.1939 | 372.3878 | 403.3305 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,1) | -173.4834 | 366.9668 | 401.3476 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,2) | -173.4834 | 368.9668 | 406.7857 |
| Student's $t$ | ARX(1,1) | EGARCH(1,1) | -177.0128 | 370.0257 | 397.5303 |
| Student's $t$ | ARX(1,1) | EGARCH(1,2) | -177.0128 | 372.0257 | 402.9684 |
| Student's $t$ | ARX(1,1) | EGARCH(2,1) | -174.1130 | 368.2259 | 402.6067 |
| Student's $t$ | ARX(1,1) | EGARCH(2,2) | -174.1130 | 370.2259 | 408.0448 |
| Student's $t$ | ARX(2,1) | GARCH(1,1) | -178.6739 | 373.3478 | 400.8524 |
| Student's $t$ | ARX(2,1) | GARCH(1,2) | -178.6739 | 375.3478 | 406.2905 |
| Student's $t$ | ARX(2,1) | GARCH(2,1) | -177.3879 | 372.7757 | 403.7184 |
| Student's $t$ | ARX(2,1) | GARCH(2,2) | -177.3879 | 374.7757 | 409.1565 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,1) | -175.8260 | 369.6521 | 400.5948 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,2) | -175.8260 | 371.6521 | 406.0328 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,1) | -172.3922 | 366.7843 | 404.6032 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,2) | -172.3922 | 368.7843 | 410.0413 |
| Student's $t$ | ARX(2,1) | EGARCH(1,1) | -175.4947 | 368.9893 | 399.9321 |
| Student's $t$ | ARX(2,1) | EGARCH(1,2) | -175.4947 | 370.9893 | 405.3701 |
| Student's $t$ | ARX(2,1) | EGARCH(2,1) | -173.2017 | 368.4034 | 406.2222 |
| Student's $t$ | ARX(2,1) | EGARCH(2,2) | -173.2017 | 370.4034 | 411.6603 |
| Student's $t$ | ARX(2,2) | GARCH(1,1) | -178.6446 | 375.2892 | 406.2319 |
| Student's $t$ | ARX(2,2) | GARCH(1,2) | -178.6446 | 377.2892 | 411.6700 |
| Student's $t$ | ARX(2,2) | GARCH(2,1) | -177.3845 | 374.7690 | 409.1498 |
| Student's $t$ | ARX(2,2) | GARCH(2,2) | -177.3845 | 376.7690 | 414.5879 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,1) | -175.8162 | 371.6323 | 406.0131 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,2) | -175.8162 | 373.6323 | 411.4512 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,1) | -172.3677 | 368.7355 | 409.9924 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,2) | -172.3677 | 370.7355 | 415.4305 |
| Student's $t$ | ARX(2,2) | EGARCH(1,1) | -175.4167 | 370.8335 | 405.2143 |
| Student's $t$ | ARX(2,2) | EGARCH(1,2) | -175.4167 | 372.8335 | 410.6524 |
| Student's $t$ | ARX(2,2) | EGARCH(2,1) | -173.1102 | 370.2204 | 411.4773 |
| Student's $t$ | ARX(2,2) | EGARCH(2,2) | -173.1102 | 372.2204 | 416.9154 |
| Gaussian mixture | Constant | GARCH(1,1) | -203.9031 | 419.8061 | 440.4346 |
| Gaussian mixture | Constant | GARCH(1,2) | -203.9031 | 421.8061 | 445.8727 |
| Gaussian mixture | Constant | GARCH(2,1) | -187.7206 | 389.4412 | 413.5077 |
| Gaussian mixture | Constant | GARCH(2,2) | -188.1816 | 392.3632 | 419.8679 |
| Gaussian mixture | Constant | GJR-GARCH(1,1) | -188.0093 | 390.0186 | 414.0851 |
| Gaussian mixture | Constant | GJR-GARCH(1,2) | -183.8946 | 383.7891 | 411.2938 |
| Gaussian mixture | Constant | GJR-GARCH(2,1) | -181.6374 | 381.2747 | 412.2174 |
| Gaussian mixture | Constant | GJR-GARCH(2,2) | -181.5324 | 383.0647 | 417.4455 |
| Gaussian mixture | Constant | EGARCH(1,1) | -179.6023 | 373.2045 | 397.2711 |
| Gaussian mixture | Constant | EGARCH(1,2) | -179.6023 | 375.2045 | 402.7091 |
| Gaussian mixture | Constant | EGARCH(2,1) | -175.9761 | 369.9523 | 400.8950 |
| Gaussian mixture | Constant | EGARCH(2,2) | -175.9761 | 371.9523 | 406.3331 |
| Gaussian mixture | ARX(1,1) | GARCH(1,1) | -180.7717 | 379.5433 | 410.4860 |
| Gaussian mixture | ARX(1,1) | GARCH(1,2) | -180.6679 | 381.3359 | 415.7167 |
| Gaussian mixture | ARX(1,1) | GARCH(2,1) | -178.7627 | 377.5255 | 411.9063 |
| Gaussian mixture | ARX(1,1) | GARCH(2,2) | -178.7627 | 379.5255 | 417.3443 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,1) | -175.9112 | 371.8223 | 406.2031 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,2) | -175.9112 | 373.8223 | 411.6412 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,1) | -172.1980 | 368.3960 | 409.6529 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,2) | -172.1980 | 370.3960 | 415.0910 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,1) | -175.1383 | 370.2766 | 404.6574 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,2) | -175.1383 | 372.2766 | 410.0955 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,1) | -171.6612 | 367.3223 | 408.5793 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,2) | -171.6612 | 369.3223 | 414.0174 |
| Gaussian mixture | ARX(2,1) | GARCH(1,1) | -181.1226 | 382.2453 | 416.6261 |
| Gaussian mixture | ARX(2,1) | GARCH(1,2) | -181.1336 | 384.2671 | 422.0860 |
| Gaussian mixture | ARX(2,1) | GARCH(2,1) | -176.4214 | 374.8428 | 412.6617 |
| Gaussian mixture | ARX(2,1) | GARCH(2,2) | -176.3439 | 376.6877 | 417.9447 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,1) | -174.0267 | 370.0534 | 407.8723 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,2) | -174.0267 | 372.0534 | 413.3104 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,1) | -170.9216 | 367.8431 | 412.5381 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,2) | -170.9216 | 369.8431 | 417.9762 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,1) | -173.2599 | 368.5197 | 406.3386 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,2) | -173.2599 | 370.5197 | 411.7767 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,1) | -170.7391 | 367.4783 | 412.1733 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,2) | -170.7391 | 369.4783 | 417.6114 |
| Gaussian mixture | ARX(2,2) | GARCH(1,1) | -180.9837 | 383.9675 | 421.7864 |
| Gaussian mixture | ARX(2,2) | GARCH(1,2) | -180.8781 | 385.7561 | 427.0131 |
| Gaussian mixture | ARX(2,2) | GARCH(2,1) | -176.4107 | 376.8214 | 418.0783 |
| Gaussian mixture | ARX(2,2) | GARCH(2,2) | -176.4790 | 378.9579 | 423.6530 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,1) | -173.9754 | 371.9508 | 413.2078 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,2) | -173.9754 | 373.9508 | 418.6459 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,1) | -170.8306 | 369.6613 | 417.7944 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,2) | -170.8306 | 371.6613 | 423.2325 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,1) | -173.0583 | 370.1165 | 411.3735 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,2) | -173.0583 | 372.1165 | 416.8115 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,1) | -170.5547 | 369.1094 | 417.2425 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,2) | -170.5547 | 371.1094 | 422.6806 |

## Best Models by Criterion and Volatility Family

For each criterion and each volatility family, the selected model is the best across mean-process choices, orders, and distributions using stable & successful fits.
The main result tables contain 144 reported model combinations.

### AIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M079` | Student's $t$ | ARX(2,1) | GARCH(2,1) | -177.3879 | 372.7757 | 403.7184 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0837 +0.1889\,\pi_t +0.1445\,\pi_{t-1} +0.6223\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.1359 +0.4260\,u_{t-1}^2 +0.3835\,u_{t-2}^2 +0.0000\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.6335 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0837 | 0.0767 | 1.0923 | 0.2747 |
| $\rho_1$ | 0.1889 | 0.1290 | 1.4647 | 0.1430 |
| $\rho_2$ | 0.1445 | 0.2045 | 0.7064 | 0.4800 |
| $\phi_1$ | 0.6223 | 0.3646 | 1.7066 | 0.0879 |
| $\omega$ | 0.1359 | 0.2370 | 0.5737 | 0.5662 |
| $\alpha_1$ | 0.4260 | 0.2243 | 1.8996 | 0.0575 |
| $\alpha_2$ | 0.3835 | 0.8233 | 0.4658 | 0.6414 |
| $\beta_1$ | 0.0000 | 1.2322 | 0.0000 | 1.0000 |
| $\nu$ | 4.6335 | 1.5222 | 3.0440 | 0.0023 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M080` | Student's $t$ | ARX(2,1) | GJR-GARCH(2,1) | -172.3922 | 366.7843 | 404.6032 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0954 +0.1599\,\pi_t +0.1132\,\pi_{t-1} +0.6951\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.1095 +0.4132\,(u_{t-1}^+)^2 +0.2919\,(u_{t-1}^-)^2 +0.6659\,(u_{t-2}^+)^2 +0.0000\,(u_{t-2}^-)^2 +0.1121\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.5940 |

For GJR-GARCH, `arch` estimates the indicator coefficient $\gamma_i-\alpha_i$ (row $\gamma_i-\alpha_i$). The negative-shock coefficient of the README specification is $\gamma_i = \alpha_i + (\gamma_i-\alpha_i)$ (row $\gamma_i$); its standard error uses the delta method on the same robust covariance matrix, $\operatorname{Var}(\hat\gamma_i) = \operatorname{Var}(\hat\alpha_i) + \operatorname{Var}(\widehat{\gamma_i-\alpha_i}) + 2\operatorname{Cov}(\hat\alpha_i, \widehat{\gamma_i-\alpha_i})$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0954 | 0.0828 | 1.1515 | 0.2495 |
| $\rho_1$ | 0.1599 | 0.0975 | 1.6402 | 0.1010 |
| $\rho_2$ | 0.1132 | 0.0919 | 1.2319 | 0.2180 |
| $\phi_1$ | 0.6951 | 0.1627 | 4.2721 | 0.0000 |
| $\omega$ | 0.1095 | 0.0325 | 3.3727 | 0.0007 |
| $\alpha_1$ | 0.4132 | 0.2078 | 1.9884 | 0.0468 |
| $\gamma_1-\alpha_1$ | -0.1213 | 0.2572 | -0.4716 | 0.6372 |
| $\gamma_1$ | 0.2919 | 0.1956 | 1.4920 | 0.1357 |
| $\alpha_2$ | 0.6659 | 0.3560 | 1.8702 | 0.0614 |
| $\gamma_2-\alpha_2$ | -0.6659 | 0.3259 | -2.0434 | 0.0410 |
| $\gamma_2$ | 0.0000 | 0.0569 | 0.0000 | 1.0000 |
| $\beta_1$ | 0.1121 | 0.1123 | 0.9984 | 0.3181 |
| $\nu$ | 5.5940 | 2.0065 | 2.7879 | 0.0053 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M117` | Gaussian mixture | ARX(1,1) | EGARCH(2,1) | -171.6612 | 367.3223 | 408.5793 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0920 +0.1895\,\pi_t +0.7553\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.3800 +0.5800\,(|z_{t-1}|-\sqrt{2/\pi}) +0.0576\,(|z_{t-2}|-\sqrt{2/\pi}) +0.1120\,z_{t-1} +0.3138\,z_{t-2} +0.6530\,\ln\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $p_1$ | 0.9441 |
| $\mu_1$ | 0.0462 |
| $\sigma_1^2$ | 0.6376 |
| $p_2$ | 0.0559 |
| $\mu_2$ | -0.7808 |
| $\sigma_2^2$ | 6.4737 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0920 | 0.0589 | 1.5609 | 0.1185 |
| $\rho_1$ | 0.1895 | 0.0922 | 2.0555 | 0.0398 |
| $\phi_1$ | 0.7553 | 0.1014 | 7.4469 | 0.0000 |
| $\omega$ | -0.3800 | 0.1823 | -2.0849 | 0.0371 |
| $\alpha_1$ | 0.5800 | 0.1604 | 3.6160 | 0.0003 |
| $\gamma_1$ | 0.1120 | 0.1043 | 1.0729 | 0.2833 |
| $\alpha_2$ | 0.0576 | 0.2238 | 0.2575 | 0.7968 |
| $\gamma_2$ | 0.3138 | 0.1010 | 3.1077 | 0.0019 |
| $\beta_1$ | 0.6530 | 0.1106 | 5.9026 | 0.0000 |
| $p_1$ | 0.9441 | 0.0343 | 27.5063 | 0.0000 |
| $\mu_1$ | 0.0462 | 0.0566 | 0.8172 | 0.4138 |
| $\sigma_1^2$ | 0.6376 | 0.1578 | 4.0397 | 0.0001 |

### BIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M055` | Student's $t$ | Constant | GARCH(2,1) | -184.4270 | 378.8540 | 396.0444 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\hat{\sigma}_t^2 = 0.1150 +0.4305\,u_{t-1}^2 +0.5206\,u_{t-2}^2 +0.0000\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.9346 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.1150 | 0.0370 | 3.1060 | 0.0019 |
| $\alpha_1$ | 0.4305 | 0.1665 | 2.5854 | 0.0097 |
| $\alpha_2$ | 0.5206 | 0.2063 | 2.5230 | 0.0116 |
| $\beta_1$ | 0.0000 | 0.1143 | 0.0000 | 1.0000 |
| $\nu$ | 4.9346 | 1.6813 | 2.9350 | 0.0033 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M050` | Student's $t$ | Constant | GJR-GARCH(1,1) | -184.7523 | 379.5047 | 396.6951 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0741 +0.7922\,(u_{t-1}^+)^2 +0.1785\,(u_{t-1}^-)^2 +0.3779\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.7200 |

For GJR-GARCH, `arch` estimates the indicator coefficient $\gamma_i-\alpha_i$ (row $\gamma_i-\alpha_i$). The negative-shock coefficient of the README specification is $\gamma_i = \alpha_i + (\gamma_i-\alpha_i)$ (row $\gamma_i$); its standard error uses the delta method on the same robust covariance matrix, $\operatorname{Var}(\hat\gamma_i) = \operatorname{Var}(\hat\alpha_i) + \operatorname{Var}(\widehat{\gamma_i-\alpha_i}) + 2\operatorname{Cov}(\hat\alpha_i, \widehat{\gamma_i-\alpha_i})$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.0741 | 0.0288 | 2.5741 | 0.0100 |
| $\alpha_1$ | 0.7922 | 0.2362 | 3.3540 | 0.0008 |
| $\gamma_1-\alpha_1$ | -0.6136 | 0.2274 | -2.6984 | 0.0070 |
| $\gamma_1$ | 0.1785 | 0.1616 | 1.1049 | 0.2692 |
| $\beta_1$ | 0.3779 | 0.1226 | 3.0827 | 0.0021 |
| $\nu$ | 4.7200 | 1.5107 | 3.1243 | 0.0018 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M051` | Student's $t$ | Constant | EGARCH(1,1) | -184.4356 | 378.8711 | 396.0615 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.2784 +0.5524\,(|z_{t-1}|-\sqrt{2/\pi}) +0.2519\,z_{t-1} +0.7757\,\ln\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.0056 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | -0.2784 | 0.1080 | -2.5776 | 0.0100 |
| $\alpha_1$ | 0.5524 | 0.1274 | 4.3355 | 0.0000 |
| $\gamma_1$ | 0.2519 | 0.0771 | 3.2660 | 0.0011 |
| $\beta_1$ | 0.7757 | 0.0614 | 12.6381 | 0.0000 |
| $\nu$ | 5.0056 | 1.6780 | 2.9830 | 0.0029 |


