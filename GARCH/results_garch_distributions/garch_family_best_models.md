```{raw:typst}
#set page(margin: auto)
```

# Model Selection Report

For each model combination, the reported row uses the highest log-likelihood estimate that passes optimizer convergence and stationarity checks across 50 starts. If no start passes all checks, the reported row uses the best optimizer-converged attempted log likelihood and is flagged in the CSV `selection_status` column.

**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**[^garch-aic-sample-note]

| Distribution | GARCH Mean | GARCH Vol | GARCH AIC | GJR Mean | GJR Vol | GJR AIC | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(2,1) | (2,1) | 387.2314 | ARX(1,1) | (2,1) | 372.5785 | ARX(1,1) | (1,2) | 358.6843 |
| Student's $t$ | ARX(2,1) | (1,1) | 361.3528 | ARX(2,1) | (2,1) | 356.9953 | ARX(2,1) | (1,2) | <span style="color:red">351.8931</span> |
| Gaussian mixture | ARX(1,1) | (2,1) | 367.1233 | ARX(2,1) | (2,1) | 358.4230 | ARX(2,1) | (1,1) | 352.7338 |

[^garch-aic-sample-note]: These numbers differ from earlier GARCH tables because earlier code used inconsistent effective sample sizes across model settings due to a sample-trimming error. The correction is discussed in the [Data Summary effective-sample section](../../DataSummary/README.md#effective-sample). The current estimates use the common **1969Q2--2022Q4** sample with **215 observations**.

**Table 3: Model selection by BIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH BIC | GJR Mean | GJR Vol | GJR BIC | EGARCH Mean | EGARCH Vol | EGARCH BIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | Constant | (2,1) | 401.2826 | Constant | (2,1) | 396.7277 | ARX(1,1) | (1,2) | 385.6494 |
| Student's $t$ | Constant | (2,1) | 386.6231 | Constant | (1,1) | 384.2364 | Constant | (1,1) | <span style="color:red">383.5813</span> |
| Gaussian mixture | ARX(1,1) | (2,1) | 400.8297 | ARX(1,1) | (1,1) | 394.1515 | Constant | (1,1) | 385.2612 |

## Best Models by Criterion and Volatility Family

For each criterion and each volatility family, the selected model is the best across mean-process choices, orders, and distributions using stable & successful fits.
The main result tables contain 144 reported model combinations.

### AIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M073` | Student's $t$ | ARX(2,1) | GARCH(1,1) | -172.6764 | 361.3528 | 388.3179 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0876 +0.2637\,\pi_t +0.1716\,\pi_{t-1} +0.5096\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0725 +0.5202\,u_{t-1}^2 +0.4294\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.3024 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0876 | 0.0777 | 1.1279 | 0.2593 |
| $\rho_1$ | 0.2637 | 0.0912 | 2.8914 | 0.0038 |
| $\rho_2$ | 0.1716 | 0.0975 | 1.7600 | 0.0784 |
| $\phi_1$ | 0.5096 | 0.1487 | 3.4274 | 0.0006 |
| $\omega$ | 0.0725 | 0.0385 | 1.8834 | 0.0596 |
| $\alpha_1$ | 0.5202 | 0.2424 | 2.1465 | 0.0318 |
| $\beta_1$ | 0.4294 | 0.1425 | 3.0126 | 0.0026 |
| $\nu$ | 4.3024 | 1.4647 | 2.9373 | 0.0033 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M080` | Student's $t$ | ARX(2,1) | GJR-GARCH(2,1) | -167.4976 | 356.9953 | 394.0723 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0901 +0.2045\,\pi_t +0.1144\,\pi_{t-1} +0.6522\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.1096 +0.4005\,(u_{t-1}^+)^2 +0.2848\,(u_{t-1}^-)^2 +0.6415\,(u_{t-2}^+)^2 +0.0000\,(u_{t-2}^-)^2 +0.1559\,\sigma_{t-1}^2
$$



Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.3421 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0901 | 0.0838 | 1.0747 | 0.2825 |
| $\rho_1$ | 0.2045 | 0.0960 | 2.1292 | 0.0332 |
| $\rho_2$ | 0.1144 | 0.1043 | 1.0969 | 0.2727 |
| $\phi_1$ | 0.6522 | 0.1740 | 3.7481 | 0.0002 |
| $\omega$ | 0.1096 | 0.0397 | 2.7593 | 0.0058 |
| $\alpha_1$ | 0.4005 | 0.2194 | 1.8250 | 0.0680 |
| $\gamma_1-\alpha_1$ | -0.1157 | 0.2773 | -0.4173 | 0.6765 |
| $\alpha_2$ | 0.6415 | 0.3571 | 1.7962 | 0.0725 |
| $\gamma_2-\alpha_2$ | -0.6415 | 0.3194 | -2.0082 | 0.0446 |
| $\beta_1$ | 0.1559 | 0.1500 | 1.0399 | 0.2984 |
| $\nu$ | 5.3421 | 1.9529 | 2.7354 | 0.0062 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M078` | Student's $t$ | ARX(2,1) | EGARCH(1,2) | -165.9465 | 351.8931 | 385.5994 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0074 +0.1430\,\pi_t +0.1015\,\pi_{t-1} +0.8896\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.0740 -0.3421\,(|z_{t-1}|-\sqrt{2/\pi}) +0.2355\,z_{t-1} +0.0412\,\ln\sigma_{t-1}^2 +0.9200\,\ln\sigma_{t-2}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.5989 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0074 | 0.0002 | 33.8794 | 0.0000 |
| $\rho_1$ | 0.1430 | 0.0016 | 91.7625 | 0.0000 |
| $\rho_2$ | 0.1015 | 0.0016 | 63.3776 | 0.0000 |
| $\phi_1$ | 0.8896 | 0.0000 | 2430766.0218 | 0.0000 |
| $\omega$ | -0.0740 | 0.0000 | -50401.4684 | 0.0000 |
| $\alpha_1$ | -0.3421 | 0.0001 | -2676.8779 | 0.0000 |
| $\gamma_1$ | 0.2355 | 0.0010 | 241.7362 | 0.0000 |
| $\beta_1$ | 0.0412 | 0.0002 | 186.9354 | 0.0000 |
| $\beta_2$ | 0.9200 | 0.0000 | 728877.9846 | 0.0000 |
| $\nu$ | 4.5989 | 0.2382 | 19.3104 | 0.0000 |

### BIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M055` | Student's $t$ | Constant | GARCH(2,1) | -179.8850 | 369.7699 | 386.6231 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\hat{\sigma}_t^2 = 0.1192 +0.4570\,u_{t-1}^2 +0.5209\,u_{t-2}^2 +0.0000\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.9416 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.1192 | 0.0417 | 2.8561 | 0.0043 |
| $\alpha_1$ | 0.4570 | 0.1774 | 2.5755 | 0.0100 |
| $\alpha_2$ | 0.5209 | 0.2133 | 2.4419 | 0.0146 |
| $\beta_1$ | 0.0000 | 0.1189 | 0.0000 | 1.0000 |
| $\nu$ | 4.9416 | 1.7105 | 2.8890 | 0.0039 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M050` | Student's $t$ | Constant | GJR-GARCH(1,1) | -178.6916 | 367.3832 | 384.2364 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0636 +0.7969\,(u_{t-1}^+)^2 +0.1469\,(u_{t-1}^-)^2 +0.4353\,\sigma_{t-1}^2
$$



Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.7915 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.0636 | 0.0255 | 2.4924 | 0.0127 |
| $\alpha_1$ | 0.7969 | 0.2566 | 3.1059 | 0.0019 |
| $\gamma_1-\alpha_1$ | -0.6499 | 0.2320 | -2.8010 | 0.0051 |
| $\beta_1$ | 0.4353 | 0.1168 | 3.7261 | 0.0002 |
| $\nu$ | 4.7915 | 1.5515 | 3.0883 | 0.0020 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M051` | Student's $t$ | Constant | EGARCH(1,1) | -178.3641 | 366.7281 | 383.5813 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.2496 +0.5580\,(|z_{t-1}|-\sqrt{2/\pi}) +0.2647\,z_{t-1} +0.7897\,\ln\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.0826 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | -0.2496 | 0.1063 | -2.3491 | 0.0188 |
| $\alpha_1$ | 0.5580 | 0.1395 | 4.0011 | 0.0001 |
| $\gamma_1$ | 0.2647 | 0.0782 | 3.3870 | 0.0007 |
| $\beta_1$ | 0.7897 | 0.0610 | 12.9555 | 0.0000 |
| $\nu$ | 5.0826 | 1.7271 | 2.9428 | 0.0033 |





## All Model Results

| Distribution | Mean Process | Volatility Process | Best Looglik | AIC | BIC |
|---|---|---|---:|---:|---:|
| Normal | Constant | GARCH(1,1) | -198.2015 | 402.4029 | 412.5148 |
| Normal | Constant | GARCH(1,2) | -198.2015 | 404.4029 | 417.8855 |
| Normal | Constant | GARCH(2,1) | -189.9000 | 387.8000 | 401.2826 |
| Normal | Constant | GARCH(2,2) | -189.9000 | 389.8000 | 406.6532 |
| Normal | Constant | GJR-GARCH(1,1) | -189.8927 | 387.7854 | 401.2680 |
| Normal | Constant | GJR-GARCH(1,2) | -189.8927 | 389.7854 | 406.6386 |
| Normal | Constant | GJR-GARCH(2,1) | -182.2520 | 376.5039 | 396.7277 |
| Normal | Constant | GJR-GARCH(2,2) | -182.2520 | 378.5039 | 402.0984 |
| Normal | Constant | EGARCH(1,1) | -188.9696 | 385.9392 | 399.4217 |
| Normal | Constant | EGARCH(1,2) | -188.9696 | 387.9392 | 404.7924 |
| Normal | Constant | EGARCH(2,1) | -182.9516 | 377.9032 | 398.1271 |
| Normal | Constant | EGARCH(2,2) | -182.9516 | 379.9032 | 403.4977 |
| Normal | ARX(1,1) | GARCH(1,1) | -191.0320 | 394.0640 | 414.2879 |
| Normal | ARX(1,1) | GARCH(1,2) | -191.0320 | 396.0640 | 419.6585 |
| Normal | ARX(1,1) | GARCH(2,1) | -186.6387 | 387.2773 | 410.8718 |
| Normal | ARX(1,1) | GARCH(2,2) | -186.6387 | 389.2773 | 416.2424 |
| Normal | ARX(1,1) | GJR-GARCH(1,1) | -183.8229 | 381.6458 | 405.2403 |
| Normal | ARX(1,1) | GJR-GARCH(1,2) | -183.8229 | 383.6458 | 410.6109 |
| Normal | ARX(1,1) | GJR-GARCH(2,1) | -177.2892 | 372.5785 | 402.9142 |
| Normal | ARX(1,1) | GJR-GARCH(2,2) | -177.2892 | 374.5785 | 408.2849 |
| Normal | ARX(1,1) | EGARCH(1,1) | -183.7595 | 381.5190 | 405.1134 |
| Normal | ARX(1,1) | EGARCH(1,2) | -171.3421 | 358.6843 | 385.6494 |
| Normal | ARX(1,1) | EGARCH(2,1) | -178.1375 | 374.2750 | 404.6107 |
| Normal | ARX(1,1) | EGARCH(2,2) | -171.2540 | 362.5081 | 396.2144 |
| Normal | ARX(2,1) | GARCH(1,1) | -188.8527 | 391.7055 | 415.2999 |
| Normal | ARX(2,1) | GARCH(1,2) | -188.8527 | 393.7055 | 420.6706 |
| Normal | ARX(2,1) | GARCH(2,1) | -185.6157 | 387.2314 | 414.1965 |
| Normal | ARX(2,1) | GARCH(2,2) | -185.6157 | 389.2314 | 419.5672 |
| Normal | ARX(2,1) | GJR-GARCH(1,1) | -183.7897 | 383.5794 | 410.5445 |
| Normal | ARX(2,1) | GJR-GARCH(1,2) | -183.7897 | 385.5794 | 415.9151 |
| Normal | ARX(2,1) | GJR-GARCH(2,1) | -176.6378 | 373.2755 | 406.9819 |
| Normal | ARX(2,1) | GJR-GARCH(2,2) | -176.6378 | 375.2755 | 412.3525 |
| Normal | ARX(2,1) | EGARCH(1,1) | -183.4410 | 382.8820 | 409.8471 |
| Normal | ARX(2,1) | EGARCH(1,2) | -171.1430 | 360.2860 | 390.6217 |
| Normal | ARX(2,1) | EGARCH(2,1) | -177.9523 | 375.9046 | 409.6110 |
| Normal | ARX(2,1) | EGARCH(2,2) | -177.9523 | 377.9046 | 414.9816 |
| Normal | ARX(2,2) | GARCH(1,1) | -188.8355 | 393.6711 | 420.6362 |
| Normal | ARX(2,2) | GARCH(1,2) | -188.8355 | 395.6711 | 426.0068 |
| Normal | ARX(2,2) | GARCH(2,1) | -185.6074 | 389.2147 | 419.5505 |
| Normal | ARX(2,2) | GARCH(2,2) | -185.6074 | 391.2147 | 424.9211 |
| Normal | ARX(2,2) | GJR-GARCH(1,1) | -183.7821 | 385.5642 | 415.9000 |
| Normal | ARX(2,2) | GJR-GARCH(1,2) | -183.7821 | 387.5642 | 421.2706 |
| Normal | ARX(2,2) | GJR-GARCH(2,1) | -176.5660 | 375.1321 | 412.2091 |
| Normal | ARX(2,2) | GJR-GARCH(2,2) | -176.5660 | 377.1321 | 417.5797 |
| Normal | ARX(2,2) | EGARCH(1,1) | -182.8681 | 383.7363 | 414.0720 |
| Normal | ARX(2,2) | EGARCH(1,2) | -182.8681 | 385.7363 | 419.4427 |
| Normal | ARX(2,2) | EGARCH(2,1) | -177.5298 | 377.0595 | 414.1365 |
| Normal | ARX(2,2) | EGARCH(2,2) | -177.5298 | 379.0595 | 419.5072 |
| Student's $t$ | Constant | GARCH(1,1) | -181.9942 | 371.9883 | 385.4709 |
| Student's $t$ | Constant | GARCH(1,2) | -181.9942 | 373.9883 | 390.8415 |
| Student's $t$ | Constant | GARCH(2,1) | -179.8850 | 369.7699 | 386.6231 |
| Student's $t$ | Constant | GARCH(2,2) | -179.8850 | 371.7699 | 391.9938 |
| Student's $t$ | Constant | GJR-GARCH(1,1) | -178.6916 | 367.3832 | 384.2364 |
| Student's $t$ | Constant | GJR-GARCH(1,2) | -178.6916 | 369.3832 | 389.6070 |
| Student's $t$ | Constant | GJR-GARCH(2,1) | -174.8160 | 363.6320 | 387.2265 |
| Student's $t$ | Constant | GJR-GARCH(2,2) | -174.8160 | 365.6320 | 392.5972 |
| Student's $t$ | Constant | EGARCH(1,1) | -178.3641 | 366.7281 | 383.5813 |
| Student's $t$ | Constant | EGARCH(1,2) | -178.3641 | 368.7281 | 388.9520 |
| Student's $t$ | Constant | EGARCH(2,1) | -175.5435 | 365.0870 | 388.6815 |
| Student's $t$ | Constant | EGARCH(2,2) | -175.5435 | 367.0870 | 394.0521 |
| Student's $t$ | ARX(1,1) | GARCH(1,1) | -174.9174 | 363.8348 | 387.4293 |
| Student's $t$ | ARX(1,1) | GARCH(1,2) | -174.9174 | 365.8348 | 392.7999 |
| Student's $t$ | ARX(1,1) | GARCH(2,1) | -173.7890 | 363.5780 | 390.5431 |
| Student's $t$ | ARX(1,1) | GARCH(2,2) | -173.7890 | 365.5780 | 395.9138 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | -171.2540 | 358.5080 | 385.4731 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,2) | -171.2540 | 360.5080 | 390.8437 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,1) | -168.5408 | 357.0815 | 390.7879 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,2) | -168.5408 | 359.0815 | 396.1586 |
| Student's $t$ | ARX(1,1) | EGARCH(1,1) | -171.0753 | 358.1506 | 385.1157 |
| Student's $t$ | ARX(1,1) | EGARCH(1,2) | -171.0753 | 360.1506 | 390.4864 |
| Student's $t$ | ARX(1,1) | EGARCH(2,1) | -168.3693 | 356.7385 | 390.4449 |
| Student's $t$ | ARX(1,1) | EGARCH(2,2) | -168.3693 | 358.7385 | 395.8155 |
| Student's $t$ | ARX(2,1) | GARCH(1,1) | -172.6764 | 361.3528 | 388.3179 |
| Student's $t$ | ARX(2,1) | GARCH(1,2) | -172.6764 | 363.3528 | 393.6885 |
| Student's $t$ | ARX(2,1) | GARCH(2,1) | -171.9737 | 361.9474 | 392.2832 |
| Student's $t$ | ARX(2,1) | GARCH(2,2) | -171.6977 | 363.3953 | 397.1017 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,1) | -170.0718 | 358.1436 | 388.4794 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,2) | -170.0718 | 360.1436 | 393.8500 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,1) | -167.4976 | 356.9953 | 394.0723 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,2) | -167.4976 | 358.9953 | 399.4429 |
| Student's $t$ | ARX(2,1) | EGARCH(1,1) | -169.7106 | 357.4213 | 387.7570 |
| Student's $t$ | ARX(2,1) | EGARCH(1,2) | -165.9465 | 351.8931 | 385.5994 |
| Student's $t$ | ARX(2,1) | EGARCH(2,1) | -167.5810 | 357.1621 | 394.2391 |
| Student's $t$ | ARX(2,1) | EGARCH(2,2) | -167.5810 | 359.1621 | 399.6098 |
| Student's $t$ | ARX(2,2) | GARCH(1,1) | -172.5010 | 363.0019 | 393.3377 |
| Student's $t$ | ARX(2,2) | GARCH(1,2) | -172.5010 | 365.0019 | 398.7083 |
| Student's $t$ | ARX(2,2) | GARCH(2,1) | -171.8786 | 363.7573 | 397.4636 |
| Student's $t$ | ARX(2,2) | GARCH(2,2) | -171.6518 | 365.3035 | 402.3805 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,1) | -169.9900 | 359.9800 | 393.6864 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,2) | -169.9900 | 361.9800 | 399.0571 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,1) | -167.3944 | 358.7887 | 399.2364 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,2) | -167.3944 | 360.7887 | 404.6070 |
| Student's $t$ | ARX(2,2) | EGARCH(1,1) | -169.4442 | 358.8885 | 392.5948 |
| Student's $t$ | ARX(2,2) | EGARCH(1,2) | -169.4442 | 360.8885 | 397.9655 |
| Student's $t$ | ARX(2,2) | EGARCH(2,1) | -167.2956 | 358.5911 | 399.0388 |
| Student's $t$ | ARX(2,2) | EGARCH(2,2) | -167.2956 | 360.5911 | 404.4094 |
| Gaussian mixture | Constant | GARCH(1,1) | -179.4873 | 370.9746 | 391.1984 |
| Gaussian mixture | Constant | GARCH(1,2) | -179.4873 | 372.9746 | 396.5690 |
| Gaussian mixture | Constant | GARCH(2,1) | -177.7161 | 369.4323 | 393.0268 |
| Gaussian mixture | Constant | GARCH(2,2) | -177.7161 | 371.4323 | 398.3974 |
| Gaussian mixture | Constant | GJR-GARCH(1,1) | -175.2123 | 364.4246 | 388.0190 |
| Gaussian mixture | Constant | GJR-GARCH(1,2) | -175.2123 | 366.4246 | 393.3897 |
| Gaussian mixture | Constant | GJR-GARCH(2,1) | -171.8210 | 361.6420 | 391.9777 |
| Gaussian mixture | Constant | GJR-GARCH(2,2) | -171.8210 | 363.6420 | 397.3484 |
| Gaussian mixture | Constant | EGARCH(1,1) | -173.8334 | 361.6667 | 385.2612 |
| Gaussian mixture | Constant | EGARCH(1,2) | -173.8334 | 363.6667 | 390.6318 |
| Gaussian mixture | Constant | EGARCH(2,1) | -170.7788 | 359.5577 | 389.8934 |
| Gaussian mixture | Constant | EGARCH(2,2) | -170.7788 | 361.5577 | 395.2641 |
| Gaussian mixture | ARX(1,1) | GARCH(1,1) | -174.5852 | 367.1704 | 397.5061 |
| Gaussian mixture | ARX(1,1) | GARCH(1,2) | -174.5852 | 369.1704 | 402.8768 |
| Gaussian mixture | ARX(1,1) | GARCH(2,1) | -173.5617 | 367.1233 | 400.8297 |
| Gaussian mixture | ARX(1,1) | GARCH(2,2) | -173.5617 | 369.1233 | 406.2004 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,1) | -170.2226 | 360.4452 | 394.1515 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,2) | -170.2226 | 362.4452 | 399.5222 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,1) | -167.4469 | 358.8939 | 399.3415 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,2) | -167.4469 | 360.8939 | 404.7122 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,1) | -169.4422 | 358.8844 | 392.5908 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,2) | -169.4422 | 360.8844 | 397.9614 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,1) | -166.2197 | 356.4394 | 396.8871 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,2) | -166.2197 | 358.4394 | 402.2577 |
| Gaussian mixture | ARX(2,1) | GARCH(1,1) | -171.3839 | 362.7678 | 396.4742 |
| Gaussian mixture | ARX(2,1) | GARCH(1,2) | -171.3839 | 364.7678 | 401.8449 |
| Gaussian mixture | ARX(2,1) | GARCH(2,1) | -170.7809 | 363.5618 | 400.6389 |
| Gaussian mixture | ARX(2,1) | GARCH(2,2) | -170.6186 | 365.2371 | 405.6848 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,1) | -168.6131 | 359.2261 | 396.3031 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,2) | -168.6131 | 361.2261 | 401.6738 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,1) | -166.2115 | 358.4230 | 402.2413 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,2) | -166.2115 | 360.4230 | 407.6120 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,1) | -165.3669 | 352.7338 | 389.8108 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,2) | -164.6031 | 353.2062 | 393.6539 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,1) | -165.2689 | 356.5377 | 400.3560 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,2) | -163.6882 | 355.3765 | 402.5654 |
| Gaussian mixture | ARX(2,2) | GARCH(1,1) | -171.1994 | 364.3989 | 401.4759 |
| Gaussian mixture | ARX(2,2) | GARCH(1,2) | -171.1994 | 366.3989 | 406.8465 |
| Gaussian mixture | ARX(2,2) | GARCH(2,1) | -170.7011 | 365.4022 | 405.8499 |
| Gaussian mixture | ARX(2,2) | GARCH(2,2) | -174.8814 | 375.7628 | 419.5811 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,1) | -168.4801 | 360.9602 | 401.4079 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,2) | -168.4801 | 362.9602 | 406.7785 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,1) | -166.0585 | 360.1169 | 407.3059 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,2) | -166.0585 | 362.1169 | 412.6765 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,1) | -167.3628 | 358.7257 | 399.1733 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,2) | -166.4468 | 358.8937 | 402.7120 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,1) | -162.5823 | 353.1646 | 400.3535 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,2) | -164.5054 | 359.0108 | 409.5704 |
