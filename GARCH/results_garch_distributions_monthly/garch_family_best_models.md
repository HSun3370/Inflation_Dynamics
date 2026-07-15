```{raw:typst}
#set page(margin: auto)
```

# Model Selection Report

For each model combination, the reported row uses the highest log-likelihood estimate that passes optimizer convergence and stationarity checks across 50 starts. If no start passes all checks, the reported row uses the best optimizer-converged attempted log likelihood and is flagged in the CSV `selection_status` column.

**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**[^garch-aic-sample-note]

| Distribution | GARCH Mean | GARCH Vol | GARCH AIC | GJR Mean | GJR Vol | GJR AIC | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(1,1) | (1,1) | -140.5477 | ARX(1,1) | (2,2) | -143.8422 | ARX(1,1) | (2,1) | -135.8873 |
| Student's $t$ | ARX(1,1) | (1,1) | -186.3088 | ARX(1,1) | (1,1) | <span style="color:red">-188.2854</span> | ARX(1,1) | (1,1) | -180.8314 |
| Gaussian mixture | ARX(1,1) | (1,1) | -185.3003 | ARX(1,1) | (1,1) | -187.6112 | ARX(1,1) | (1,1) | -181.8980 |

[^garch-aic-sample-note]: These numbers differ from earlier GARCH tables because earlier code used inconsistent effective sample sizes across model settings due to a sample-trimming error. The correction is discussed in the [Data Summary effective-sample section](../../DataSummary/README.md#effective-sample). The current estimates use a common effective sample with **688 observations**.

**Table 3: Model selection by BIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH BIC | GJR Mean | GJR Vol | GJR BIC | EGARCH Mean | EGARCH Vol | EGARCH BIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(1,1) | (1,1) | -113.3450 | ARX(1,1) | (1,1) | -108.7612 | ARX(1,1) | (1,1) | -95.4457 |
| Student's $t$ | ARX(1,1) | (1,1) | <span style="color:red">-154.5723</span> | ARX(1,1) | (1,1) | -152.0151 | ARX(1,1) | (1,1) | -144.5611 |
| Gaussian mixture | ARX(1,1) | (1,1) | -144.4962 | ARX(1,1) | (1,1) | -142.2734 | ARX(1,1) | (1,1) | -136.5601 |

## All Model Results

| Distribution | Mean Process | Volatility Process | Best LogLik | AIC | BIC |
|---|---|---|---:|---:|---:|
| Normal | Constant | GARCH(1,1) | 21.1614 | -36.3228 | -22.7215 |
| Normal | Constant | GARCH(1,2) | 22.6158 | -37.2315 | -19.0963 |
| Normal | Constant | GARCH(2,1) | 21.1614 | -34.3228 | -16.1877 |
| Normal | Constant | GARCH(2,2) | 22.6158 | -35.2315 | -12.5626 |
| Normal | Constant | GJR-GARCH(1,1) | 21.3677 | -34.7353 | -16.6002 |
| Normal | Constant | GJR-GARCH(1,2) | 22.9861 | -35.9722 | -13.3033 |
| Normal | Constant | GJR-GARCH(2,1) | 21.6293 | -31.2587 | -4.0559 |
| Normal | Constant | GJR-GARCH(2,2) | 25.9373 | -37.8747 | -6.1381 |
| Normal | Constant | EGARCH(1,1) | 16.5685 | -25.1371 | -7.0019 |
| Normal | Constant | EGARCH(1,2) | 19.0223 | -28.0446 | -5.3756 |
| Normal | Constant | EGARCH(2,1) | 25.9907 | -39.9814 | -12.7787 |
| Normal | Constant | EGARCH(2,2) | 25.9907 | -37.9814 | -6.2449 |
| Normal | ARX(1,1) | GARCH(1,1) | 76.2739 | -140.5477 | -113.3450 |
| Normal | ARX(1,1) | GARCH(1,2) | 76.2746 | -138.5492 | -106.8127 |
| Normal | ARX(1,1) | GARCH(2,1) | 76.2739 | -138.5477 | -106.8112 |
| Normal | ARX(1,1) | GARCH(2,2) | 76.3475 | -136.6950 | -100.4247 |
| Normal | ARX(1,1) | GJR-GARCH(1,1) | 77.2489 | -140.4978 | -108.7612 |
| Normal | ARX(1,1) | GJR-GARCH(1,2) | 77.2499 | -138.4998 | -102.2295 |
| Normal | ARX(1,1) | GJR-GARCH(2,1) | 79.6561 | -141.3123 | -100.5082 |
| Normal | ARX(1,1) | GJR-GARCH(2,2) | 81.9211 | -143.8422 | -98.5043 |
| Normal | ARX(1,1) | EGARCH(1,1) | 70.5911 | -127.1822 | -95.4457 |
| Normal | ARX(1,1) | EGARCH(1,2) | 70.8116 | -125.6232 | -89.3529 |
| Normal | ARX(1,1) | EGARCH(2,1) | 76.9436 | -135.8873 | -95.0832 |
| Normal | ARX(1,1) | EGARCH(2,2) | 76.9436 | -133.8873 | -88.5494 |
| Normal | ARX(2,1) | GARCH(1,1) | 76.4837 | -138.9675 | -107.2310 |
| Normal | ARX(2,1) | GARCH(1,2) | 76.4837 | -136.9675 | -100.6972 |
| Normal | ARX(2,1) | GARCH(2,1) | 76.4848 | -136.9695 | -100.6992 |
| Normal | ARX(2,1) | GARCH(2,2) | 76.5935 | -135.1870 | -94.3829 |
| Normal | ARX(2,1) | GJR-GARCH(1,1) | 77.6015 | -139.2029 | -102.9326 |
| Normal | ARX(2,1) | GJR-GARCH(1,2) | 77.6015 | -137.2029 | -96.3988 |
| Normal | ARX(2,1) | GJR-GARCH(2,1) | 80.0966 | -140.1932 | -94.8553 |
| Normal | ARX(2,1) | GJR-GARCH(2,2) | 82.5502 | -143.1004 | -93.2288 |
| Normal | ARX(2,1) | EGARCH(1,1) | 70.8342 | -125.6684 | -89.3981 |
| Normal | ARX(2,1) | EGARCH(1,2) | 71.0550 | -124.1099 | -83.3058 |
| Normal | ARX(2,1) | EGARCH(2,1) | 77.3960 | -134.7920 | -89.4541 |
| Normal | ARX(2,1) | EGARCH(2,2) | 77.3960 | -132.7920 | -82.9203 |
| Normal | ARX(2,2) | GARCH(1,1) | 76.6971 | -137.3941 | -101.1238 |
| Normal | ARX(2,2) | GARCH(1,2) | 76.6971 | -135.3941 | -94.5900 |
| Normal | ARX(2,2) | GARCH(2,1) | 76.7001 | -135.4003 | -94.5962 |
| Normal | ARX(2,2) | GARCH(2,2) | 76.8234 | -133.6467 | -88.3088 |
| Normal | ARX(2,2) | GJR-GARCH(1,1) | 77.8171 | -137.6342 | -96.8301 |
| Normal | ARX(2,2) | GJR-GARCH(1,2) | 77.8171 | -135.6342 | -90.2963 |
| Normal | ARX(2,2) | GJR-GARCH(2,1) | 80.5584 | -139.1168 | -89.2451 |
| Normal | ARX(2,2) | GJR-GARCH(2,2) | 83.1729 | -142.3459 | -87.9404 |
| Normal | ARX(2,2) | EGARCH(1,1) | 71.0135 | -124.0269 | -83.2228 |
| Normal | ARX(2,2) | EGARCH(1,2) | 71.2294 | -122.4588 | -77.1209 |
| Normal | ARX(2,2) | EGARCH(2,1) | 77.8360 | -133.6719 | -83.8003 |
| Normal | ARX(2,2) | EGARCH(2,2) | 77.8360 | -131.6719 | -77.2665 |
| Student's $t$ | Constant | GARCH(1,1) | 38.7813 | -69.5626 | -51.4274 |
| Student's $t$ | Constant | GARCH(1,2) | 39.7888 | -69.5775 | -46.9086 |
| Student's $t$ | Constant | GARCH(2,1) | 38.7813 | -67.5626 | -44.8937 |
| Student's $t$ | Constant | GARCH(2,2) | 39.7888 | -67.5775 | -40.3748 |
| Student's $t$ | Constant | GJR-GARCH(1,1) | 39.7463 | -69.4925 | -46.8236 |
| Student's $t$ | Constant | GJR-GARCH(1,2) | 40.9705 | -69.9410 | -42.7383 |
| Student's $t$ | Constant | GJR-GARCH(2,1) | 39.7996 | -65.5991 | -33.8626 |
| Student's $t$ | Constant | GJR-GARCH(2,2) | 42.0009 | -68.0017 | -31.7314 |
| Student's $t$ | Constant | EGARCH(1,1) | 36.5166 | -63.0332 | -40.3642 |
| Student's $t$ | Constant | EGARCH(1,2) | 38.1862 | -64.3725 | -37.1697 |
| Student's $t$ | Constant | EGARCH(2,1) | 42.9506 | -71.9012 | -40.1647 |
| Student's $t$ | Constant | EGARCH(2,2) | 42.9506 | -69.9012 | -33.6309 |
| Student's $t$ | ARX(1,1) | GARCH(1,1) | 100.1544 | -186.3088 | -154.5723 |
| Student's $t$ | ARX(1,1) | GARCH(1,2) | 100.1544 | -184.3088 | -148.0385 |
| Student's $t$ | ARX(1,1) | GARCH(2,1) | 100.1553 | -184.3106 | -148.0403 |
| Student's $t$ | ARX(1,1) | GARCH(2,2) | 100.1623 | -182.3247 | -141.5206 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | 102.1427 | -188.2854 | -152.0151 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(1,2) | 102.1431 | -186.2861 | -145.4820 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,1) | 102.5336 | -185.0673 | -139.7294 |
| Student's $t$ | ARX(1,1) | GJR-GARCH(2,2) | 103.3727 | -184.7455 | -134.8738 |
| Student's $t$ | ARX(1,1) | EGARCH(1,1) | 98.4157 | -180.8314 | -144.5611 |
| Student's $t$ | ARX(1,1) | EGARCH(1,2) | 98.5319 | -179.0639 | -138.2598 |
| Student's $t$ | ARX(1,1) | EGARCH(2,1) | 100.0980 | -180.1960 | -134.8581 |
| Student's $t$ | ARX(1,1) | EGARCH(2,2) | 101.1880 | -180.3761 | -130.5044 |
| Student's $t$ | ARX(2,1) | GARCH(1,1) | 100.2634 | -184.5268 | -148.2564 |
| Student's $t$ | ARX(2,1) | GARCH(1,2) | 100.2634 | -182.5268 | -141.7227 |
| Student's $t$ | ARX(2,1) | GARCH(2,1) | 100.2639 | -182.5278 | -141.7237 |
| Student's $t$ | ARX(2,1) | GARCH(2,2) | 100.2702 | -180.5404 | -135.2025 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,1) | 102.4946 | -186.9891 | -146.1850 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(1,2) | 102.4953 | -184.9906 | -139.6527 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,1) | 102.9181 | -183.8363 | -133.9646 |
| Student's $t$ | ARX(2,1) | GJR-GARCH(2,2) | 103.8765 | -183.7530 | -129.3475 |
| Student's $t$ | ARX(2,1) | EGARCH(1,1) | 98.7448 | -179.4897 | -138.6856 |
| Student's $t$ | ARX(2,1) | EGARCH(1,2) | 98.8712 | -177.7423 | -132.4045 |
| Student's $t$ | ARX(2,1) | EGARCH(2,1) | 100.6370 | -179.2741 | -129.4024 |
| Student's $t$ | ARX(2,1) | EGARCH(2,2) | 101.6258 | -179.2516 | -124.8461 |
| Student's $t$ | ARX(2,2) | GARCH(1,1) | 100.6088 | -183.2175 | -142.4134 |
| Student's $t$ | ARX(2,2) | GARCH(1,2) | 100.6088 | -181.2175 | -135.8796 |
| Student's $t$ | ARX(2,2) | GARCH(2,1) | 100.6112 | -181.2224 | -135.8845 |
| Student's $t$ | ARX(2,2) | GARCH(2,2) | 100.6194 | -179.2387 | -129.3670 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,1) | 102.8861 | -185.7722 | -140.4343 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(1,2) | 102.8861 | -183.7722 | -133.9005 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,1) | 103.4125 | -182.8251 | -128.4196 |
| Student's $t$ | ARX(2,2) | GJR-GARCH(2,2) | 104.4932 | -182.9864 | -124.0471 |
| Student's $t$ | ARX(2,2) | EGARCH(1,1) | 99.1348 | -178.2695 | -132.9316 |
| Student's $t$ | ARX(2,2) | EGARCH(1,2) | 99.2458 | -176.4916 | -126.6200 |
| Student's $t$ | ARX(2,2) | EGARCH(2,1) | 101.1255 | -178.2510 | -123.8455 |
| Student's $t$ | ARX(2,2) | EGARCH(2,2) | 102.2040 | -178.4080 | -119.4687 |
| Gaussian mixture | Constant | GARCH(1,1) | 40.8067 | -69.6135 | -42.4107 |
| Gaussian mixture | Constant | GARCH(1,2) | 42.2087 | -70.4173 | -38.6808 |
| Gaussian mixture | Constant | GARCH(2,1) | 40.8067 | -67.6135 | -35.8770 |
| Gaussian mixture | Constant | GARCH(2,2) | 42.2087 | -68.4173 | -32.1470 |
| Gaussian mixture | Constant | GJR-GARCH(1,1) | 43.3968 | -72.7935 | -41.0570 |
| Gaussian mixture | Constant | GJR-GARCH(1,2) | 44.9503 | -73.9006 | -37.6303 |
| Gaussian mixture | Constant | GJR-GARCH(2,1) | 43.3968 | -68.7935 | -27.9894 |
| Gaussian mixture | Constant | GJR-GARCH(2,2) | 45.0894 | -70.1788 | -24.8409 |
| Gaussian mixture | Constant | EGARCH(1,1) | 41.4377 | -68.8754 | -37.1389 |
| Gaussian mixture | Constant | EGARCH(1,2) | 43.0855 | -70.1710 | -33.9007 |
| Gaussian mixture | Constant | EGARCH(2,1) | 45.6972 | -73.3943 | -32.5902 |
| Gaussian mixture | Constant | EGARCH(2,2) | 45.6972 | -71.3943 | -26.0565 |
| Gaussian mixture | ARX(1,1) | GARCH(1,1) | 101.6502 | -185.3003 | -144.4962 |
| Gaussian mixture | ARX(1,1) | GARCH(1,2) | 101.6503 | -183.3005 | -137.9626 |
| Gaussian mixture | ARX(1,1) | GARCH(2,1) | 101.6502 | -183.3003 | -137.9624 |
| Gaussian mixture | ARX(1,1) | GARCH(2,2) | 101.6599 | -181.3198 | -131.4481 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,1) | 103.8056 | -187.6112 | -142.2734 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(1,2) | 103.8056 | -185.6112 | -135.7396 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,1) | 104.2851 | -184.5702 | -130.1647 |
| Gaussian mixture | ARX(1,1) | GJR-GARCH(2,2) | 104.9765 | -183.9530 | -125.0138 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,1) | 100.9490 | -181.8980 | -136.5601 |
| Gaussian mixture | ARX(1,1) | EGARCH(1,2) | 100.9582 | -179.9164 | -130.0447 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,1) | 102.4071 | -180.8143 | -126.4088 |
| Gaussian mixture | ARX(1,1) | EGARCH(2,2) | 103.2633 | -180.5267 | -121.5874 |
| Gaussian mixture | ARX(2,1) | GARCH(1,1) | 101.8307 | -183.6615 | -138.3236 |
| Gaussian mixture | ARX(2,1) | GARCH(1,2) | 101.8308 | -181.6616 | -131.7899 |
| Gaussian mixture | ARX(2,1) | GARCH(2,1) | 101.8307 | -181.6615 | -131.7898 |
| Gaussian mixture | ARX(2,1) | GARCH(2,2) | 101.8386 | -179.6772 | -125.2718 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,1) | 104.4273 | -186.8546 | -136.9829 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(1,2) | 104.4273 | -184.8546 | -130.4492 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,1) | 104.9099 | -183.8198 | -124.8805 |
| Gaussian mixture | ARX(2,1) | GJR-GARCH(2,2) | 105.6055 | -183.2110 | -119.7380 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,1) | 101.6090 | -181.2179 | -131.3462 |
| Gaussian mixture | ARX(2,1) | EGARCH(1,2) | 101.6161 | -179.2322 | -124.8267 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,1) | 103.1390 | -180.2781 | -121.3388 |
| Gaussian mixture | ARX(2,1) | EGARCH(2,2) | 103.8033 | -179.6065 | -116.1335 |
| Gaussian mixture | ARX(2,2) | GARCH(1,1) | 102.0243 | -182.0486 | -132.1769 |
| Gaussian mixture | ARX(2,2) | GARCH(1,2) | 102.0243 | -180.0486 | -125.6431 |
| Gaussian mixture | ARX(2,2) | GARCH(2,1) | 102.0243 | -180.0486 | -125.6431 |
| Gaussian mixture | ARX(2,2) | GARCH(2,2) | 102.0359 | -178.0717 | -119.1325 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,1) | 104.7171 | -185.4342 | -131.0287 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(1,2) | 104.7171 | -183.4342 | -124.4949 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,1) | 105.2849 | -182.5698 | -119.0968 |
| Gaussian mixture | ARX(2,2) | GJR-GARCH(2,2) | 106.1043 | -182.2086 | -114.2018 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,1) | 101.9001 | -179.8001 | -125.3947 |
| Gaussian mixture | ARX(2,2) | EGARCH(1,2) | 101.9044 | -177.8089 | -118.8696 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,1) | 103.5168 | -179.0336 | -115.5606 |
| Gaussian mixture | ARX(2,2) | EGARCH(2,2) | 104.2678 | -178.5357 | -110.5288 |

## Best Models by Criterion and Volatility Family

For each criterion and each volatility family, the selected model is the best across mean-process choices, orders, and distributions using stable & successful fits.
The main result tables contain 144 reported model combinations.

### AIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M061` | Student's $t$ | ARX(1,1) | GARCH(1,1) | 100.1544 | -186.3088 | -154.5723 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0148 +0.4441\,\pi_t +0.5718\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0061 +0.2477\,u_{t-1}^2 +0.6670\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.0899 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0148 | 0.0180 | 0.8241 | 0.4099 |
| $\rho_1$ | 0.4441 | 0.0451 | 9.8407 | 0.0000 |
| $\phi_1$ | 0.5718 | 0.0760 | 7.5191 | 0.0000 |
| $\omega$ | 0.0061 | 0.0024 | 2.5927 | 0.0095 |
| $\alpha_1$ | 0.2477 | 0.0666 | 3.7198 | 0.0002 |
| $\beta_1$ | 0.6670 | 0.0735 | 9.0700 | 0.0000 |
| $\nu$ | 5.0899 | 0.8695 | 5.8540 | 0.0000 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M062` | Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | 102.1427 | -188.2854 | -152.0151 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0216 +0.4340\,\pi_t +0.5660\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0054 +0.3015\,(u_{t-1}^+)^2 +0.1484\,(u_{t-1}^-)^2 +0.6979\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.0138 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0216 | 0.0183 | 1.1819 | 0.2373 |
| $\rho_1$ | 0.4340 | 0.0440 | 9.8613 | 0.0000 |
| $\phi_1$ | 0.5660 | 0.0741 | 7.6411 | 0.0000 |
| $\omega$ | 0.0054 | 0.0020 | 2.7726 | 0.0056 |
| $\alpha_1$ | 0.3015 | 0.0733 | 4.1159 | 0.0000 |
| $\gamma_1-\alpha_1$ | -0.1531 | 0.0695 | -2.2040 | 0.0275 |
| $\beta_1$ | 0.6979 | 0.0695 | 10.0370 | 0.0000 |
| $\nu$ | 5.0138 | 0.8531 | 5.8772 | 0.0000 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M111` | Gaussian mixture | ARX(1,1) | EGARCH(1,1) | 100.9490 | -181.8980 | -136.5601 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0176 +0.4272\,\pi_t +0.5975\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.2797 +0.3475\,(|z_{t-1}|-\sqrt{2/\pi}) +0.0628\,z_{t-1} +0.9011\,\ln\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $p_1$ | 0.8289 |
| $\mu_1$ | 0.0039 |
| $\sigma_1^2$ | 0.5544 |
| $p_2$ | 0.1711 |
| $\mu_2$ | -0.0191 |
| $\sigma_2^2$ | 3.1581 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0176 | 0.0165 | 1.0676 | 0.2857 |
| $\rho_1$ | 0.4272 | 0.0422 | 10.1279 | 0.0000 |
| $\phi_1$ | 0.5975 | 0.0669 | 8.9321 | 0.0000 |
| $\omega$ | -0.2797 | 0.1456 | -1.9212 | 0.0547 |
| $\alpha_1$ | 0.3475 | 0.0960 | 3.6209 | 0.0003 |
| $\gamma_1$ | 0.0628 | 0.0370 | 1.6985 | 0.0894 |
| $\beta_1$ | 0.9011 | 0.0490 | 18.3830 | 0.0000 |
| $p_1$ | 0.8289 | 0.0604 | 13.7212 | 0.0000 |
| $\mu_1$ | 0.0039 | 0.0405 | 0.0974 | 0.9224 |
| $\sigma_1^2$ | 0.5544 | 0.0690 | 8.0391 | 0.0000 |

### BIC

#### GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M061` | Student's $t$ | ARX(1,1) | GARCH(1,1) | 100.1544 | -186.3088 | -154.5723 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0148 +0.4441\,\pi_t +0.5718\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0061 +0.2477\,u_{t-1}^2 +0.6670\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.0899 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0148 | 0.0180 | 0.8241 | 0.4099 |
| $\rho_1$ | 0.4441 | 0.0451 | 9.8407 | 0.0000 |
| $\phi_1$ | 0.5718 | 0.0760 | 7.5191 | 0.0000 |
| $\omega$ | 0.0061 | 0.0024 | 2.5927 | 0.0095 |
| $\alpha_1$ | 0.2477 | 0.0666 | 3.7198 | 0.0002 |
| $\beta_1$ | 0.6670 | 0.0735 | 9.0700 | 0.0000 |
| $\nu$ | 5.0899 | 0.8695 | 5.8540 | 0.0000 |

#### GJR-GARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M062` | Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | 102.1427 | -188.2854 | -152.0151 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0216 +0.4340\,\pi_t +0.5660\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0054 +0.3015\,(u_{t-1}^+)^2 +0.1484\,(u_{t-1}^-)^2 +0.6979\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.0138 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0216 | 0.0183 | 1.1819 | 0.2373 |
| $\rho_1$ | 0.4340 | 0.0440 | 9.8613 | 0.0000 |
| $\phi_1$ | 0.5660 | 0.0741 | 7.6411 | 0.0000 |
| $\omega$ | 0.0054 | 0.0020 | 2.7726 | 0.0056 |
| $\alpha_1$ | 0.3015 | 0.0733 | 4.1159 | 0.0000 |
| $\gamma_1-\alpha_1$ | -0.1531 | 0.0695 | -2.2040 | 0.0275 |
| $\beta_1$ | 0.6979 | 0.0695 | 10.0370 | 0.0000 |
| $\nu$ | 5.0138 | 0.8531 | 5.8772 | 0.0000 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M063` | Student's $t$ | ARX(1,1) | EGARCH(1,1) | 98.4157 | -180.8314 | -144.5611 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0175 +0.4342\,\pi_t +0.5867\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.3089 +0.3702\,(|z_{t-1}|-\sqrt{2/\pi}) +0.0551\,z_{t-1} +0.8902\,\ln\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.7812 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0175 | 0.0044 | 3.9949 | 0.0001 |
| $\rho_1$ | 0.4342 | 0.0233 | 18.6127 | 0.0000 |
| $\phi_1$ | 0.5867 | 0.0353 | 16.6237 | 0.0000 |
| $\omega$ | -0.3089 | 0.1518 | -2.0349 | 0.0419 |
| $\alpha_1$ | 0.3702 | 0.1193 | 3.1018 | 0.0019 |
| $\gamma_1$ | 0.0551 | 0.0395 | 1.3949 | 0.1631 |
| $\beta_1$ | 0.8902 | 0.0520 | 17.1277 | 0.0000 |
| $\nu$ | 4.7812 | 0.7685 | 6.2216 | 0.0000 |


