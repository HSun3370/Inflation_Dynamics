```{raw:typst}
#set page(margin: auto)
```

# Model Selection Report

**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**[^garch-aic-sample-note]

| Distribution | GARCH Mean | GARCH Vol | GARCH AIC | GJR Mean | GJR Vol | GJR AIC | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(2,1) | (2,1) | 387.2314 | ARX(1,1) | (2,1) | 372.5785 | ARX(2,1) | (1,2) | 368.6220 |
| Student's $t$ | ARX(2,1) | (1,1) | 361.3528 | ARX(2,1) | (2,1) | 356.9953 | ARX(2,1) | (1,2) | <span style="color:red">353.1752</span> |
| Gaussian mixture | ARX(1,1) | (2,1) | 367.1233 | ARX(2,1) | (2,1) | 358.4230 | ARX(1,1) | (1,2) | 353.2873 |

[^garch-aic-sample-note]: These numbers differ from earlier GARCH tables because earlier code used inconsistent effective sample sizes across model settings due to a sample-trimming error. The correction is discussed in the [Data Summary effective-sample section](../../DataSummary/README.md#effective-sample). The current estimates use the common **1969Q2--2022Q4** sample with **215 observations**.

**Table 3: Model selection by BIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH BIC | GJR Mean | GJR Vol | GJR BIC | EGARCH Mean | EGARCH Vol | EGARCH BIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | Constant | (2,1) | 401.2826 | Constant | (2,1) | 396.7277 | Constant | (2,1) | 398.1271 |
| Student's $t$ | Constant | (2,1) | 386.6231 | Constant | (1,1) | 384.2364 | Constant | (1,1) | <span style="color:red">383.5813</span> |
| Gaussian mixture | ARX(1,1) | (2,1) | 400.8297 | ARX(1,1) | (1,1) | 394.1515 | Constant | (1,1) | 385.2612 |

## Best Models by Criterion and Volatility Family

For each criterion and each volatility family, the selected model is the best across mean-process choices, orders, and distributions using stable & successful fits. 

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
\hat{\sigma}_t^2 = 0.0725 +0.5203\,u_{t-1}^2 +0.4294\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.3023 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0876 | 0.0777 | 1.1280 | 0.2593 |
| $\rho_1$ | 0.2637 | 0.0912 | 2.8914 | 0.0038 |
| $\rho_2$ | 0.1716 | 0.0975 | 1.7600 | 0.0784 |
| $\phi_1$ | 0.5096 | 0.1487 | 3.4274 | 0.0006 |
| $\omega$ | 0.0725 | 0.0385 | 1.8834 | 0.0596 |
| $\alpha_1$ | 0.5203 | 0.2424 | 2.1464 | 0.0318 |
| $\beta_1$ | 0.4294 | 0.1425 | 3.0126 | 0.0026 |
| $\nu$ | 4.3023 | 1.4648 | 2.9372 | 0.0033 |

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
\hat{\sigma}_t^2 = 0.1096 +0.4005\,(u_{t-1}^+)^2 +0.2848\,(u_{t-1}^-)^2 +0.6415\,(u_{t-2}^+)^2 -0.0000\,(u_{t-2}^-)^2 +0.1559\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.3421 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0901 | 0.0838 | 1.0747 | 0.2825 |
| $\rho_1$ | 0.2045 | 0.0960 | 2.1291 | 0.0332 |
| $\rho_2$ | 0.1144 | 0.1043 | 1.0969 | 0.2727 |
| $\phi_1$ | 0.6522 | 0.1740 | 3.7481 | 0.0002 |
| $\omega$ | 0.1096 | 0.0397 | 2.7594 | 0.0058 |
| $\alpha_1$ | 0.4005 | 0.2194 | 1.8250 | 0.0680 |
| $\gamma_1-\alpha_1$ | -0.1157 | 0.2773 | -0.4172 | 0.6765 |
| $\alpha_2$ | 0.6415 | 0.3571 | 1.7963 | 0.0725 |
| $\gamma_2-\alpha_2$ | -0.6415 | 0.3194 | -2.0082 | 0.0446 |
| $\beta_1$ | 0.1559 | 0.1499 | 1.0399 | 0.2984 |
| $\nu$ | 5.3421 | 1.9529 | 2.7354 | 0.0062 |

#### EGARCH
| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M078` | Student's $t$ | ARX(2,1) | EGARCH(1,2) | -166.5876 | 353.1752 | 386.8816 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0496 +0.0937\,\pi_t +0.0735\,\pi_{t-1} +0.9380\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\ln \hat{\sigma}_t^2 = -0.0946 -0.3695\,(|z_{t-1}|-\sqrt{2/\pi}) +0.2773\,z_{t-1} +0.0679\,\ln\sigma_{t-1}^2 +0.8601\,\ln\sigma_{t-2}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 5.2826 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0496 | 0.0466 | 1.0655 | 0.2867 |
| $\rho_1$ | 0.0937 | 0.0833 | 1.1251 | 0.2606 |
| $\rho_2$ | 0.0735 | 0.0440 | 1.6718 | 0.0946 |
| $\phi_1$ | 0.9380 | 0.0000 | 36243962.4808 | 0.0000 |
| $\omega$ | -0.0946 | 0.0001 | -1626.9937 | 0.0000 |
| $\alpha_1$ | -0.3695 | 0.0003 | -1381.2872 | 0.0000 |
| $\gamma_1$ | 0.2773 | 0.0112 | 24.7779 | 0.0000 |
| $\beta_1$ | 0.0679 | 0.0084 | 8.0880 | 0.0000 |
| $\beta_2$ | 0.8601 | 0.0000 | 1485807.1736 | 0.0000 |
| $\nu$ | 5.2826 | 1.7878 | 2.9548 | 0.0031 |

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
| $\nu$ | 4.9417 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.1192 | 0.0417 | 2.8561 | 0.0043 |
| $\alpha_1$ | 0.4570 | 0.1774 | 2.5755 | 0.0100 |
| $\alpha_2$ | 0.5209 | 0.2133 | 2.4419 | 0.0146 |
| $\beta_1$ | 0.0000 | 0.1189 | 0.0000 | 1.0000 |
| $\nu$ | 4.9417 | 1.7105 | 2.8890 | 0.0039 |

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

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.7916 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.0636 | 0.0255 | 2.4925 | 0.0127 |
| $\alpha_1$ | 0.7969 | 0.2566 | 3.1058 | 0.0019 |
| $\gamma_1-\alpha_1$ | -0.6499 | 0.2320 | -2.8009 | 0.0051 |
| $\beta_1$ | 0.4353 | 0.1168 | 3.7262 | 0.0002 |
| $\nu$ | 4.7916 | 1.5515 | 3.0883 | 0.0020 |

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
| $\beta_1$ | 0.7897 | 0.0610 | 12.9553 | 0.0000 |
| $\nu$ | 5.0826 | 1.7272 | 2.9427 | 0.0033 |

## GARCH(1,1) and GJR-GARCH(1,1) Comparisons

These estimates are reported by mean process and innovation distribution because the GARCH(1,1) and GJR-GARCH(1,1) recursions are useful benchmarks for the BEGE specifications.

### Constant

#### Normal

##### GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M001` | Normal | Constant | GARCH(1,1) | -198.2015 | 402.4029 | 412.5148 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0694 +0.4799\,u_{t-1}^2 +0.4725\,\sigma_{t-1}^2
$$

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.0694 | 0.0401 | 1.7321 | 0.0833 |
| $\alpha_1$ | 0.4799 | 0.1653 | 2.9029 | 0.0037 |
| $\beta_1$ | 0.4725 | 0.0979 | 4.8248 | 0.0000 |

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M002` | Normal | Constant | GJR-GARCH(1,1) | -189.8927 | 387.7854 | 401.2680 |

Mean process:

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0573 +0.5990\,(u_{t-1}^+)^2 +0.0194\,(u_{t-1}^-)^2 +0.5578\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.0573 | 0.0223 | 2.5676 | 0.0102 |
| $\alpha_1$ | 0.5990 | 0.3145 | 1.9049 | 0.0568 |
| $\gamma_1-\alpha_1$ | -0.5795 | 0.2577 | -2.2489 | 0.0245 |
| $\beta_1$ | 0.5578 | 0.1685 | 3.3096 | 0.0009 |

#### Student's $t$

##### GARCH(1,1)

No successful model.

##### GJR-GARCH(1,1)

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

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.7916 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.0636 | 0.0255 | 2.4925 | 0.0127 |
| $\alpha_1$ | 0.7969 | 0.2566 | 3.1058 | 0.0019 |
| $\gamma_1-\alpha_1$ | -0.6499 | 0.2320 | -2.8009 | 0.0051 |
| $\beta_1$ | 0.4353 | 0.1168 | 3.7262 | 0.0002 |
| $\nu$ | 4.7916 | 1.5515 | 3.0883 | 0.0020 |

### ARX(1,1)

#### Normal

##### GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M013` | Normal | ARX(1,1) | GARCH(1,1) | -191.0320 | 394.0640 | 414.2879 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.1137 +0.4783\,\pi_t +0.4677\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0978 +0.7142\,u_{t-1}^2 +0.2710\,\sigma_{t-1}^2
$$

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.1137 | 0.0735 | 1.5469 | 0.1219 |
| $\rho_1$ | 0.4783 | 0.2460 | 1.9445 | 0.0518 |
| $\phi_1$ | 0.4677 | 0.2487 | 1.8806 | 0.0600 |
| $\omega$ | 0.0978 | 0.0460 | 2.1266 | 0.0335 |
| $\alpha_1$ | 0.7142 | 0.5291 | 1.3499 | 0.1771 |
| $\beta_1$ | 0.2710 | 0.2284 | 1.1869 | 0.2353 |

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M014` | Normal | ARX(1,1) | GJR-GARCH(1,1) | -183.8229 | 381.6458 | 405.2403 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0840 +0.2067\,\pi_t +0.7961\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0580 +0.6052\,(u_{t-1}^+)^2 +0.0035\,(u_{t-1}^-)^2 +0.6048\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0840 | 0.1082 | 0.7764 | 0.4375 |
| $\rho_1$ | 0.2067 | 0.0925 | 2.2336 | 0.0255 |
| $\phi_1$ | 0.7961 | 0.1511 | 5.2670 | 0.0000 |
| $\omega$ | 0.0580 | 0.0519 | 1.1174 | 0.2638 |
| $\alpha_1$ | 0.6052 | 0.5587 | 1.0832 | 0.2787 |
| $\gamma_1-\alpha_1$ | -0.6017 | 0.4985 | -1.2070 | 0.2274 |
| $\beta_1$ | 0.6048 | 0.3484 | 1.7358 | 0.0826 |

#### Student's $t$

##### GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M061` | Student's $t$ | ARX(1,1) | GARCH(1,1) | -174.9174 | 363.8348 | 387.4293 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0827 +0.2800\,\pi_t +0.6858\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0910 +0.5462\,u_{t-1}^2 +0.3726\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.1326 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0827 | 0.0743 | 1.1132 | 0.2656 |
| $\rho_1$ | 0.2800 | 0.0999 | 2.8028 | 0.0051 |
| $\phi_1$ | 0.6858 | 0.1126 | 6.0885 | 0.0000 |
| $\omega$ | 0.0910 | 0.0426 | 2.1349 | 0.0328 |
| $\alpha_1$ | 0.5462 | 0.2386 | 2.2895 | 0.0221 |
| $\beta_1$ | 0.3726 | 0.1329 | 2.8042 | 0.0050 |
| $\nu$ | 4.1326 | 1.2722 | 3.2484 | 0.0012 |

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M062` | Student's $t$ | ARX(1,1) | GJR-GARCH(1,1) | -171.2540 | 358.5080 | 385.4731 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.1091 +0.2377\,\pi_t +0.7159\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0698 +0.7129\,(u_{t-1}^+)^2 +0.0914\,(u_{t-1}^-)^2 +0.4892\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.6954 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.1091 | 0.0771 | 1.4159 | 0.1568 |
| $\rho_1$ | 0.2377 | 0.0931 | 2.5532 | 0.0107 |
| $\phi_1$ | 0.7159 | 0.1146 | 6.2455 | 0.0000 |
| $\omega$ | 0.0698 | 0.0258 | 2.7016 | 0.0069 |
| $\alpha_1$ | 0.7129 | 0.3163 | 2.2537 | 0.0242 |
| $\gamma_1-\alpha_1$ | -0.6215 | 0.2629 | -2.3637 | 0.0181 |
| $\beta_1$ | 0.4892 | 0.1522 | 3.2139 | 0.0013 |
| $\nu$ | 4.6954 | 1.5763 | 2.9787 | 0.0029 |

#### Gaussian mixture

##### GARCH(1,1)

No successful model.

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M110` | Gaussian mixture | ARX(1,1) | GJR-GARCH(1,1) | -170.2226 | 360.4452 | 394.1515 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.1002 +0.2332\,\pi_t +0.7215\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0775 +0.7766\,(u_{t-1}^+)^2 +0.1301\,(u_{t-1}^-)^2 +0.4604\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $p_1$ | 0.9528 |
| $\mu_1$ | 0.0316 |
| $\sigma_1^2$ | 0.6599 |
| $p_2$ | 0.0472 |
| $\mu_2$ | -0.6370 |
| $\sigma_2^2$ | 7.4387 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.1002 | 0.0729 | 1.3749 | 0.1692 |
| $\rho_1$ | 0.2332 | 0.1055 | 2.2112 | 0.0270 |
| $\phi_1$ | 0.7215 | 0.1141 | 6.3256 | 0.0000 |
| $\omega$ | 0.0775 | 0.0486 | 1.5942 | 0.1109 |
| $\alpha_1$ | 0.7766 | 0.3535 | 2.1972 | 0.0280 |
| $\gamma_1-\alpha_1$ | -0.6465 | 0.2751 | -2.3496 | 0.0188 |
| $\beta_1$ | 0.4604 | 0.1843 | 2.4986 | 0.0125 |
| $p_1$ | 0.9528 | 0.0608 | 15.6677 | 0.0000 |
| $\mu_1$ | 0.0316 | 0.0586 | 0.5386 | 0.5902 |
| $\sigma_1^2$ | 0.6599 | 0.2142 | 3.0803 | 0.0021 |

### ARX(2,1)

#### Normal

##### GARCH(1,1)

No successful model.

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M026` | Normal | ARX(2,1) | GJR-GARCH(1,1) | -183.7897 | 383.5794 | 410.5445 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0828 +0.2078\,\pi_t +0.0096\,\pi_{t-1} +0.7866\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0571 +0.6023\,(u_{t-1}^+)^2 +0.0048\,(u_{t-1}^-)^2 +0.6090\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0828 | 0.1091 | 0.7592 | 0.4478 |
| $\rho_1$ | 0.2078 | 0.0927 | 2.2413 | 0.0250 |
| $\rho_2$ | 0.0096 | 0.1014 | 0.0946 | 0.9246 |
| $\phi_1$ | 0.7866 | 0.2178 | 3.6120 | 0.0003 |
| $\omega$ | 0.0571 | 0.0547 | 1.0441 | 0.2964 |
| $\alpha_1$ | 0.6023 | 0.5989 | 1.0057 | 0.3146 |
| $\gamma_1-\alpha_1$ | -0.5975 | 0.5362 | -1.1144 | 0.2651 |
| $\beta_1$ | 0.6090 | 0.3734 | 1.6307 | 0.1029 |

#### Student's $t$

##### GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M073` | Student's $t$ | ARX(2,1) | GARCH(1,1) | -172.6764 | 361.3528 | 388.3179 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0876 +0.2637\,\pi_t +0.1716\,\pi_{t-1} +0.5096\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0725 +0.5203\,u_{t-1}^2 +0.4294\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.3023 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0876 | 0.0777 | 1.1280 | 0.2593 |
| $\rho_1$ | 0.2637 | 0.0912 | 2.8914 | 0.0038 |
| $\rho_2$ | 0.1716 | 0.0975 | 1.7600 | 0.0784 |
| $\phi_1$ | 0.5096 | 0.1487 | 3.4274 | 0.0006 |
| $\omega$ | 0.0725 | 0.0385 | 1.8834 | 0.0596 |
| $\alpha_1$ | 0.5203 | 0.2424 | 2.1464 | 0.0318 |
| $\beta_1$ | 0.4294 | 0.1425 | 3.0126 | 0.0026 |
| $\nu$ | 4.3023 | 1.4648 | 2.9372 | 0.0033 |

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M074` | Student's $t$ | ARX(2,1) | GJR-GARCH(1,1) | -170.0718 | 358.1436 | 388.4794 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.1086 +0.2335\,\pi_t +0.1251\,\pi_{t-1} +0.5871\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0611 +0.6588\,(u_{t-1}^+)^2 +0.1271\,(u_{t-1}^-)^2 +0.5231\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.5950 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.1086 | 0.0791 | 1.3737 | 0.1695 |
| $\rho_1$ | 0.2335 | 0.0872 | 2.6767 | 0.0074 |
| $\rho_2$ | 0.1251 | 0.0962 | 1.3002 | 0.1935 |
| $\phi_1$ | 0.5871 | 0.1515 | 3.8739 | 0.0001 |
| $\omega$ | 0.0611 | 0.0282 | 2.1700 | 0.0300 |
| $\alpha_1$ | 0.6588 | 0.3260 | 2.0205 | 0.0433 |
| $\gamma_1-\alpha_1$ | -0.5316 | 0.2567 | -2.0711 | 0.0384 |
| $\beta_1$ | 0.5231 | 0.1823 | 2.8688 | 0.0041 |
| $\nu$ | 4.5950 | 1.5577 | 2.9498 | 0.0032 |

#### Gaussian mixture

##### GARCH(1,1)

No successful model.

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M122` | Gaussian mixture | ARX(2,1) | GJR-GARCH(1,1) | -168.6131 | 359.2261 | 396.3031 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.1039 +0.2327\,\pi_t +0.1491\,\pi_{t-1} +0.5478\,SPF_t + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0720 +0.8102\,(u_{t-1}^+)^2 +0.2212\,(u_{t-1}^-)^2 +0.4566\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $p_1$ | 0.9629 |
| $\mu_1$ | 0.0278 |
| $\sigma_1^2$ | 0.6478 |
| $p_2$ | 0.0371 |
| $\mu_2$ | -0.7218 |
| $\sigma_2^2$ | 9.5904 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.1039 | 0.0685 | 1.5176 | 0.1291 |
| $\rho_1$ | 0.2327 | 0.0994 | 2.3402 | 0.0193 |
| $\rho_2$ | 0.1491 | 0.0954 | 1.5630 | 0.1180 |
| $\phi_1$ | 0.5478 | 0.1606 | 3.4108 | 0.0006 |
| $\omega$ | 0.0720 | 0.0500 | 1.4414 | 0.1495 |
| $\alpha_1$ | 0.8102 | 0.3943 | 2.0550 | 0.0399 |
| $\gamma_1-\alpha_1$ | -0.5889 | 0.2797 | -2.1056 | 0.0352 |
| $\beta_1$ | 0.4566 | 0.1677 | 2.7223 | 0.0065 |
| $p_1$ | 0.9629 | 0.0454 | 21.2086 | 0.0000 |
| $\mu_1$ | 0.0278 | 0.0575 | 0.4838 | 0.6285 |
| $\sigma_1^2$ | 0.6478 | 0.2282 | 2.8394 | 0.0045 |

### ARX(2,2)

#### Normal

##### GARCH(1,1)

No successful model.

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M038` | Normal | ARX(2,2) | GJR-GARCH(1,1) | -183.7821 | 385.5642 | 415.9000 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0822 +0.2087\,\pi_t +0.0102\,\pi_{t-1} +0.7570\,SPF_t +0.0279\,SPF_{t-1} + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0566 +0.5988\,(u_{t-1}^+)^2 +0.0044\,(u_{t-1}^-)^2 +0.6116\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0822 | 0.1120 | 0.7342 | 0.4628 |
| $\rho_1$ | 0.2087 | 0.0935 | 2.2310 | 0.0257 |
| $\rho_2$ | 0.0102 | 0.1021 | 0.1000 | 0.9204 |
| $\phi_1$ | 0.7570 | 0.4131 | 1.8325 | 0.0669 |
| $\phi_2$ | 0.0279 | 0.3190 | 0.0874 | 0.9304 |
| $\omega$ | 0.0566 | 0.0580 | 0.9757 | 0.3292 |
| $\alpha_1$ | 0.5988 | 0.6262 | 0.9563 | 0.3389 |
| $\gamma_1-\alpha_1$ | -0.5945 | 0.5631 | -1.0557 | 0.2911 |
| $\beta_1$ | 0.6116 | 0.3941 | 1.5518 | 0.1207 |

#### Student's $t$

##### GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M085` | Student's $t$ | ARX(2,2) | GARCH(1,1) | -172.5010 | 363.0019 | 393.3377 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.0841 +0.2681\,\pi_t +0.1768\,\pi_{t-1} +0.3222\,SPF_t +0.1807\,SPF_{t-1} + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0730 +0.5164\,u_{t-1}^2 +0.4303\,\sigma_{t-1}^2
$$

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.2836 |

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.0841 | 0.0798 | 1.0530 | 0.2924 |
| $\rho_1$ | 0.2681 | 0.0925 | 2.8982 | 0.0038 |
| $\rho_2$ | 0.1768 | 0.0989 | 1.7884 | 0.0737 |
| $\phi_1$ | 0.3222 | 0.3874 | 0.8316 | 0.4057 |
| $\phi_2$ | 0.1807 | 0.3505 | 0.5156 | 0.6061 |
| $\omega$ | 0.0730 | 0.0408 | 1.7903 | 0.0734 |
| $\alpha_1$ | 0.5164 | 0.2519 | 2.0501 | 0.0404 |
| $\beta_1$ | 0.4303 | 0.1615 | 2.6639 | 0.0077 |
| $\nu$ | 4.2836 | 1.4494 | 2.9554 | 0.0031 |

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M086` | Student's $t$ | ARX(2,2) | GJR-GARCH(1,1) | -169.9900 | 359.9800 | 393.6864 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.1061 +0.2373\,\pi_t +0.1287\,\pi_{t-1} +0.4591\,SPF_t +0.1216\,SPF_{t-1} + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0622 +0.6620\,(u_{t-1}^+)^2 +0.1336\,(u_{t-1}^-)^2 +0.5170\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $\nu$ | 4.5814 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.1061 | 0.0810 | 1.3095 | 0.1904 |
| $\rho_1$ | 0.2373 | 0.0892 | 2.6599 | 0.0078 |
| $\rho_2$ | 0.1287 | 0.0980 | 1.3140 | 0.1888 |
| $\phi_1$ | 0.4591 | 0.4103 | 1.1191 | 0.2631 |
| $\phi_2$ | 0.1216 | 0.3701 | 0.3284 | 0.7426 |
| $\omega$ | 0.0622 | 0.0301 | 2.0638 | 0.0390 |
| $\alpha_1$ | 0.6620 | 0.3330 | 1.9881 | 0.0468 |
| $\gamma_1-\alpha_1$ | -0.5284 | 0.2661 | -1.9860 | 0.0470 |
| $\beta_1$ | 0.5170 | 0.1907 | 2.7106 | 0.0067 |
| $\nu$ | 4.5814 | 1.5473 | 2.9609 | 0.0031 |

#### Gaussian mixture

##### GARCH(1,1)

No successful model.

##### GJR-GARCH(1,1)

| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |
|---|---|---|---|---:|---:|---:|
| `M134` | Gaussian mixture | ARX(2,2) | GJR-GARCH(1,1) | -168.4801 | 360.9602 | 401.4079 |

Mean process:

$$
\hat{\pi}_{t+1} = 0.1043 +0.2348\,\pi_t +0.1514\,\pi_{t-1} +0.3929\,SPF_t +0.1444\,SPF_{t-1} + \mu_{t+1}
$$

Volatility process:

$$
\hat{\sigma}_t^2 = 0.0729 +0.8250\,(u_{t-1}^+)^2 +0.2282\,(u_{t-1}^-)^2 +0.4512\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

Distribution parameters:

| Parameter | Estimate |
|---|---:|
| $p_1$ | 0.9611 |
| $\mu_1$ | 0.0299 |
| $\sigma_1^2$ | 0.6371 |
| $p_2$ | 0.0389 |
| $\mu_2$ | -0.7376 |
| $\sigma_2^2$ | 9.4010 |

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.1043 | 0.0681 | 1.5304 | 0.1259 |
| $\rho_1$ | 0.2348 | 0.0990 | 2.3724 | 0.0177 |
| $\rho_2$ | 0.1514 | 0.0948 | 1.5959 | 0.1105 |
| $\phi_1$ | 0.3929 | 0.3600 | 1.0914 | 0.2751 |
| $\phi_2$ | 0.1444 | 0.3070 | 0.4703 | 0.6381 |
| $\omega$ | 0.0729 | 0.0494 | 1.4773 | 0.1396 |
| $\alpha_1$ | 0.8250 | 0.3999 | 2.0631 | 0.0391 |
| $\gamma_1-\alpha_1$ | -0.5968 | 0.2880 | -2.0721 | 0.0383 |
| $\beta_1$ | 0.4512 | 0.1784 | 2.5297 | 0.0114 |
| $p_1$ | 0.9611 | 0.0445 | 21.5747 | 0.0000 |
| $\mu_1$ | 0.0299 | 0.0597 | 0.5001 | 0.6170 |
| $\sigma_1^2$ | 0.6371 | 0.2281 | 2.7931 | 0.0052 |


## Excluded Model Combinations

The combinations below did not produce an optimizer-converged fit satisfying the stationarity proxy with the `1e-6` margin after the restart search. They are omitted from the model-selection tables and retained in `garch_family_failed_models.csv` for audit.

| Model ID | Distribution | Mean process | Volatility process | Attempts | Best attempted LogLik |
|---|---|---|---|---:|---:|
| `M025` | Normal | ARX(2,1) | GARCH(1,1) | 50 | -188.8527 |
| `M028` | Normal | ARX(2,1) | GARCH(1,2) | 50 | -188.8527 |
| `M037` | Normal | ARX(2,2) | GARCH(1,1) | 50 | -188.8355 |
| `M040` | Normal | ARX(2,2) | GARCH(1,2) | 50 | -188.8355 |
| `M049` | Student's $t$ | Constant | GARCH(1,1) | 50 | -181.9942 |
| `M052` | Student's $t$ | Constant | GARCH(1,2) | 50 | -181.9942 |
| `M097` | Gaussian mixture | Constant | GARCH(1,1) | 50 | -179.4873 |
| `M098` | Gaussian mixture | Constant | GJR-GARCH(1,1) | 50 | -175.2123 |
| `M100` | Gaussian mixture | Constant | GARCH(1,2) | 50 | -179.4873 |
| `M101` | Gaussian mixture | Constant | GJR-GARCH(1,2) | 50 | -175.2123 |
| `M103` | Gaussian mixture | Constant | GARCH(2,1) | 50 | -177.7161 |
| `M104` | Gaussian mixture | Constant | GJR-GARCH(2,1) | 50 | -171.8210 |
| `M106` | Gaussian mixture | Constant | GARCH(2,2) | 50 | -177.7161 |
| `M107` | Gaussian mixture | Constant | GJR-GARCH(2,2) | 50 | -171.8210 |
| `M109` | Gaussian mixture | ARX(1,1) | GARCH(1,1) | 50 | -174.5852 |
| `M112` | Gaussian mixture | ARX(1,1) | GARCH(1,2) | 50 | -174.5852 |
| `M121` | Gaussian mixture | ARX(2,1) | GARCH(1,1) | 50 | -171.3839 |
| `M124` | Gaussian mixture | ARX(2,1) | GARCH(1,2) | 50 | -171.3839 |
| `M127` | Gaussian mixture | ARX(2,1) | GARCH(2,1) | 50 | -170.7809 |
| `M130` | Gaussian mixture | ARX(2,1) | GARCH(2,2) | 50 | -170.6186 |
| `M133` | Gaussian mixture | ARX(2,2) | GARCH(1,1) | 50 | -171.1994 |
| `M139` | Gaussian mixture | ARX(2,2) | GARCH(2,1) | 50 | -170.7011 |


## EGARCH and Initialization Notes

- Effective sample is fixed at 215 observations for all models (`hold_back=0` with explicit lag regressors in the mean equation).
- In `arch`, EGARCH uses the package's centered term `|z|-sqrt(2/pi)` in the recursion for all distributions.
- Under non-Gaussian errors (e.g., Student's t or Gaussian mixture), this is an intercept reparameterization; fitted dynamics and likelihood are still valid.
- Conditional-variance recursion starts use the default initialization implemented by the `arch` package; this report no longer iterates the `backcast` to the parameter-implied unconditional variance.
- Each model combination is estimated from the package default start plus randomized feasible starts, and only optimizer-converged fits that pass the stationarity proxy with a `1e-6` margin below one are collected in the main result tables.
- Stability is monitored using a persistence proxy (`< 1 - 1e-6`):
  GARCH uses `sum(alpha)+sum(beta)`, GJR uses `sum(alpha)+0.5*sum(gamma)+sum(beta)`, EGARCH uses `sum(beta)`.
