```{raw:typst}
#set page(margin: auto)
```

# Model Selection Report

**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH AIC | GJR Mean | GJR Vol | GJR AIC | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | ARX(1,1) | (2,1) | 387.6789 | ARX(1,1) | (2,1) | 372.8425 | ARX(1,1) | (2,1) | 374.2995 |
| Student's $t$ | ARX(2,1) | (2,1) | 362.6718 | ARX(1,1) | (2,1) | 357.3153 | ARX(1,1) | (2,1) | 356.7451 |
| Gaussian mixture | ARX(2,1) | (1,2) | 365.6060 | ARX(2,1) | (2,1) | 359.0690 | ARX(1,1) | (2,1) | <span style="color:red">356.4744</span> |

**Table 3: Model selection by BIC: GARCH vs GJR vs EGARCH**

| Distribution | GARCH Mean | GARCH Vol | GARCH BIC | GJR Mean | GJR Vol | GJR BIC | EGARCH Mean | EGARCH Vol | EGARCH BIC |
|---|---|---|---:|---|---|---:|---|---|---:|
| Normal | Constant | (2,1) | 401.4156 | Constant | (2,1) | 396.6581 | Constant | (2,1) | 398.1066 |
| Student's $t$ | Constant | (2,1) | 387.7770 | Constant | (1,1) | 384.2113 | Constant | (1,1) | <span style="color:red">383.8115</span> |
| Gaussian mixture | ARX(1,1) | (2,1) | 401.6779 | Constant | (2,1) | 391.9670 | Constant | (1,1) | 385.2567 |

## Best Models by Criterion and Volatility Family

For each criterion and each volatility family, the selected model is the best across mean-process choices, orders, and distributions using stable & successful fits.

### AIC

#### GARCH
- Model ID: `M079`
- Distribution: Student's $t$
- Mean process: ARX(2,1) (`ARX_2_1`)
- Volatility process: GARCH(2,1)

$$
\hat{\pi}_{t+1} = 0.0754 +0.2410\,\pi_t +0.1539\,\pi_{t-1} +0.5650\,SPF_t + \mu_{t+1}
$$

Robust SE: $c$ (0.0804), $\rho_1$ (0.1565), $\rho_2$ (0.2555), $\phi_1$ (0.4729).

$$
\hat{\sigma}_t^2 = 0.1369 +0.4194\,u_{t-1}^2 +0.3468\,u_{t-2}^2 +0.0588\,\sigma_{t-1}^2
$$

- Number of observations: **215**
- Log-likelihood: **-172.335915**
- AIC: **362.671831**, BIC: **393.007573**
- Optimizer success: **True**

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.075394 | 0.080383 | 0.937930 | 0.348280 |
| $\rho_1$ | 0.241000 | 0.156521 | 1.539733 | 0.123625 |
| $\rho_2$ | 0.153880 | 0.255527 | 0.602208 | 0.547036 |
| $\phi_1$ | 0.565016 | 0.472887 | 1.194822 | 0.232156 |
| $\omega$ | 0.136928 | 0.470167 | 0.291233 | 0.770873 |
| $\alpha_1$ | 0.419351 | 0.260560 | 1.609418 | 0.107525 |
| $\alpha_2$ | 0.346759 | 1.053861 | 0.329037 | 0.742128 |
| $\beta_1$ | 0.058760 | 2.090575 | 0.028107 | 0.977577 |
| $\nu$ | 4.472871 | 1.614077 | 2.771164 | 0.005586 |

#### GJR-GARCH
- Model ID: `M068`
- Distribution: Student's $t$
- Mean process: ARX(1,1) (`ARX_1_1`)
- Volatility process: GJR-GARCH(2,1)

$$
\hat{\pi}_{t+1} = 0.0846 +0.1984\,\pi_t +0.7847\,SPF_t + \mu_{t+1}
$$

Robust SE: $c$ (0.0816), $\rho_1$ (0.0996), $\phi_1$ (0.1317).

$$
\hat{\sigma}_t^2 = 0.1168 +0.4252\,(u_{t-1}^+)^2 +0.2174\,(u_{t-1}^-)^2 +0.6575\,(u_{t-2}^+)^2 +0.0000\,(u_{t-2}^-)^2 +0.1354\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

- Number of observations: **215**
- Log-likelihood: **-168.657674**
- AIC: **357.315348**, BIC: **391.021728**
- Optimizer success: **True**

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.084561 | 0.081554 | 1.036868 | 0.299797 |
| $\rho_1$ | 0.198379 | 0.099598 | 1.991802 | 0.046393 |
| $\phi_1$ | 0.784679 | 0.131672 | 5.959354 | 0.000000 |
| $\omega$ | 0.116818 | 0.042038 | 2.778885 | 0.005455 |
| $\alpha_1$ | 0.425191 | 0.219441 | 1.937612 | 0.052671 |
| $\gamma_1-\alpha_1$ | -0.207829 | 0.251448 | -0.826528 | 0.408504 |
| $\alpha_2$ | 0.657467 | 0.374661 | 1.754833 | 0.079288 |
| $\gamma_2-\alpha_2$ | -0.657467 | 0.305776 | -2.150159 | 0.031543 |
| $\beta_1$ | 0.135397 | 0.172961 | 0.782816 | 0.433735 |
| $\nu$ | 5.349660 | 2.200946 | 2.430618 | 0.015073 |

#### EGARCH
- Model ID: `M117`
- Distribution: Gaussian mixture
- Mean process: ARX(1,1) (`ARX_1_1`)
- Volatility process: EGARCH(2,1)

$$
\hat{\pi}_{t+1} = 0.0861 +0.2148\,\pi_t +0.7325\,SPF_t + \mu_{t+1}
$$

Robust SE: $c$ (0.0317), $\rho_1$ (0.0705), $\phi_1$ (0.0017).

$$
\ln \hat{\sigma}_t^2 = -0.3146 +0.5764\,(|z_{t-1}|-\sqrt{2/\pi}) +0.0053\,(|z_{t-2}|-\sqrt{2/\pi}) +0.1006\,z_{t-1} +0.3061\,z_{t-2} +0.6993\,\ln\sigma_{t-1}^2
$$

- Number of observations: **215**
- Log-likelihood: **-166.237213**
- AIC: **356.474426**, BIC: **396.922083**
- Optimizer success: **True**

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $c$ | 0.086145 | 0.031694 | 2.718014 | 0.006568 |
| $\rho_1$ | 0.214824 | 0.070517 | 3.046416 | 0.002316 |
| $\phi_1$ | 0.732514 | 0.001739 | 421.149922 | 0.000000 |
| $\omega$ | -0.314557 | 0.158733 | -1.981672 | 0.047516 |
| $\alpha_1$ | 0.576377 | 0.147631 | 3.904164 | 0.000095 |
| $\alpha_2$ | 0.005343 | 0.238386 | 0.022413 | 0.982118 |
| $\gamma_1$ | 0.100565 | 0.099839 | 1.007278 | 0.313801 |
| $\gamma_2$ | 0.306148 | 0.099347 | 3.081593 | 0.002059 |
| $\beta_1$ | 0.699319 | 0.105518 | 6.627490 | 0.000000 |
| $p_1$ | 0.934951 | 0.041199 | 22.693639 | 0.000000 |
| $\mu_1$ | 0.048865 | 0.057085 | 0.855992 | 0.392002 |
| $\sigma_1^2$ | 0.625460 | 0.155173 | 4.030724 | 0.000056 |

### BIC

#### GARCH
- Model ID: `M055`
- Distribution: Student's $t$
- Mean process: Constant (`Constant_anchor`)
- Volatility process: GARCH(2,1)

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

$$
\hat{\sigma}_t^2 = 0.1221 +0.4616\,u_{t-1}^2 +0.4956\,u_{t-2}^2 +0.0000\,\sigma_{t-1}^2
$$

- Number of observations: **215**
- Log-likelihood: **-180.461891**
- AIC: **370.923781**, BIC: **387.776971**
- Optimizer success: **True**

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.122062 | 0.045238 | 2.698212 | 0.006971 |
| $\alpha_1$ | 0.461596 | 0.179330 | 2.574004 | 0.010053 |
| $\alpha_2$ | 0.495589 | 0.229219 | 2.162077 | 0.030612 |
| $\beta_1$ | 0.000000 | 0.151356 | 0.000000 | 1.000000 |
| $\nu$ | 4.807722 | 1.636402 | 2.937984 | 0.003304 |

#### GJR-GARCH
- Model ID: `M050`
- Distribution: Student's $t$
- Mean process: Constant (`Constant_anchor`)
- Volatility process: GJR-GARCH(1,1)

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

$$
\hat{\sigma}_t^2 = 0.0636 +0.7929\,(u_{t-1}^+)^2 +0.1465\,(u_{t-1}^-)^2 +0.4366\,\sigma_{t-1}^2
$$

Reported in README notation. (`arch` estimates the equivalent indicator form.)

- Number of observations: **215**
- Log-likelihood: **-178.679065**
- AIC: **367.358130**, BIC: **384.211320**
- Optimizer success: **True**

For GJR-GARCH, $\gamma_i-\alpha_i$ is the raw `arch` indicator coefficient; the equation above reports the negative-shock coefficient $\gamma_i$.

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | 0.063611 | 0.025467 | 2.497746 | 0.012499 |
| $\alpha_1$ | 0.792917 | 0.256260 | 3.094187 | 0.001974 |
| $\gamma_1-\alpha_1$ | -0.646461 | 0.231689 | -2.790208 | 0.005267 |
| $\beta_1$ | 0.436615 | 0.116958 | 3.733109 | 0.000189 |
| $\nu$ | 4.760505 | 1.533912 | 3.103507 | 0.001912 |

#### EGARCH
- Model ID: `M051`
- Distribution: Student's $t$
- Mean process: Constant (`Constant_anchor`)
- Volatility process: EGARCH(1,1)

$$
\hat{\pi}_{t+1} = SPF_t + \mu_{t+1}
$$

No mean-process coefficients are estimated in this anchored specification.

$$
\ln \hat{\sigma}_t^2 = -0.2520 +0.5595\,(|z_{t-1}|-\sqrt{2/\pi}) +0.2663\,z_{t-1} +0.7882\,\ln\sigma_{t-1}^2
$$

- Number of observations: **215**
- Log-likelihood: **-178.479161**
- AIC: **366.958322**, BIC: **383.811512**
- Optimizer success: **True**

| Parameter | Coef | Std Err | t-value | p-value |
|---|---:|---:|---:|---:|
| $\omega$ | -0.251989 | 0.105679 | -2.384477 | 0.017103 |
| $\alpha_1$ | 0.559468 | 0.139250 | 4.017715 | 0.000059 |
| $\gamma_1$ | 0.266302 | 0.077913 | 3.417950 | 0.000631 |
| $\beta_1$ | 0.788161 | 0.061181 | 12.882450 | 0.000000 |
| $\nu$ | 5.151108 | 1.764823 | 2.918768 | 0.003514 |

## EGARCH and Initialization Notes

- Effective sample is fixed at 215 observations for all models (`hold_back=0` with explicit lag regressors in the mean equation).
- In `arch`, EGARCH uses the package's centered term `|z|-sqrt(2/pi)` in the recursion for all distributions.
- Under non-Gaussian errors (e.g., Student's t or Gaussian mixture), this is an intercept reparameterization; fitted dynamics and likelihood are still valid.
- The recursion start (`backcast`) is updated iteratively during estimation in this script: fit -> implied initial variance from current parameters -> refit.
- Stability is monitored using a persistence proxy (`<1`):
  GARCH uses `sum(alpha)+sum(beta)`, GJR uses `sum(alpha)+0.5*sum(gamma)+sum(beta)`, EGARCH uses `sum(beta)`.
