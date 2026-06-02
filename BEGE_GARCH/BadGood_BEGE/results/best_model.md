```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-06-02T13:18:52`
Total estimations: `8000`
Converged estimations: `7841`
Eligible estimations for best-model selection: `7841`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `20` admissible estimates by corrected log likelihood. Standard errors are shown below substituted equation coefficients in parentheses.

```{note}
Flagged 107 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## constant

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 5 | 34 | -76.863954 | 169.727908 | 196.693012 | 867.260443 | 3.169626 | yes | `computed` |
| 2 | 8 | 1 | -95.900687 | 207.801374 | 234.766478 | 1520.179511 | 5.579027 | yes | `computed` |
| 3 | 47 | 13 | -97.335535 | 210.671070 | 237.636174 | 1083.813607 | 6.125048 | yes | `computed` |
| 4 | 19 | 2 | -108.585176 | 233.170353 | 260.135457 | 948.187840 | 2.327896 | yes | `computed` |
| 5 | 46 | 3 | -109.985324 | 235.970647 | 262.935752 | 61548.919879 | 3662.321713 | yes | `computed` |
| 6 | 14 | 20 | -117.853950 | 251.707900 | 278.673005 | 174.313864 | 8.347038 | yes | `computed` |
| 7 | 16 | 23 | -128.176823 | 272.353646 | 299.318751 | 12164.348950 | 5.257785 | yes | `computed` |
| 8 | 34 | 39 | -136.885780 | 289.771560 | 316.736664 | 953.424855 | 10.268280 | yes | `computed` |
| 9 | 23 | 35 | -138.452606 | 292.905213 | 319.870317 | 142.535126 | 8.373567 | yes | `computed` |
| 10 | 22 | 32 | -145.302230 | 306.604460 | 333.569565 | 1926.178015 | 3.389090 | yes | `computed` |
| 11 | 41 | 33 | -145.323491 | 306.646982 | 333.612086 | 2028.535018 | 19.463403 | yes | `computed` |
| 12 | 35 | 15 | -150.021615 | 316.043230 | 343.008335 | 2963.898021 | 2.963542 | no | `computed` |
| 13 | 8 | 14 | -150.204287 | 316.408573 | 343.373677 | 134.295836 | 6.575508 | no | `computed` |
| 14 | 23 | 34 | -159.181719 | 334.363437 | 361.328542 | 589.937274 | 2.870028 | no | `computed` |
| 15 | 4 | 29 | -161.048007 | 338.096014 | 365.061118 | 2202.631821 | 4.888881 | no | `computed` |
| 16 | 30 | 10 | -162.765267 | 341.530534 | 368.495638 | 168.207573 | 6.011504 | no | `computed` |
| 17 | 25 | 32 | -164.966670 | 345.933340 | 372.898444 | 4200.908914 | 3.719964 | no | `computed` |
| 18 | 30 | 21 | -170.682126 | 357.364252 | 384.329356 | 132.223170 | 6.300655 | no | `computed` |
| 19 | 43 | 34 | -171.343390 | 358.686780 | 385.651885 | 144.415497 | 7.798619 | no | `computed` |
| 20 | 16 | 11 | -172.511551 | 361.023102 | 387.988206 | 1207.236395 | 6.724274 | no | `computed` |

### Rank 1: Seed 5, Draw 34

- LogLik: `-76.863954`; AIC: `169.727908`; BIC: `196.693012`
- Max shape path: `867.260443`; max implied variance: `3.169626`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.031380}\,\omega_{p,t} - \underset{(0.000002)}{0.058727}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.008475)}{2.270537} + \underset{(0.000002)}{0.983074}\,p_{t-1} + \frac{\underset{(0.000002)}{0.002637}}{2(\underset{(0.000002)}{0.031380})^2}\,u_{t-1}^2,\\
n_t &= \underset{(5.809644)}{2.637307} + \underset{(0.028545)}{0.639193}\,n_{t-1} + \frac{\underset{(0.093734)}{0.348569}}{2(\underset{(0.000002)}{0.058727})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 2.270537 | 0.008475 |
| n0 | 2.637307 | 5.809644 |
| rho_p | 0.983074 | 0.000002 |
| rho_n | 0.639193 | 0.028545 |
| phi_p | 0.002637 | 0.000002 |
| phi_n | 0.348569 | 0.093734 |
| sigma_p | 0.031380 | 0.000002 |
| sigma_n | 0.058727 | 0.000002 |

### Rank 2: Seed 8, Draw 1

- LogLik: `-95.900687`; AIC: `207.801374`; BIC: `234.766478`
- Max shape path: `1520.179511`; max implied variance: `5.579027`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.022691}\,\omega_{p,t} - \underset{(0.004854)}{0.189781}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.942718)}{7.851649} + \underset{(0.002731)}{0.199140}\,p_{t-1} + \frac{\underset{(0.000008)}{0.096206}}{2(\underset{(0.000002)}{0.022691})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1.208422)}{9.784475} + \underset{(0.104374)}{0.269556}\,n_{t-1} + \frac{\underset{(0.247899)}{0.531636}}{2(\underset{(0.004854)}{0.189781})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 7.851649 | 0.942718 |
| n0 | 9.784475 | 1.208422 |
| rho_p | 0.199140 | 0.002731 |
| rho_n | 0.269556 | 0.104374 |
| phi_p | 0.096206 | 0.000008 |
| phi_n | 0.531636 | 0.247899 |
| sigma_p | 0.022691 | 0.000002 |
| sigma_n | 0.189781 | 0.004854 |

### Rank 3: Seed 47, Draw 13

- LogLik: `-97.335535`; AIC: `210.671070`; BIC: `237.636174`
- Max shape path: `1083.813607`; max implied variance: `6.125048`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.333981)}{0.032827}\,\omega_{p,t} - \underset{(0.510248)}{0.094885}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(18.360111)}{9.919752} + \underset{(0.135491)}{0.825077}\,p_{t-1} + \frac{\underset{(0.312339)}{0.120216}}{2(\underset{(0.333981)}{0.032827})^2}\,u_{t-1}^2,\\
n_t &= \underset{(5.841164)}{3.864534} + \underset{(0.057311)}{0.410286}\,n_{t-1} + \frac{\underset{(0.057824)}{0.583847}}{2(\underset{(0.510248)}{0.094885})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.919752 | 18.360111 |
| n0 | 3.864534 | 5.841164 |
| rho_p | 0.825077 | 0.135491 |
| rho_n | 0.410286 | 0.057311 |
| phi_p | 0.120216 | 0.312339 |
| phi_n | 0.583847 | 0.057824 |
| sigma_p | 0.032827 | 0.333981 |
| sigma_n | 0.094885 | 0.510248 |

### Rank 4: Seed 19, Draw 2

- LogLik: `-108.585176`; AIC: `233.170353`; BIC: `260.135457`
- Max shape path: `948.187840`; max implied variance: `2.327896`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000022)}{0.028391}\,\omega_{p,t} - \underset{(0.259101)}{0.105153}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1169.573075)}{7.379104} + \underset{(8.768755)}{0.821244}\,p_{t-1} + \frac{\underset{(1.054678)}{0.079647}}{2(\underset{(0.000022)}{0.028391})^2}\,u_{t-1}^2,\\
n_t &= \underset{(528.148541)}{6.148920} + \underset{(14.756094)}{0.739039}\,n_{t-1} + \frac{\underset{(0.075479)}{0.148665}}{2(\underset{(0.259101)}{0.105153})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 7.379104 | 1169.573075 |
| n0 | 6.148920 | 528.148541 |
| rho_p | 0.821244 | 8.768755 |
| rho_n | 0.739039 | 14.756094 |
| phi_p | 0.079647 | 1.054678 |
| phi_n | 0.148665 | 0.075479 |
| sigma_p | 0.028391 | 0.000022 |
| sigma_n | 0.105153 | 0.259101 |

### Rank 5: Seed 46, Draw 3

- LogLik: `-109.985324`; AIC: `235.970647`; BIC: `262.935752`
- Max shape path: `61548.919879`; max implied variance: `3662.321713`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(8.065829)}{0.074653}\,\omega_{p,t} - \underset{(0.399560)}{0.243931}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.996150)}{0.220103} + \underset{(86.347094)}{0.691252}\,p_{t-1} + \frac{\underset{(27.313443)}{0.109814}}{2(\underset{(8.065829)}{0.074653})^2}\,u_{t-1}^2,\\
n_t &= \underset{(512.889385)}{0.693730} + \underset{(0.149202)}{0.249436}\,n_{t-1} + \frac{\underset{(0.146740)}{0.750553}}{2(\underset{(0.399560)}{0.243931})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.220103 | 3.996150 |
| n0 | 0.693730 | 512.889385 |
| rho_p | 0.691252 | 86.347094 |
| rho_n | 0.249436 | 0.149202 |
| phi_p | 0.109814 | 27.313443 |
| phi_n | 0.750553 | 0.146740 |
| sigma_p | 0.074653 | 8.065829 |
| sigma_n | 0.243931 | 0.399560 |

### Rank 6: Seed 14, Draw 20

- LogLik: `-117.853950`; AIC: `251.707900`; BIC: `278.673005`
- Max shape path: `174.313864`; max implied variance: `8.347038`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.089399}\,\omega_{p,t} - \underset{(0.000765)}{0.232485}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.042965)}{4.771565} + \underset{(0.003239)}{0.499618}\,p_{t-1} + \frac{\underset{(0.000764)}{0.158651}}{2(\underset{(0.000002)}{0.089399})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.006020)}{0.006532} + \underset{(0.000184)}{0.137587}\,n_{t-1} + \frac{\underset{(0.000128)}{0.862284}}{2(\underset{(0.000765)}{0.232485})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 4.771565 | 0.042965 |
| n0 | 0.006532 | 0.006020 |
| rho_p | 0.499618 | 0.003239 |
| rho_n | 0.137587 | 0.000184 |
| phi_p | 0.158651 | 0.000764 |
| phi_n | 0.862284 | 0.000128 |
| sigma_p | 0.089399 | 0.000002 |
| sigma_n | 0.232485 | 0.000765 |

### Rank 7: Seed 16, Draw 23

- LogLik: `-128.176823`; AIC: `272.353646`; BIC: `299.318751`
- Max shape path: `12164.348950`; max implied variance: `5.257785`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.739807)}{0.137759}\,\omega_{p,t} - \underset{(0.001602)}{0.007495}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(900.193600)}{9.107851} + \underset{(0.027031)}{0.214406}\,p_{t-1} + \frac{\underset{(1148.241720)}{0.538316}}{2(\underset{(1.739807)}{0.137759})^2}\,u_{t-1}^2,\\
n_t &= \underset{(543.322372)}{0.964315} + \underset{(1.715165)}{0.559720}\,n_{t-1} + \frac{\underset{(0.022315)}{0.081549}}{2(\underset{(0.001602)}{0.007495})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.107851 | 900.193600 |
| n0 | 0.964315 | 543.322372 |
| rho_p | 0.214406 | 0.027031 |
| rho_n | 0.559720 | 1.715165 |
| phi_p | 0.538316 | 1148.241720 |
| phi_n | 0.081549 | 0.022315 |
| sigma_p | 0.137759 | 1.739807 |
| sigma_n | 0.007495 | 0.001602 |

### Rank 8: Seed 34, Draw 39

- LogLik: `-136.885780`; AIC: `289.771560`; BIC: `316.736664`
- Max shape path: `953.424855`; max implied variance: `10.268280`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.055402}\,\omega_{p,t} - \underset{(0.069907)}{0.365482}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(11.168220)}{8.558710} + \underset{(0.003399)}{0.604759}\,p_{t-1} + \frac{\underset{(0.071911)}{0.338578}}{2(\underset{(0.000002)}{0.055402})^2}\,u_{t-1}^2,\\
n_t &= \underset{(2.656900)}{5.335059} + \underset{(0.071956)}{0.867861}\,n_{t-1} + \frac{\underset{(0.095265)}{0.059455}}{2(\underset{(0.069907)}{0.365482})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 8.558710 | 11.168220 |
| n0 | 5.335059 | 2.656900 |
| rho_p | 0.604759 | 0.003399 |
| rho_n | 0.867861 | 0.071956 |
| phi_p | 0.338578 | 0.071911 |
| phi_n | 0.059455 | 0.095265 |
| sigma_p | 0.055402 | 0.000002 |
| sigma_n | 0.365482 | 0.069907 |

### Rank 9: Seed 23, Draw 35

- LogLik: `-138.452606`; AIC: `292.905213`; BIC: `319.870317`
- Max shape path: `142.535126`; max implied variance: `8.373567`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.106080}\,\omega_{p,t} - \underset{(1.340908)}{0.226115}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1037.847659)}{8.100573} + \underset{(23.539129)}{0.318474}\,p_{t-1} + \frac{\underset{(1.346660)}{0.180558}}{2(\underset{(0.000002)}{0.106080})^2}\,u_{t-1}^2,\\
n_t &= \underset{(508.747048)}{0.043556} + \underset{(31.113961)}{0.156621}\,n_{t-1} + \frac{\underset{(12.483895)}{0.838625}}{2(\underset{(1.340908)}{0.226115})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 8.100573 | 1037.847659 |
| n0 | 0.043556 | 508.747048 |
| rho_p | 0.318474 | 23.539129 |
| rho_n | 0.156621 | 31.113961 |
| phi_p | 0.180558 | 1.346660 |
| phi_n | 0.838625 | 12.483895 |
| sigma_p | 0.106080 | 0.000002 |
| sigma_n | 0.226115 | 1.340908 |

### Rank 10: Seed 22, Draw 32

- LogLik: `-145.302230`; AIC: `306.604460`; BIC: `333.569565`
- Max shape path: `1926.178015`; max implied variance: `3.389090`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000234)}{0.022432}\,\omega_{p,t} - \underset{(0.097886)}{0.230997}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(7721.702131)}{1.123836} + \underset{(20.514503)}{0.148353}\,p_{t-1} + \frac{\underset{(2.718398)}{0.120071}}{2(\underset{(0.000234)}{0.022432})^2}\,u_{t-1}^2,\\
n_t &= \underset{(53.809603)}{4.489887} + \underset{(2.874539)}{0.642055}\,n_{t-1} + \frac{\underset{(0.939993)}{0.205598}}{2(\underset{(0.097886)}{0.230997})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 1.123836 | 7721.702131 |
| n0 | 4.489887 | 53.809603 |
| rho_p | 0.148353 | 20.514503 |
| rho_n | 0.642055 | 2.874539 |
| phi_p | 0.120071 | 2.718398 |
| phi_n | 0.205598 | 0.939993 |
| sigma_p | 0.022432 | 0.000234 |
| sigma_n | 0.230997 | 0.097886 |

### Rank 11: Seed 41, Draw 33

- LogLik: `-145.323491`; AIC: `306.646982`; BIC: `333.612086`
- Max shape path: `2028.535018`; max implied variance: `19.463403`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.005276)}{0.161326}\,\omega_{p,t} - \underset{(0.000011)}{0.019543}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.476109)}{9.918717} + \underset{(0.006004)}{0.549033}\,p_{t-1} + \frac{\underset{(0.009328)}{0.437702}}{2(\underset{(0.005276)}{0.161326})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.110455)}{1.643506} + \underset{(0.000064)}{0.582536}\,n_{t-1} + \frac{\underset{(0.000051)}{0.091928}}{2(\underset{(0.000011)}{0.019543})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.918717 | 0.476109 |
| n0 | 1.643506 | 0.110455 |
| rho_p | 0.549033 | 0.006004 |
| rho_n | 0.582536 | 0.000064 |
| phi_p | 0.437702 | 0.009328 |
| phi_n | 0.091928 | 0.000051 |
| sigma_p | 0.161326 | 0.005276 |
| sigma_n | 0.019543 | 0.000011 |

### Rank 12: Seed 35, Draw 15

- LogLik: `-150.021615`; AIC: `316.043230`; BIC: `343.008335`
- Max shape path: `2963.898021`; max implied variance: `2.963542`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.018177)}{0.226426}\,\omega_{p,t} - \underset{(0.000004)}{0.020806}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.620462)}{5.781321} + \underset{(0.000000)}{0.723366}\,p_{t-1} + \frac{\underset{(1.209000)}{0.069899}}{2(\underset{(0.018177)}{0.226426})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.340289)}{5.262270} + \underset{(0.000002)}{0.693767}\,n_{t-1} + \frac{\underset{(0.000007)}{0.147807}}{2(\underset{(0.000004)}{0.020806})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 5.781321 | 3.620462 |
| n0 | 5.262270 | 0.340289 |
| rho_p | 0.723366 | 0.000000 |
| rho_n | 0.693767 | 0.000002 |
| phi_p | 0.069899 | 1.209000 |
| phi_n | 0.147807 | 0.000007 |
| sigma_p | 0.226426 | 0.018177 |
| sigma_n | 0.020806 | 0.000004 |

### Rank 13: Seed 8, Draw 14

- LogLik: `-150.204287`; AIC: `316.408573`; BIC: `343.373677`
- Max shape path: `134.295836`; max implied variance: `6.575508`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001648)}{0.098668}\,\omega_{p,t} - \underset{(0.001745)}{0.202341}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(352.582916)}{9.991022} + \underset{(21.903686)}{0.091113}\,p_{t-1} + \frac{\underset{(0.001713)}{0.149038}}{2(\underset{(0.001648)}{0.098668})^2}\,u_{t-1}^2,\\
n_t &= \underset{(373.292975)}{1.250770} + \underset{(15.358354)}{0.303172}\,n_{t-1} + \frac{\underset{(0.001765)}{0.638701}}{2(\underset{(0.001745)}{0.202341})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.991022 | 352.582916 |
| n0 | 1.250770 | 373.292975 |
| rho_p | 0.091113 | 21.903686 |
| rho_n | 0.303172 | 15.358354 |
| phi_p | 0.149038 | 0.001713 |
| phi_n | 0.638701 | 0.001765 |
| sigma_p | 0.098668 | 0.001648 |
| sigma_n | 0.202341 | 0.001745 |

### Rank 14: Seed 23, Draw 34

- LogLik: `-159.181719`; AIC: `334.363437`; BIC: `361.328542`
- Max shape path: `589.937274`; max implied variance: `2.870028`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.523332)}{0.172158}\,\omega_{p,t} - \underset{(0.000213)}{0.039360}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(2.208720)}{5.423059} + \underset{(0.006047)}{0.849784}\,p_{t-1} + \frac{\underset{(0.669681)}{0.094042}}{2(\underset{(0.523332)}{0.172158})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1.502840)}{1.343587} + \underset{(0.006064)}{0.644756}\,n_{t-1} + \frac{\underset{(0.000002)}{0.106586}}{2(\underset{(0.000213)}{0.039360})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 5.423059 | 2.208720 |
| n0 | 1.343587 | 1.502840 |
| rho_p | 0.849784 | 0.006047 |
| rho_n | 0.644756 | 0.006064 |
| phi_p | 0.094042 | 0.669681 |
| phi_n | 0.106586 | 0.000002 |
| sigma_p | 0.172158 | 0.523332 |
| sigma_n | 0.039360 | 0.000213 |

### Rank 15: Seed 4, Draw 29

- LogLik: `-161.048007`; AIC: `338.096014`; BIC: `365.061118`
- Max shape path: `2202.631821`; max implied variance: `4.888881`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.003564)}{0.043193}\,\omega_{p,t} - \underset{(1.423672)}{0.071534}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(41721.130698)}{0.062874} + \underset{(49.019599)}{0.452183}\,p_{t-1} + \frac{\underset{(249.870110)}{0.498072}}{2(\underset{(0.003564)}{0.043193})^2}\,u_{t-1}^2,\\
n_t &= \underset{(42.048898)}{9.565028} + \underset{(1.422322)}{0.791802}\,n_{t-1} + \frac{\underset{(0.000005)}{0.060550}}{2(\underset{(1.423672)}{0.071534})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.062874 | 41721.130698 |
| n0 | 9.565028 | 42.048898 |
| rho_p | 0.452183 | 49.019599 |
| rho_n | 0.791802 | 1.422322 |
| phi_p | 0.498072 | 249.870110 |
| phi_n | 0.060550 | 0.000005 |
| sigma_p | 0.043193 | 0.003564 |
| sigma_n | 0.071534 | 1.423672 |

### Rank 16: Seed 30, Draw 10

- LogLik: `-162.765267`; AIC: `341.530534`; BIC: `368.495638`
- Max shape path: `168.207573`; max implied variance: `6.011504`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1303.440927)}{0.032026}\,\omega_{p,t} - \underset{(2811.808550)}{0.189373}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(107.627823)}{0.919110} + \underset{(0.469709)}{0.994513}\,p_{t-1} + \frac{\underset{(0.126119)}{0.000017}}{2(\underset{(1303.440927)}{0.032026})^2}\,u_{t-1}^2,\\
n_t &= \underset{(5915.910176)}{0.000312} + \underset{(34277.321771)}{0.280917}\,n_{t-1} + \frac{\underset{(34788.295506)}{0.719051}}{2(\underset{(2811.808550)}{0.189373})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.919110 | 107.627823 |
| n0 | 0.000312 | 5915.910176 |
| rho_p | 0.994513 | 0.469709 |
| rho_n | 0.280917 | 34277.321771 |
| phi_p | 0.000017 | 0.126119 |
| phi_n | 0.719051 | 34788.295506 |
| sigma_p | 0.032026 | 1303.440927 |
| sigma_n | 0.189373 | 2811.808550 |

### Rank 17: Seed 25, Draw 32

- LogLik: `-164.966670`; AIC: `345.933340`; BIC: `372.898444`
- Max shape path: `4200.908914`; max implied variance: `3.719964`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000090)}{0.022643}\,\omega_{p,t} - \underset{(0.236266)}{0.261582}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(175.271902)}{7.156245} + \underset{(0.099325)}{0.356725}\,p_{t-1} + \frac{\underset{(0.451666)}{0.262986}}{2(\underset{(0.000090)}{0.022643})^2}\,u_{t-1}^2,\\
n_t &= \underset{(37.905820)}{4.457450} + \underset{(1.865741)}{0.787709}\,n_{t-1} + \frac{\underset{(0.288725)}{0.014423}}{2(\underset{(0.236266)}{0.261582})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 7.156245 | 175.271902 |
| n0 | 4.457450 | 37.905820 |
| rho_p | 0.356725 | 0.099325 |
| rho_n | 0.787709 | 1.865741 |
| phi_p | 0.262986 | 0.451666 |
| phi_n | 0.014423 | 0.288725 |
| sigma_p | 0.022643 | 0.000090 |
| sigma_n | 0.261582 | 0.236266 |

### Rank 18: Seed 30, Draw 21

- LogLik: `-170.682126`; AIC: `357.364252`; BIC: `384.329356`
- Max shape path: `132.223170`; max implied variance: `6.300655`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.240312}\,\omega_{p,t} - \underset{(0.000003)}{0.091881}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(23.952889)}{0.324488} + \underset{(16.492112)}{0.259613}\,p_{t-1} + \frac{\underset{(0.008474)}{0.636197}}{2(\underset{(0.000002)}{0.240312})^2}\,u_{t-1}^2,\\
n_t &= \underset{(40.341813)}{6.990926} + \underset{(0.431059)}{0.555210}\,n_{t-1} + \frac{\underset{(0.000002)}{0.117490}}{2(\underset{(0.000003)}{0.091881})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.324488 | 23.952889 |
| n0 | 6.990926 | 40.341813 |
| rho_p | 0.259613 | 16.492112 |
| rho_n | 0.555210 | 0.431059 |
| phi_p | 0.636197 | 0.008474 |
| phi_n | 0.117490 | 0.000002 |
| sigma_p | 0.240312 | 0.000002 |
| sigma_n | 0.091881 | 0.000003 |

### Rank 19: Seed 43, Draw 34

- LogLik: `-171.343390`; AIC: `358.686780`; BIC: `385.651885`
- Max shape path: `144.415497`; max implied variance: `7.798619`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.002728)}{0.117116}\,\omega_{p,t} - \underset{(0.001862)}{0.204537}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1029.801550)}{7.410245} + \underset{(0.002084)}{0.147403}\,p_{t-1} + \frac{\underset{(0.002252)}{0.203029}}{2(\underset{(0.002728)}{0.117116})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1029.800508)}{0.016927} + \underset{(0.002596)}{0.254020}\,n_{t-1} + \frac{\underset{(0.002562)}{0.745162}}{2(\underset{(0.001862)}{0.204537})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 7.410245 | 1029.801550 |
| n0 | 0.016927 | 1029.800508 |
| rho_p | 0.147403 | 0.002084 |
| rho_n | 0.254020 | 0.002596 |
| phi_p | 0.203029 | 0.002252 |
| phi_n | 0.745162 | 0.002562 |
| sigma_p | 0.117116 | 0.002728 |
| sigma_n | 0.204537 | 0.001862 |

### Rank 20: Seed 16, Draw 11

- LogLik: `-172.511551`; AIC: `361.023102`; BIC: `387.988206`
- Max shape path: `1207.236395`; max implied variance: `6.724274`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.161415)}{0.341777}\,\omega_{p,t} - \underset{(0.000046)}{0.046570}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(15.690289)}{6.256788} + \underset{(0.000014)}{0.808606}\,p_{t-1} + \frac{\underset{(2.883316)}{0.031607}}{2(\underset{(0.161415)}{0.341777})^2}\,u_{t-1}^2,\\
n_t &= \underset{(2.750173)}{6.054967} + \underset{(0.000046)}{0.466783}\,n_{t-1} + \frac{\underset{(0.000022)}{0.313819}}{2(\underset{(0.000046)}{0.046570})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 6.256788 | 15.690289 |
| n0 | 6.054967 | 2.750173 |
| rho_p | 0.808606 | 0.000014 |
| rho_n | 0.466783 | 0.000046 |
| phi_p | 0.031607 | 2.883316 |
| phi_n | 0.313819 | 0.000022 |
| sigma_p | 0.341777 | 0.161415 |
| sigma_n | 0.046570 | 0.000046 |

## ARX(1,1)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 16 | 24 | 134.033609 | -246.067218 | -208.990199 | 3024.746008 | 11.591510 | yes | `computed` |
| 2 | 25 | 21 | -45.494390 | 112.988780 | 150.065798 | 5826.439656 | 3.575903 | yes | `computed` |
| 3 | 27 | 22 | -54.652832 | 131.305664 | 168.382682 | 726.516171 | 16.268748 | yes | `computed` |
| 4 | 38 | 5 | -75.025748 | 172.051495 | 209.128514 | 169.896634 | 6.238326 | yes | `computed` |
| 5 | 6 | 34 | -80.401075 | 182.802151 | 219.879169 | 8005.029925 | 3.056479 | yes | `computed` |
| 6 | 13 | 38 | -83.282128 | 188.564255 | 225.641274 | 693.888728 | 2.939347 | yes | `computed` |
| 7 | 29 | 6 | -90.805228 | 203.610457 | 240.687475 | 2369.101890 | 5.117885 | yes | `computed` |
| 8 | 16 | 11 | -96.846150 | 215.692299 | 252.769317 | 4029.034588 | 6.134566 | yes | `computed` |
| 9 | 10 | 28 | -98.687506 | 219.375011 | 256.452029 | 364.419808 | 7.010941 | yes | `computed` |
| 10 | 8 | 12 | -106.845960 | 235.691920 | 272.768938 | 719.545577 | 1.479354 | yes | `computed` |
| 11 | 34 | 15 | -115.635411 | 253.270821 | 290.347839 | 147.870665 | 2.863474 | yes | `computed` |
| 12 | 35 | 35 | -116.037733 | 254.075466 | 291.152484 | 1801.592978 | 4.559126 | yes | `computed` |
| 13 | 40 | 40 | -116.374451 | 254.748902 | 291.825920 | 153.424492 | 5.067660 | yes | `computed` |
| 14 | 26 | 31 | -121.777924 | 265.555849 | 302.632867 | 36828.498097 | 3.638906 | yes | `computed` |
| 15 | 32 | 14 | -126.033200 | 274.066401 | 311.143419 | 18392.097474 | 3.273369 | yes | `computed` |
| 16 | 18 | 8 | -132.605379 | 287.210759 | 324.287777 | 462.117689 | 5.916924 | yes | `computed` |
| 17 | 5 | 6 | -134.374481 | 290.748962 | 327.825980 | 143.679475 | 6.077848 | yes | `computed` |
| 18 | 37 | 10 | -139.063501 | 300.127002 | 337.204021 | 16737.427325 | 5.180386 | yes | `computed` |
| 19 | 22 | 35 | -141.232005 | 304.464009 | 341.541027 | 146.687705 | 5.217560 | yes | `computed` |
| 20 | 10 | 20 | -141.697460 | 305.394920 | 342.471939 | 141.233867 | 4.358484 | yes | `computed` |

### Rank 1: Seed 16, Draw 24

- LogLik: `134.033609`; AIC: `-246.067218`; BIC: `-208.990199`
- Max shape path: `3024.746008`; max implied variance: `11.591510`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.166266)}{0.378918} + \underset{(0.168882)}{0.445463}\,\pi_t + \underset{(0.363165)}{1.135712}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(3.400131)}{0.271844}\,\omega_{p,t} - \underset{(0.000202)}{0.048968}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.252231)}{4.246196} + \underset{(0.000002)}{0.921285}\,p_{t-1} + \frac{\underset{(0.131516)}{0.022016}}{2(\underset{(3.400131)}{0.271844})^2}\,u_{t-1}^2,\\
n_t &= \underset{(610.137957)}{9.681746} + \underset{(0.000010)}{0.232777}\,n_{t-1} + \frac{\underset{(2.932584)}{0.626957}}{2(\underset{(0.000202)}{0.048968})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.378918 | 0.166266 |
| rho_1 | 0.445463 | 0.168882 |
| phi_1 | 1.135712 | 0.363165 |
| p0 | 4.246196 | 3.252231 |
| n0 | 9.681746 | 610.137957 |
| rho_p | 0.921285 | 0.000002 |
| rho_n | 0.232777 | 0.000010 |
| phi_p | 0.022016 | 0.131516 |
| phi_n | 0.626957 | 2.932584 |
| sigma_p | 0.271844 | 3.400131 |
| sigma_n | 0.048968 | 0.000202 |

### Rank 2: Seed 25, Draw 21

- LogLik: `-45.494390`; AIC: `112.988780`; BIC: `150.065798`
- Max shape path: `5826.439656`; max implied variance: `3.575903`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(5.743572)}{0.077521} + \underset{(7.121810)}{0.382617}\,\pi_t + \underset{(21.814943)}{0.677602}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000000)}{0.016682}\,\omega_{p,t} - \underset{(0.182096)}{0.097515}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(841.418891)}{9.421150} + \underset{(0.392293)}{0.665092}\,p_{t-1} + \frac{\underset{(5.005505)}{0.178494}}{2(\underset{(0.000000)}{0.016682})^2}\,u_{t-1}^2,\\
n_t &= \underset{(10.908074)}{6.048270} + \underset{(0.036787)}{0.733906}\,n_{t-1} + \frac{\underset{(0.624381)}{0.188749}}{2(\underset{(0.182096)}{0.097515})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.077521 | 5.743572 |
| rho_1 | 0.382617 | 7.121810 |
| phi_1 | 0.677602 | 21.814943 |
| p0 | 9.421150 | 841.418891 |
| n0 | 6.048270 | 10.908074 |
| rho_p | 0.665092 | 0.392293 |
| rho_n | 0.733906 | 0.036787 |
| phi_p | 0.178494 | 5.005505 |
| phi_n | 0.188749 | 0.624381 |
| sigma_p | 0.016682 | 0.000000 |
| sigma_n | 0.097515 | 0.182096 |

### Rank 3: Seed 27, Draw 22

- LogLik: `-54.652832`; AIC: `131.305664`; BIC: `168.382682`
- Max shape path: `726.516171`; max implied variance: `16.268748`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(2.359752)}{0.086861} + \underset{(2.421809)}{0.359913}\,\pi_t + \underset{(4.744872)}{0.609843}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000046)}{0.040600}\,\omega_{p,t} - \underset{(0.122833)}{0.148771}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(5.201824)}{9.999913} + \underset{(0.122891)}{0.852835}\,p_{t-1} + \frac{\underset{(0.000003)}{0.059870}}{2(\underset{(0.000046)}{0.040600})^2}\,u_{t-1}^2,\\
n_t &= \underset{(2.707527)}{0.003265} + \underset{(0.011986)}{0.250377}\,n_{t-1} + \frac{\underset{(0.011986)}{0.749619}}{2(\underset{(0.122833)}{0.148771})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.086861 | 2.359752 |
| rho_1 | 0.359913 | 2.421809 |
| phi_1 | 0.609843 | 4.744872 |
| p0 | 9.999913 | 5.201824 |
| n0 | 0.003265 | 2.707527 |
| rho_p | 0.852835 | 0.122891 |
| rho_n | 0.250377 | 0.011986 |
| phi_p | 0.059870 | 0.000003 |
| phi_n | 0.749619 | 0.011986 |
| sigma_p | 0.040600 | 0.000046 |
| sigma_n | 0.148771 | 0.122833 |

### Rank 4: Seed 38, Draw 5

- LogLik: `-75.025748`; AIC: `172.051495`; BIC: `209.128514`
- Max shape path: `169.896634`; max implied variance: `6.238326`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(273.900588)}{0.101891} + \underset{(46.205918)}{0.367629}\,\pi_t + \underset{(95.785601)}{0.609300}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(25.813409)}{0.215479}\,\omega_{p,t} - \underset{(0.000021)}{0.076014}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(242.701127)}{2.382816} + \underset{(48.437181)}{0.114423}\,p_{t-1} + \frac{\underset{(202.180497)}{0.602979}}{2(\underset{(25.813409)}{0.215479})^2}\,u_{t-1}^2,\\
n_t &= \underset{(214.141383)}{4.925862} + \underset{(62.794663)}{0.648157}\,n_{t-1} + \frac{\underset{(10.433231)}{0.101198}}{2(\underset{(0.000021)}{0.076014})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.101891 | 273.900588 |
| rho_1 | 0.367629 | 46.205918 |
| phi_1 | 0.609300 | 95.785601 |
| p0 | 2.382816 | 242.701127 |
| n0 | 4.925862 | 214.141383 |
| rho_p | 0.114423 | 48.437181 |
| rho_n | 0.648157 | 62.794663 |
| phi_p | 0.602979 | 202.180497 |
| phi_n | 0.101198 | 10.433231 |
| sigma_p | 0.215479 | 25.813409 |
| sigma_n | 0.076014 | 0.000021 |

### Rank 5: Seed 6, Draw 34

- LogLik: `-80.401075`; AIC: `182.802151`; BIC: `219.879169`
- Max shape path: `8005.029925`; max implied variance: `3.056479`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.435063)}{0.346872} + \underset{(1.062392)}{0.499570}\,\pi_t + \underset{(2.339341)}{0.093800}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.064274)}{0.210631}\,\omega_{p,t} - \underset{(0.000031)}{0.014850}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(2.039298)}{3.822369} + \underset{(0.141358)}{0.768891}\,p_{t-1} + \frac{\underset{(0.408142)}{0.059272}}{2(\underset{(0.064274)}{0.210631})^2}\,u_{t-1}^2,\\
n_t &= \underset{(35.488758)}{8.400677} + \underset{(0.255587)}{0.735106}\,n_{t-1} + \frac{\underset{(0.575944)}{0.189896}}{2(\underset{(0.000031)}{0.014850})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.346872 | 0.435063 |
| rho_1 | 0.499570 | 1.062392 |
| phi_1 | 0.093800 | 2.339341 |
| p0 | 3.822369 | 2.039298 |
| n0 | 8.400677 | 35.488758 |
| rho_p | 0.768891 | 0.141358 |
| rho_n | 0.735106 | 0.255587 |
| phi_p | 0.059272 | 0.408142 |
| phi_n | 0.189896 | 0.575944 |
| sigma_p | 0.210631 | 0.064274 |
| sigma_n | 0.014850 | 0.000031 |

### Rank 6: Seed 13, Draw 38

- LogLik: `-83.282128`; AIC: `188.564255`; BIC: `225.641274`
- Max shape path: `693.888728`; max implied variance: `2.939347`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1.817956)}{0.018498} + \underset{(0.354400)}{0.090236}\,\pi_t + \underset{(0.586596)}{0.778131}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000914)}{0.032554}\,\omega_{p,t} - \underset{(8.302264)}{0.203354}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(109.273938)}{7.389909} + \underset{(0.000001)}{0.701341}\,p_{t-1} + \frac{\underset{(0.002939)}{0.083836}}{2(\underset{(0.000914)}{0.032554})^2}\,u_{t-1}^2,\\
n_t &= \underset{(79.629122)}{6.045925} + \underset{(0.000002)}{0.839245}\,n_{t-1} + \frac{\underset{(11.090470)}{0.070580}}{2(\underset{(8.302264)}{0.203354})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.018498 | 1.817956 |
| rho_1 | 0.090236 | 0.354400 |
| phi_1 | 0.778131 | 0.586596 |
| p0 | 7.389909 | 109.273938 |
| n0 | 6.045925 | 79.629122 |
| rho_p | 0.701341 | 0.000001 |
| rho_n | 0.839245 | 0.000002 |
| phi_p | 0.083836 | 0.002939 |
| phi_n | 0.070580 | 11.090470 |
| sigma_p | 0.032554 | 0.000914 |
| sigma_n | 0.203354 | 8.302264 |

### Rank 7: Seed 29, Draw 6

- LogLik: `-90.805228`; AIC: `203.610457`; BIC: `240.687475`
- Max shape path: `2369.101890`; max implied variance: `5.117885`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.022442)}{0.463869} + \underset{(0.003015)}{0.236983}\,\pi_t + \underset{(0.032045)}{1.599005}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.083132)}{0.241319}\,\omega_{p,t} - \underset{(0.000004)}{0.030605}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.915996)}{3.402791} + \underset{(0.008846)}{0.874727}\,p_{t-1} + \frac{\underset{(0.063869)}{0.085529}}{2(\underset{(0.083132)}{0.241319})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.917769)}{0.263598} + \underset{(0.010392)}{0.526180}\,n_{t-1} + \frac{\underset{(0.000075)}{0.171930}}{2(\underset{(0.000004)}{0.030605})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.463869 | 0.022442 |
| rho_1 | 0.236983 | 0.003015 |
| phi_1 | 1.599005 | 0.032045 |
| p0 | 3.402791 | 0.915996 |
| n0 | 0.263598 | 0.917769 |
| rho_p | 0.874727 | 0.008846 |
| rho_n | 0.526180 | 0.010392 |
| phi_p | 0.085529 | 0.063869 |
| phi_n | 0.171930 | 0.000075 |
| sigma_p | 0.241319 | 0.083132 |
| sigma_n | 0.030605 | 0.000004 |

### Rank 8: Seed 16, Draw 11

- LogLik: `-96.846150`; AIC: `215.692299`; BIC: `252.769317`
- Max shape path: `4029.034588`; max implied variance: `6.134566`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000001)}{0.257805} + \underset{(0.000013)}{0.339104}\,\pi_t + \underset{(0.000006)}{1.137621}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000011)}{0.029466}\,\omega_{p,t} - \underset{(0.000007)}{0.239100}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000013)}{6.035244} + \underset{(0.000004)}{0.438761}\,p_{t-1} + \frac{\underset{(0.000014)}{0.326266}}{2(\underset{(0.000011)}{0.029466})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.000015)}{6.776058} + \underset{(0.000016)}{0.758548}\,n_{t-1} + \frac{\underset{(0.000003)}{0.091935}}{2(\underset{(0.000007)}{0.239100})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.257805 | 0.000001 |
| rho_1 | 0.339104 | 0.000013 |
| phi_1 | 1.137621 | 0.000006 |
| p0 | 6.035244 | 0.000013 |
| n0 | 6.776058 | 0.000015 |
| rho_p | 0.438761 | 0.000004 |
| rho_n | 0.758548 | 0.000016 |
| phi_p | 0.326266 | 0.000014 |
| phi_n | 0.091935 | 0.000003 |
| sigma_p | 0.029466 | 0.000011 |
| sigma_n | 0.239100 | 0.000007 |

### Rank 9: Seed 10, Draw 28

- LogLik: `-98.687506`; AIC: `219.375011`; BIC: `256.452029`
- Max shape path: `364.419808`; max implied variance: `7.010941`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1.055472)}{0.143694} + \underset{(3.179614)}{0.311843}\,\pi_t + \underset{(5.445608)}{0.714754}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001311)}{0.031020}\,\omega_{p,t} - \underset{(0.262854)}{0.135844}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(6.394351)}{9.314380} + \underset{(0.001315)}{0.903611}\,p_{t-1} + \frac{\underset{(0.000004)}{0.017680}}{2(\underset{(0.001311)}{0.031020})^2}\,u_{t-1}^2,\\
n_t &= \underset{(12.411958)}{7.497473} + \underset{(4.753351)}{0.198243}\,n_{t-1} + \frac{\underset{(4.849876)}{0.743011}}{2(\underset{(0.262854)}{0.135844})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.143694 | 1.055472 |
| rho_1 | 0.311843 | 3.179614 |
| phi_1 | 0.714754 | 5.445608 |
| p0 | 9.314380 | 6.394351 |
| n0 | 7.497473 | 12.411958 |
| rho_p | 0.903611 | 0.001315 |
| rho_n | 0.198243 | 4.753351 |
| phi_p | 0.017680 | 0.000004 |
| phi_n | 0.743011 | 4.849876 |
| sigma_p | 0.031020 | 0.001311 |
| sigma_n | 0.135844 | 0.262854 |

### Rank 10: Seed 8, Draw 12

- LogLik: `-106.845960`; AIC: `235.691920`; BIC: `272.768938`
- Max shape path: `719.545577`; max implied variance: `1.479354`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000026)}{0.083429} + \underset{(0.237933)}{0.235354}\,\pi_t + \underset{(0.000029)}{0.779840}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000001)}{0.087669}\,\omega_{p,t} - \underset{(0.000075)}{0.034540}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000039)}{5.178299} + \underset{(0.000312)}{0.928892}\,p_{t-1} + \frac{\underset{(0.000293)}{0.005427}}{2(\underset{(0.000001)}{0.087669})^2}\,u_{t-1}^2,\\
n_t &= \underset{(20.836791)}{5.377522} + \underset{(0.238261)}{0.820104}\,n_{t-1} + \frac{\underset{(0.000062)}{0.086906}}{2(\underset{(0.000075)}{0.034540})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.083429 | 0.000026 |
| rho_1 | 0.235354 | 0.237933 |
| phi_1 | 0.779840 | 0.000029 |
| p0 | 5.178299 | 0.000039 |
| n0 | 5.377522 | 20.836791 |
| rho_p | 0.928892 | 0.000312 |
| rho_n | 0.820104 | 0.238261 |
| phi_p | 0.005427 | 0.000293 |
| phi_n | 0.086906 | 0.000062 |
| sigma_p | 0.087669 | 0.000001 |
| sigma_n | 0.034540 | 0.000075 |

### Rank 11: Seed 34, Draw 15

- LogLik: `-115.635411`; AIC: `253.270821`; BIC: `290.347839`
- Max shape path: `147.870665`; max implied variance: `2.863474`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(496.982562)}{0.159381} + \underset{(151.873002)}{0.449706}\,\pi_t + \underset{(1048.028556)}{0.370367}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.016424)}{0.040705}\,\omega_{p,t} - \underset{(11.966633)}{0.205085}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(259.398931)}{9.657799} + \underset{(36.735924)}{0.424848}\,p_{t-1} + \frac{\underset{(0.016413)}{0.025410}}{2(\underset{(0.016424)}{0.040705})^2}\,u_{t-1}^2,\\
n_t &= \underset{(174.153571)}{4.950329} + \underset{(1344.125480)}{0.418174}\,n_{t-1} + \frac{\underset{(95.130070)}{0.264639}}{2(\underset{(11.966633)}{0.205085})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.159381 | 496.982562 |
| rho_1 | 0.449706 | 151.873002 |
| phi_1 | 0.370367 | 1048.028556 |
| p0 | 9.657799 | 259.398931 |
| n0 | 4.950329 | 174.153571 |
| rho_p | 0.424848 | 36.735924 |
| rho_n | 0.418174 | 1344.125480 |
| phi_p | 0.025410 | 0.016413 |
| phi_n | 0.264639 | 95.130070 |
| sigma_p | 0.040705 | 0.016424 |
| sigma_n | 0.205085 | 11.966633 |

### Rank 12: Seed 35, Draw 35

- LogLik: `-116.037733`; AIC: `254.075466`; BIC: `291.152484`
- Max shape path: `1801.592978`; max implied variance: `4.559126`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.521823)}{0.133698} + \underset{(0.315638)}{0.332974}\,\pi_t + \underset{(0.000248)}{0.957927}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.206097)}{0.093493}\,\omega_{p,t} - \underset{(0.000006)}{0.015922}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.237455)}{4.269874} + \underset{(0.310074)}{0.720090}\,p_{t-1} + \frac{\underset{(0.587329)}{0.271708}}{2(\underset{(0.206097)}{0.093493})^2}\,u_{t-1}^2,\\
n_t &= \underset{(2.129242)}{7.636972} + \underset{(0.000004)}{0.732443}\,n_{t-1} + \frac{\underset{(0.000003)}{0.045085}}{2(\underset{(0.000006)}{0.015922})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.133698 | 0.521823 |
| rho_1 | 0.332974 | 0.315638 |
| phi_1 | 0.957927 | 0.000248 |
| p0 | 4.269874 | 1.237455 |
| n0 | 7.636972 | 2.129242 |
| rho_p | 0.720090 | 0.310074 |
| rho_n | 0.732443 | 0.000004 |
| phi_p | 0.271708 | 0.587329 |
| phi_n | 0.045085 | 0.000003 |
| sigma_p | 0.093493 | 0.206097 |
| sigma_n | 0.015922 | 0.000006 |

### Rank 13: Seed 40, Draw 40

- LogLik: `-116.374451`; AIC: `254.748902`; BIC: `291.825920`
- Max shape path: `153.424492`; max implied variance: `5.067660`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.128327} + \underset{(0.000044)}{0.317835}\,\pi_t + \underset{(0.000003)}{0.635187}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000004)}{0.069463}\,\omega_{p,t} - \underset{(0.000003)}{0.199633}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.039273)}{4.295817} + \underset{(0.000002)}{0.794685}\,p_{t-1} + \frac{\underset{(0.000005)}{0.068157}}{2(\underset{(0.000004)}{0.069463})^2}\,u_{t-1}^2,\\
n_t &= \underset{(6.793786)}{1.485757} + \underset{(0.515526)}{0.168638}\,n_{t-1} + \frac{\underset{(0.000008)}{0.497110}}{2(\underset{(0.000003)}{0.199633})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.128327 | 0.000002 |
| rho_1 | 0.317835 | 0.000044 |
| phi_1 | 0.635187 | 0.000003 |
| p0 | 4.295817 | 1.039273 |
| n0 | 1.485757 | 6.793786 |
| rho_p | 0.794685 | 0.000002 |
| rho_n | 0.168638 | 0.515526 |
| phi_p | 0.068157 | 0.000005 |
| phi_n | 0.497110 | 0.000008 |
| sigma_p | 0.069463 | 0.000004 |
| sigma_n | 0.199633 | 0.000003 |

### Rank 14: Seed 26, Draw 31

- LogLik: `-121.777924`; AIC: `265.555849`; BIC: `302.632867`
- Max shape path: `36828.498097`; max implied variance: `3.638906`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(15.253059)}{0.102852} + \underset{(6.855047)}{0.362021}\,\pi_t + \underset{(51.949445)}{0.569005}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.385104)}{0.044896}\,\omega_{p,t} - \underset{(0.004039)}{0.008019}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(35.350675)}{7.969809} + \underset{(5.207561)}{0.791950}\,p_{t-1} + \frac{\underset{(17.585817)}{0.128809}}{2(\underset{(0.385104)}{0.044896})^2}\,u_{t-1}^2,\\
n_t &= \underset{(10022.447514)}{9.945609} + \underset{(15.896702)}{0.585841}\,n_{t-1} + \frac{\underset{(6.204789)}{0.271935}}{2(\underset{(0.004039)}{0.008019})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.102852 | 15.253059 |
| rho_1 | 0.362021 | 6.855047 |
| phi_1 | 0.569005 | 51.949445 |
| p0 | 7.969809 | 35.350675 |
| n0 | 9.945609 | 10022.447514 |
| rho_p | 0.791950 | 5.207561 |
| rho_n | 0.585841 | 15.896702 |
| phi_p | 0.128809 | 17.585817 |
| phi_n | 0.271935 | 6.204789 |
| sigma_p | 0.044896 | 0.385104 |
| sigma_n | 0.008019 | 0.004039 |

### Rank 15: Seed 32, Draw 14

- LogLik: `-126.033200`; AIC: `274.066401`; BIC: `311.143419`
- Max shape path: `18392.097474`; max implied variance: `3.273369`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(8.485740)}{0.071773} + \underset{(0.003572)}{0.356474}\,\pi_t + \underset{(8.489323)}{0.478725}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.004501)}{0.157506}\,\omega_{p,t} - \underset{(0.000224)}{0.011795}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.003590)}{4.310604} + \underset{(5.054758)}{0.776877}\,p_{t-1} + \frac{\underset{(0.000001)}{0.026510}}{2(\underset{(0.004501)}{0.157506})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1646.777511)}{6.274928} + \underset{(0.004733)}{0.579327}\,n_{t-1} + \frac{\underset{(5.058793)}{0.305880}}{2(\underset{(0.000224)}{0.011795})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.071773 | 8.485740 |
| rho_1 | 0.356474 | 0.003572 |
| phi_1 | 0.478725 | 8.489323 |
| p0 | 4.310604 | 0.003590 |
| n0 | 6.274928 | 1646.777511 |
| rho_p | 0.776877 | 5.054758 |
| rho_n | 0.579327 | 0.004733 |
| phi_p | 0.026510 | 0.000001 |
| phi_n | 0.305880 | 5.058793 |
| sigma_p | 0.157506 | 0.004501 |
| sigma_n | 0.011795 | 0.000224 |

### Rank 16: Seed 18, Draw 8

- LogLik: `-132.605379`; AIC: `287.210759`; BIC: `324.287777`
- Max shape path: `462.117689`; max implied variance: `5.916924`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(6.567408)}{0.011052} + \underset{(50.619677)}{0.442330}\,\pi_t + \underset{(20.904547)}{0.559416}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000363)}{0.049541}\,\omega_{p,t} - \underset{(0.004423)}{0.130131}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(288.779457)}{3.602849} + \underset{(11.389075)}{0.746760}\,p_{t-1} + \frac{\underset{(0.004485)}{0.123166}}{2(\underset{(0.000363)}{0.049541})^2}\,u_{t-1}^2,\\
n_t &= \underset{(394.706046)}{4.302403} + \underset{(55.639624)}{0.238013}\,n_{t-1} + \frac{\underset{(1.752290)}{0.567119}}{2(\underset{(0.004423)}{0.130131})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.011052 | 6.567408 |
| rho_1 | 0.442330 | 50.619677 |
| phi_1 | 0.559416 | 20.904547 |
| p0 | 3.602849 | 288.779457 |
| n0 | 4.302403 | 394.706046 |
| rho_p | 0.746760 | 11.389075 |
| rho_n | 0.238013 | 55.639624 |
| phi_p | 0.123166 | 0.004485 |
| phi_n | 0.567119 | 1.752290 |
| sigma_p | 0.049541 | 0.000363 |
| sigma_n | 0.130131 | 0.004423 |

### Rank 17: Seed 5, Draw 6

- LogLik: `-134.374481`; AIC: `290.748962`; BIC: `327.825980`
- Max shape path: `143.679475`; max implied variance: `6.077848`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.436341)}{0.095947} + \underset{(2.143594)}{0.326207}\,\pi_t + \underset{(1.927454)}{0.705286}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.226340)}{0.035319}\,\omega_{p,t} - \underset{(0.254944)}{0.287022}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(10.312539)}{2.263426} + \underset{(0.065693)}{0.981895}\,p_{t-1} + \frac{\underset{(0.010474)}{0.001291}}{2(\underset{(0.226340)}{0.035319})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.031573)}{0.012603} + \underset{(0.474258)}{0.319572}\,n_{t-1} + \frac{\underset{(0.446807)}{0.678887}}{2(\underset{(0.254944)}{0.287022})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.095947 | 0.436341 |
| rho_1 | 0.326207 | 2.143594 |
| phi_1 | 0.705286 | 1.927454 |
| p0 | 2.263426 | 10.312539 |
| n0 | 0.012603 | 0.031573 |
| rho_p | 0.981895 | 0.065693 |
| rho_n | 0.319572 | 0.474258 |
| phi_p | 0.001291 | 0.010474 |
| phi_n | 0.678887 | 0.446807 |
| sigma_p | 0.035319 | 0.226340 |
| sigma_n | 0.287022 | 0.254944 |

### Rank 18: Seed 37, Draw 10

- LogLik: `-139.063501`; AIC: `300.127002`; BIC: `337.204021`
- Max shape path: `16737.427325`; max implied variance: `5.180386`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(3.846496)}{0.123275} + \underset{(9.123321)}{0.211208}\,\pi_t + \underset{(5.698046)}{0.827158}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000270)}{0.014669}\,\omega_{p,t} - \underset{(0.411301)}{0.160859}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3361.187508)}{0.595241} + \underset{(0.679557)}{0.174244}\,p_{t-1} + \frac{\underset{(12.025952)}{0.412924}}{2(\underset{(0.000270)}{0.014669})^2}\,u_{t-1}^2,\\
n_t &= \underset{(98.842810)}{7.073319} + \underset{(3.390129)}{0.581678}\,n_{t-1} + \frac{\underset{(1.084719)}{0.127203}}{2(\underset{(0.411301)}{0.160859})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.123275 | 3.846496 |
| rho_1 | 0.211208 | 9.123321 |
| phi_1 | 0.827158 | 5.698046 |
| p0 | 0.595241 | 3361.187508 |
| n0 | 7.073319 | 98.842810 |
| rho_p | 0.174244 | 0.679557 |
| rho_n | 0.581678 | 3.390129 |
| phi_p | 0.412924 | 12.025952 |
| phi_n | 0.127203 | 1.084719 |
| sigma_p | 0.014669 | 0.000270 |
| sigma_n | 0.160859 | 0.411301 |

### Rank 19: Seed 22, Draw 35

- LogLik: `-141.232005`; AIC: `304.464009`; BIC: `341.541027`
- Max shape path: `146.687705`; max implied variance: `5.217560`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1948.308831)}{0.174800} + \underset{(537.824618)}{0.289328}\,\pi_t + \underset{(2790.260238)}{0.780154}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.008135)}{0.077704}\,\omega_{p,t} - \underset{(0.000028)}{0.184491}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(5528.418635)}{2.407037} + \underset{(122.984174)}{0.815859}\,p_{t-1} + \frac{\underset{(0.007833)}{0.081279}}{2(\underset{(0.008135)}{0.077704})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1394.999281)}{1.530141} + \underset{(467.314367)}{0.283432}\,n_{t-1} + \frac{\underset{(0.167044)}{0.468753}}{2(\underset{(0.000028)}{0.184491})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.174800 | 1948.308831 |
| rho_1 | 0.289328 | 537.824618 |
| phi_1 | 0.780154 | 2790.260238 |
| p0 | 2.407037 | 5528.418635 |
| n0 | 1.530141 | 1394.999281 |
| rho_p | 0.815859 | 122.984174 |
| rho_n | 0.283432 | 467.314367 |
| phi_p | 0.081279 | 0.007833 |
| phi_n | 0.468753 | 0.167044 |
| sigma_p | 0.077704 | 0.008135 |
| sigma_n | 0.184491 | 0.000028 |

### Rank 20: Seed 10, Draw 20

- LogLik: `-141.697460`; AIC: `305.394920`; BIC: `342.471939`
- Max shape path: `141.233867`; max implied variance: `4.358484`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{-0.000230} + \underset{(0.000004)}{0.296193}\,\pi_t + \underset{(0.000001)}{0.657805}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.069280}\,\omega_{p,t} - \underset{(0.000002)}{0.186544}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000000)}{4.973751} + \underset{(0.000002)}{0.785458}\,p_{t-1} + \frac{\underset{(0.000002)}{0.063602}}{2(\underset{(0.000002)}{0.069280})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.000000)}{1.280865} + \underset{(0.000001)}{0.364621}\,n_{t-1} + \frac{\underset{(0.000002)}{0.444276}}{2(\underset{(0.000002)}{0.186544})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.000230 | 0.000002 |
| rho_1 | 0.296193 | 0.000004 |
| phi_1 | 0.657805 | 0.000001 |
| p0 | 4.973751 | 0.000000 |
| n0 | 1.280865 | 0.000000 |
| rho_p | 0.785458 | 0.000002 |
| rho_n | 0.364621 | 0.000001 |
| phi_p | 0.063602 | 0.000002 |
| phi_n | 0.444276 | 0.000002 |
| sigma_p | 0.069280 | 0.000002 |
| sigma_n | 0.186544 | 0.000002 |

## ARX(2,1)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 34 | 2 | 65.425865 | -106.851731 | -66.404074 | 2681.319149 | 2.901267 | yes | `computed` |
| 2 | 21 | 4 | -38.978702 | 101.957403 | 142.405060 | 3879.803639 | 7.343046 | yes | `computed` |
| 3 | 34 | 11 | -66.762579 | 157.525158 | 197.972814 | 7029.672813 | 11.252888 | yes | `computed` |
| 4 | 2 | 32 | -72.190224 | 168.380448 | 208.828105 | 178.018364 | 2.558309 | yes | `computed` |
| 5 | 8 | 1 | -92.524490 | 209.048981 | 249.496637 | 1885.253562 | 6.440858 | yes | `computed` |
| 6 | 6 | 17 | -97.172439 | 218.344877 | 258.792534 | 167.719733 | 7.411704 | yes | `computed` |
| 7 | 42 | 6 | -99.111119 | 222.222238 | 262.669895 | 6304.015999 | 6.987319 | yes | `computed` |
| 8 | 46 | 18 | -111.908248 | 247.816496 | 288.264152 | 174.379573 | 6.680092 | yes | `computed` |
| 9 | 34 | 5 | -119.504760 | 263.009520 | 303.457176 | 132.146965 | 7.602133 | yes | `computed` |
| 10 | 37 | 3 | -124.286119 | 272.572238 | 313.019894 | 39804.153031 | 9.242907 | yes | `computed` |
| 11 | 42 | 8 | -130.599930 | 285.199859 | 325.647516 | 137.835288 | 7.519326 | yes | `computed` |
| 12 | 41 | 33 | -140.028600 | 304.057200 | 344.504856 | 787.343413 | 19.899003 | yes | `computed` |
| 13 | 43 | 29 | -140.710286 | 305.420572 | 345.868228 | 146.013094 | 5.860258 | yes | `computed` |
| 14 | 8 | 28 | -142.441876 | 308.883752 | 349.331408 | 2223.906059 | 3.372832 | yes | `computed` |
| 15 | 49 | 40 | -145.491274 | 314.982548 | 355.430204 | 31619.281464 | 1839.566309 | yes | `computed` |
| 16 | 8 | 21 | -149.607728 | 323.215455 | 363.663112 | 220404.436247 | 7.016946 | yes | `computed` |
| 17 | 27 | 33 | -150.926065 | 325.852130 | 366.299787 | 164.582941 | 4.522415 | no | `computed` |
| 18 | 4 | 33 | -152.543626 | 329.087252 | 369.534908 | 134.632804 | 9.471505 | no | `computed` |
| 19 | 14 | 36 | -153.046409 | 330.092819 | 370.540475 | 159.849241 | 7.752837 | no | `computed` |
| 20 | 14 | 32 | -156.034070 | 336.068140 | 376.515796 | 1983.525858 | 4.448180 | no | `computed` |

### Rank 1: Seed 34, Draw 2

- LogLik: `65.425865`; AIC: `-106.851731`; BIC: `-66.404074`
- Max shape path: `2681.319149`; max implied variance: `2.901267`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(42.162690)}{-0.640609} + \underset{(1.045881)}{0.568106}\,\pi_t + \underset{(45.712137)}{0.065923}\,\pi_{t-1} + \underset{(89.095956)}{1.628255}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000492)}{0.024174}\,\omega_{p,t} - \underset{(3.585134)}{0.097351}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(82.152803)}{4.300604} + \underset{(0.000002)}{0.863914}\,p_{t-1} + \frac{\underset{(0.000002)}{0.083489}}{2(\underset{(0.000492)}{0.024174})^2}\,u_{t-1}^2,\\
n_t &= \underset{(146.243020)}{8.633965} + \underset{(0.000281)}{0.871306}\,n_{t-1} + \frac{\underset{(4.108763)}{0.036444}}{2(\underset{(3.585134)}{0.097351})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.640609 | 42.162690 |
| rho_1 | 0.568106 | 1.045881 |
| rho_2 | 0.065923 | 45.712137 |
| phi_1 | 1.628255 | 89.095956 |
| p0 | 4.300604 | 82.152803 |
| n0 | 8.633965 | 146.243020 |
| rho_p | 0.863914 | 0.000002 |
| rho_n | 0.871306 | 0.000281 |
| phi_p | 0.083489 | 0.000002 |
| phi_n | 0.036444 | 4.108763 |
| sigma_p | 0.024174 | 0.000492 |
| sigma_n | 0.097351 | 3.585134 |

### Rank 2: Seed 21, Draw 4

- LogLik: `-38.978702`; AIC: `101.957403`; BIC: `142.405060`
- Max shape path: `3879.803639`; max implied variance: `7.343046`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(133.759078)}{-0.295516} + \underset{(131.324091)}{0.188505}\,\pi_t + \underset{(122.619325)}{0.015717}\,\pi_{t-1} + \underset{(355.787643)}{-0.008748}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.002573)}{0.030375}\,\omega_{p,t} - \underset{(24.900141)}{0.375567}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(18578.681407)}{3.446004} + \underset{(9.556833)}{0.467251}\,p_{t-1} + \frac{\underset{(19.961003)}{0.328018}}{2(\underset{(0.002573)}{0.030375})^2}\,u_{t-1}^2,\\
n_t &= \underset{(205.986154)}{3.489810} + \underset{(14.104322)}{0.854431}\,n_{t-1} + \frac{\underset{(54.400618)}{0.015730}}{2(\underset{(24.900141)}{0.375567})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.295516 | 133.759078 |
| rho_1 | 0.188505 | 131.324091 |
| rho_2 | 0.015717 | 122.619325 |
| phi_1 | -0.008748 | 355.787643 |
| p0 | 3.446004 | 18578.681407 |
| n0 | 3.489810 | 205.986154 |
| rho_p | 0.467251 | 9.556833 |
| rho_n | 0.854431 | 14.104322 |
| phi_p | 0.328018 | 19.961003 |
| phi_n | 0.015730 | 54.400618 |
| sigma_p | 0.030375 | 0.002573 |
| sigma_n | 0.375567 | 24.900141 |

### Rank 3: Seed 34, Draw 11

- LogLik: `-66.762579`; AIC: `157.525158`; BIC: `197.972814`
- Max shape path: `7029.672813`; max implied variance: `11.252888`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.164597)}{0.588458} + \underset{(0.156105)}{0.424659}\,\pi_t + \underset{(0.000001)}{0.209958}\,\pi_{t-1} + \underset{(0.164598)}{0.122050}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000018)}{0.314423}\,\omega_{p,t} - \underset{(0.000021)}{0.033006}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.020145)}{5.523491} + \underset{(0.166503)}{0.817999}\,p_{t-1} + \frac{\underset{(0.156115)}{0.047638}}{2(\underset{(0.000018)}{0.314423})^2}\,u_{t-1}^2,\\
n_t &= \underset{(45.163190)}{0.570582} + \underset{(1.020135)}{0.007286}\,n_{t-1} + \frac{\underset{(0.166493)}{0.678814}}{2(\underset{(0.000021)}{0.033006})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.588458 | 0.164597 |
| rho_1 | 0.424659 | 0.156105 |
| rho_2 | 0.209958 | 0.000001 |
| phi_1 | 0.122050 | 0.164598 |
| p0 | 5.523491 | 1.020145 |
| n0 | 0.570582 | 45.163190 |
| rho_p | 0.817999 | 0.166503 |
| rho_n | 0.007286 | 1.020135 |
| phi_p | 0.047638 | 0.156115 |
| phi_n | 0.678814 | 0.166493 |
| sigma_p | 0.314423 | 0.000018 |
| sigma_n | 0.033006 | 0.000021 |

### Rank 4: Seed 2, Draw 32

- LogLik: `-72.190224`; AIC: `168.380448`; BIC: `208.828105`
- Max shape path: `178.018364`; max implied variance: `2.558309`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(12.361372)}{0.133147} + \underset{(18.647624)}{0.331288}\,\pi_t + \underset{(60.907104)}{0.149878}\,\pi_{t-1} + \underset{(106.924176)}{0.671700}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000028)}{0.040932}\,\omega_{p,t} - \underset{(2.828311)}{0.141988}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(266.546354)}{2.891717} + \underset{(1.394575)}{0.957764}\,p_{t-1} + \frac{\underset{(0.000000)}{0.012760}}{2(\underset{(0.000028)}{0.040932})^2}\,u_{t-1}^2,\\
n_t &= \underset{(652.486778)}{5.914930} + \underset{(60.952500)}{0.250195}\,n_{t-1} + \frac{\underset{(1.432219)}{0.210882}}{2(\underset{(2.828311)}{0.141988})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.133147 | 12.361372 |
| rho_1 | 0.331288 | 18.647624 |
| rho_2 | 0.149878 | 60.907104 |
| phi_1 | 0.671700 | 106.924176 |
| p0 | 2.891717 | 266.546354 |
| n0 | 5.914930 | 652.486778 |
| rho_p | 0.957764 | 1.394575 |
| rho_n | 0.250195 | 60.952500 |
| phi_p | 0.012760 | 0.000000 |
| phi_n | 0.210882 | 1.432219 |
| sigma_p | 0.040932 | 0.000028 |
| sigma_n | 0.141988 | 2.828311 |

### Rank 5: Seed 8, Draw 1

- LogLik: `-92.524490`; AIC: `209.048981`; BIC: `249.496637`
- Max shape path: `1885.253562`; max implied variance: `6.440858`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(6.955667)}{0.212374} + \underset{(9.978872)}{0.202864}\,\pi_t + \underset{(8.985222)}{0.035718}\,\pi_{t-1} + \underset{(5.398771)}{0.722230}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000140)}{0.022091}\,\omega_{p,t} - \underset{(0.752679)}{0.196116}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(14.971401)}{8.014245} + \underset{(6.575025)}{0.181151}\,p_{t-1} + \frac{\underset{(0.058316)}{0.100607}}{2(\underset{(0.000140)}{0.022091})^2}\,u_{t-1}^2,\\
n_t &= \underset{(20.934092)}{9.777479} + \underset{(6.895081)}{0.243178}\,n_{t-1} + \frac{\underset{(10.111358)}{0.551071}}{2(\underset{(0.752679)}{0.196116})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.212374 | 6.955667 |
| rho_1 | 0.202864 | 9.978872 |
| rho_2 | 0.035718 | 8.985222 |
| phi_1 | 0.722230 | 5.398771 |
| p0 | 8.014245 | 14.971401 |
| n0 | 9.777479 | 20.934092 |
| rho_p | 0.181151 | 6.575025 |
| rho_n | 0.243178 | 6.895081 |
| phi_p | 0.100607 | 0.058316 |
| phi_n | 0.551071 | 10.111358 |
| sigma_p | 0.022091 | 0.000140 |
| sigma_n | 0.196116 | 0.752679 |

### Rank 6: Seed 6, Draw 17

- LogLik: `-97.172439`; AIC: `218.344877`; BIC: `258.792534`
- Max shape path: `167.719733`; max implied variance: `7.411704`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.247807)}{0.045474} + \underset{(0.161309)}{-0.059905}\,\pi_t + \underset{(0.172013)}{0.178912}\,\pi_{t-1} + \underset{(0.432681)}{1.189193}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000023)}{0.065698}\,\omega_{p,t} - \underset{(0.014349)}{0.341479}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.970818)}{5.205102} + \underset{(0.038418)}{0.649152}\,p_{t-1} + \frac{\underset{(0.000025)}{0.064362}}{2(\underset{(0.000023)}{0.065698})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1.420003)}{1.930779} + \underset{(0.197980)}{0.227134}\,n_{t-1} + \frac{\underset{(0.049016)}{0.639238}}{2(\underset{(0.014349)}{0.341479})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.045474 | 0.247807 |
| rho_1 | -0.059905 | 0.161309 |
| rho_2 | 0.178912 | 0.172013 |
| phi_1 | 1.189193 | 0.432681 |
| p0 | 5.205102 | 1.970818 |
| n0 | 1.930779 | 1.420003 |
| rho_p | 0.649152 | 0.038418 |
| rho_n | 0.227134 | 0.197980 |
| phi_p | 0.064362 | 0.000025 |
| phi_n | 0.639238 | 0.049016 |
| sigma_p | 0.065698 | 0.000023 |
| sigma_n | 0.341479 | 0.014349 |

### Rank 7: Seed 42, Draw 6

- LogLik: `-99.111119`; AIC: `222.222238`; BIC: `262.669895`
- Max shape path: `6304.015999`; max implied variance: `6.987319`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(13.321100)}{0.169288} + \underset{(51.052001)}{0.301163}\,\pi_t + \underset{(109.914253)}{0.105244}\,\pi_{t-1} + \underset{(38.822286)}{0.478101}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000274)}{0.032660}\,\omega_{p,t} - \underset{(3.111403)}{0.046582}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(6713.329512)}{0.654170} + \underset{(0.438498)}{0.221036}\,p_{t-1} + \frac{\underset{(53.649663)}{0.735441}}{2(\underset{(0.000274)}{0.032660})^2}\,u_{t-1}^2,\\
n_t &= \underset{(27.310450)}{8.476222} + \underset{(1.115370)}{0.908702}\,n_{t-1} + \frac{\underset{(1.115315)}{0.005430}}{2(\underset{(3.111403)}{0.046582})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.169288 | 13.321100 |
| rho_1 | 0.301163 | 51.052001 |
| rho_2 | 0.105244 | 109.914253 |
| phi_1 | 0.478101 | 38.822286 |
| p0 | 0.654170 | 6713.329512 |
| n0 | 8.476222 | 27.310450 |
| rho_p | 0.221036 | 0.438498 |
| rho_n | 0.908702 | 1.115370 |
| phi_p | 0.735441 | 53.649663 |
| phi_n | 0.005430 | 1.115315 |
| sigma_p | 0.032660 | 0.000274 |
| sigma_n | 0.046582 | 3.111403 |

### Rank 8: Seed 46, Draw 18

- LogLik: `-111.908248`; AIC: `247.816496`; BIC: `288.264152`
- Max shape path: `174.379573`; max implied variance: `6.680092`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(73.644202)}{0.110569} + \underset{(165.907768)}{0.471075}\,\pi_t + \underset{(2.621340)}{0.067257}\,\pi_{t-1} + \underset{(75.344558)}{0.530101}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001041)}{0.195212}\,\omega_{p,t} - \underset{(0.000991)}{0.085798}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.041477)}{0.000510} + \underset{(2.603865)}{0.421839}\,p_{t-1} + \frac{\underset{(2.620668)}{0.577932}}{2(\underset{(0.001041)}{0.195212})^2}\,u_{t-1}^2,\\
n_t &= \underset{(246.542688)}{8.928987} + \underset{(15.995233)}{0.585679}\,n_{t-1} + \frac{\underset{(0.000018)}{0.118593}}{2(\underset{(0.000991)}{0.085798})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.110569 | 73.644202 |
| rho_1 | 0.471075 | 165.907768 |
| rho_2 | 0.067257 | 2.621340 |
| phi_1 | 0.530101 | 75.344558 |
| p0 | 0.000510 | 0.041477 |
| n0 | 8.928987 | 246.542688 |
| rho_p | 0.421839 | 2.603865 |
| rho_n | 0.585679 | 15.995233 |
| phi_p | 0.577932 | 2.620668 |
| phi_n | 0.118593 | 0.000018 |
| sigma_p | 0.195212 | 0.001041 |
| sigma_n | 0.085798 | 0.000991 |

### Rank 9: Seed 34, Draw 5

- LogLik: `-119.504760`; AIC: `263.009520`; BIC: `303.457176`
- Max shape path: `132.146965`; max implied variance: `7.602133`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.004434)}{0.153446} + \underset{(0.004884)}{0.165469}\,\pi_t + \underset{(0.034572)}{0.219561}\,\pi_{t-1} + \underset{(0.014120)}{0.430268}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.027963)}{0.086583}\,\omega_{p,t} - \underset{(0.005095)}{0.309110}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.012253)}{3.560994} + \underset{(0.170676)}{0.720759}\,p_{t-1} + \frac{\underset{(0.065648)}{0.090290}}{2(\underset{(0.027963)}{0.086583})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.004428)}{0.003901} + \underset{(0.027352)}{0.271269}\,n_{t-1} + \frac{\underset{(0.042916)}{0.695997}}{2(\underset{(0.005095)}{0.309110})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.153446 | 0.004434 |
| rho_1 | 0.165469 | 0.004884 |
| rho_2 | 0.219561 | 0.034572 |
| phi_1 | 0.430268 | 0.014120 |
| p0 | 3.560994 | 0.012253 |
| n0 | 0.003901 | 0.004428 |
| rho_p | 0.720759 | 0.170676 |
| rho_n | 0.271269 | 0.027352 |
| phi_p | 0.090290 | 0.065648 |
| phi_n | 0.695997 | 0.042916 |
| sigma_p | 0.086583 | 0.027963 |
| sigma_n | 0.309110 | 0.005095 |

### Rank 10: Seed 37, Draw 3

- LogLik: `-124.286119`; AIC: `272.572238`; BIC: `313.019894`
- Max shape path: `39804.153031`; max implied variance: `9.242907`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.285953)}{0.424845} + \underset{(22.946638)}{0.200963}\,\pi_t + \underset{(5.057441)}{0.060367}\,\pi_{t-1} + \underset{(12.387043)}{0.440677}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000513)}{0.128858}\,\omega_{p,t} - \underset{(0.000609)}{0.014747}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.378508)}{0.486249} + \underset{(0.000770)}{0.977660}\,p_{t-1} + \frac{\underset{(0.000001)}{0.012910}}{2(\underset{(0.000513)}{0.128858})^2}\,u_{t-1}^2,\\
n_t &= \underset{(536.491766)}{0.075647} + \underset{(11.082629)}{0.000000}\,n_{t-1} + \frac{\underset{(52.494304)}{0.915715}}{2(\underset{(0.000609)}{0.014747})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.424845 | 0.285953 |
| rho_1 | 0.200963 | 22.946638 |
| rho_2 | 0.060367 | 5.057441 |
| phi_1 | 0.440677 | 12.387043 |
| p0 | 0.486249 | 1.378508 |
| n0 | 0.075647 | 536.491766 |
| rho_p | 0.977660 | 0.000770 |
| rho_n | 0.000000 | 11.082629 |
| phi_p | 0.012910 | 0.000001 |
| phi_n | 0.915715 | 52.494304 |
| sigma_p | 0.128858 | 0.000513 |
| sigma_n | 0.014747 | 0.000609 |

### Rank 11: Seed 42, Draw 8

- LogLik: `-130.599930`; AIC: `285.199859`; BIC: `325.647516`
- Max shape path: `137.835288`; max implied variance: `7.519326`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.241095} + \underset{(0.001259)}{0.151668}\,\pi_t + \underset{(0.000002)}{0.118079}\,\pi_{t-1} + \underset{(0.000002)}{0.704665}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.070429}\,\omega_{p,t} - \underset{(0.000000)}{0.412308}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000000)}{5.837181} + \underset{(0.000002)}{0.718763}\,p_{t-1} + \frac{\underset{(0.000002)}{0.057578}}{2(\underset{(0.000002)}{0.070429})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.000002)}{0.442270} + \underset{(0.001259)}{0.200608}\,n_{t-1} + \frac{\underset{(0.000002)}{0.694649}}{2(\underset{(0.000000)}{0.412308})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.241095 | 0.000002 |
| rho_1 | 0.151668 | 0.001259 |
| rho_2 | 0.118079 | 0.000002 |
| phi_1 | 0.704665 | 0.000002 |
| p0 | 5.837181 | 0.000000 |
| n0 | 0.442270 | 0.000002 |
| rho_p | 0.718763 | 0.000002 |
| rho_n | 0.200608 | 0.001259 |
| phi_p | 0.057578 | 0.000002 |
| phi_n | 0.694649 | 0.000002 |
| sigma_p | 0.070429 | 0.000002 |
| sigma_n | 0.412308 | 0.000000 |

### Rank 12: Seed 41, Draw 33

- LogLik: `-140.028600`; AIC: `304.057200`; BIC: `344.504856`
- Max shape path: `787.343413`; max implied variance: `19.899003`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1.170402)}{0.084839} + \underset{(0.469239)}{0.359972}\,\pi_t + \underset{(0.764226)}{0.018206}\,\pi_{t-1} + \underset{(0.226510)}{0.629463}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.260163)}{0.158939}\,\omega_{p,t} - \underset{(0.000002)}{0.043749}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(398.224448)}{9.923285} + \underset{(3.619507)}{0.574752}\,p_{t-1} + \frac{\underset{(1.421760)}{0.412645}}{2(\underset{(0.260163)}{0.158939})^2}\,u_{t-1}^2,\\
n_t &= \underset{(34.356338)}{1.800057} + \underset{(0.253324)}{0.549253}\,n_{t-1} + \frac{\underset{(0.006722)}{0.087679}}{2(\underset{(0.000002)}{0.043749})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.084839 | 1.170402 |
| rho_1 | 0.359972 | 0.469239 |
| rho_2 | 0.018206 | 0.764226 |
| phi_1 | 0.629463 | 0.226510 |
| p0 | 9.923285 | 398.224448 |
| n0 | 1.800057 | 34.356338 |
| rho_p | 0.574752 | 3.619507 |
| rho_n | 0.549253 | 0.253324 |
| phi_p | 0.412645 | 1.421760 |
| phi_n | 0.087679 | 0.006722 |
| sigma_p | 0.158939 | 0.260163 |
| sigma_n | 0.043749 | 0.000002 |

### Rank 13: Seed 43, Draw 29

- LogLik: `-140.710286`; AIC: `305.420572`; BIC: `345.868228`
- Max shape path: `146.013094`; max implied variance: `5.860258`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(52.474094)}{0.155990} + \underset{(22.164007)}{0.307032}\,\pi_t + \underset{(95.040179)}{0.166638}\,\pi_{t-1} + \underset{(209.309717)}{0.504707}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000238)}{0.084657}\,\omega_{p,t} - \underset{(0.002921)}{0.188863}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(382.917535)}{3.674167} + \underset{(3.297454)}{0.705368}\,p_{t-1} + \frac{\underset{(0.000318)}{0.095115}}{2(\underset{(0.000238)}{0.084657})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1026.660689)}{1.511303} + \underset{(161.369927)}{0.281491}\,n_{t-1} + \frac{\underset{(3.066362)}{0.488776}}{2(\underset{(0.002921)}{0.188863})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.155990 | 52.474094 |
| rho_1 | 0.307032 | 22.164007 |
| rho_2 | 0.166638 | 95.040179 |
| phi_1 | 0.504707 | 209.309717 |
| p0 | 3.674167 | 382.917535 |
| n0 | 1.511303 | 1026.660689 |
| rho_p | 0.705368 | 3.297454 |
| rho_n | 0.281491 | 161.369927 |
| phi_p | 0.095115 | 0.000318 |
| phi_n | 0.488776 | 3.066362 |
| sigma_p | 0.084657 | 0.000238 |
| sigma_n | 0.188863 | 0.002921 |

### Rank 14: Seed 8, Draw 28

- LogLik: `-142.441876`; AIC: `308.883752`; BIC: `349.331408`
- Max shape path: `2223.906059`; max implied variance: `3.372832`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(10.304165)}{-0.009763} + \underset{(27.546351)}{0.679694}\,\pi_t + \underset{(7.236924)}{0.229717}\,\pi_{t-1} + \underset{(90.584017)}{-0.011810}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000018)}{0.031949}\,\omega_{p,t} - \underset{(0.000795)}{0.083123}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(41.801033)}{3.780019} + \underset{(11.088869)}{0.567725}\,p_{t-1} + \frac{\underset{(11.021301)}{0.237554}}{2(\underset{(0.000018)}{0.031949})^2}\,u_{t-1}^2,\\
n_t &= \underset{(100.090246)}{6.935235} + \underset{(0.000805)}{0.777218}\,n_{t-1} + \frac{\underset{(0.000000)}{0.086207}}{2(\underset{(0.000795)}{0.083123})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.009763 | 10.304165 |
| rho_1 | 0.679694 | 27.546351 |
| rho_2 | 0.229717 | 7.236924 |
| phi_1 | -0.011810 | 90.584017 |
| p0 | 3.780019 | 41.801033 |
| n0 | 6.935235 | 100.090246 |
| rho_p | 0.567725 | 11.088869 |
| rho_n | 0.777218 | 0.000805 |
| phi_p | 0.237554 | 11.021301 |
| phi_n | 0.086207 | 0.000000 |
| sigma_p | 0.031949 | 0.000018 |
| sigma_n | 0.083123 | 0.000795 |

### Rank 15: Seed 49, Draw 40

- LogLik: `-145.491274`; AIC: `314.982548`; BIC: `355.430204`
- Max shape path: `31619.281464`; max implied variance: `1839.566309`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(6200.809191)}{-0.082572} + \underset{(1926.183230)}{0.224202}\,\pi_t + \underset{(283.357935)}{0.170911}\,\pi_{t-1} + \underset{(1893.262104)}{0.553739}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(152.612840)}{0.090116}\,\omega_{p,t} - \underset{(910.584044)}{0.241192}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(103201.282631)}{8.823624} + \underset{(641.815345)}{0.414617}\,p_{t-1} + \frac{\underset{(49.034761)}{0.120220}}{2(\underset{(152.612840)}{0.090116})^2}\,u_{t-1}^2,\\
n_t &= \underset{(116998.423367)}{0.747992} + \underset{(4810.047381)}{0.271040}\,n_{t-1} + \frac{\underset{(4804.701582)}{0.728936}}{2(\underset{(910.584044)}{0.241192})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.082572 | 6200.809191 |
| rho_1 | 0.224202 | 1926.183230 |
| rho_2 | 0.170911 | 283.357935 |
| phi_1 | 0.553739 | 1893.262104 |
| p0 | 8.823624 | 103201.282631 |
| n0 | 0.747992 | 116998.423367 |
| rho_p | 0.414617 | 641.815345 |
| rho_n | 0.271040 | 4810.047381 |
| phi_p | 0.120220 | 49.034761 |
| phi_n | 0.728936 | 4804.701582 |
| sigma_p | 0.090116 | 152.612840 |
| sigma_n | 0.241192 | 910.584044 |

### Rank 16: Seed 8, Draw 21

- LogLik: `-149.607728`; AIC: `323.215455`; BIC: `363.663112`
- Max shape path: `220404.436247`; max implied variance: `7.016946`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1177.103957)}{0.109070} + \underset{(777.545196)}{0.566139}\,\pi_t + \underset{(225.635674)}{0.092806}\,\pi_{t-1} + \underset{(1743.517801)}{0.281065}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(14.936142)}{0.188540}\,\omega_{p,t} - \underset{(0.004480)}{0.005200}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(218.683448)}{0.526282} + \underset{(48.875387)}{0.840413}\,p_{t-1} + \frac{\underset{(1.120806)}{0.090234}}{2(\underset{(14.936142)}{0.188540})^2}\,u_{t-1}^2,\\
n_t &= \underset{(13336.893874)}{2.921890} + \underset{(0.000083)}{0.214037}\,n_{t-1} + \frac{\underset{(61.084690)}{0.657909}}{2(\underset{(0.004480)}{0.005200})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.109070 | 1177.103957 |
| rho_1 | 0.566139 | 777.545196 |
| rho_2 | 0.092806 | 225.635674 |
| phi_1 | 0.281065 | 1743.517801 |
| p0 | 0.526282 | 218.683448 |
| n0 | 2.921890 | 13336.893874 |
| rho_p | 0.840413 | 48.875387 |
| rho_n | 0.214037 | 0.000083 |
| phi_p | 0.090234 | 1.120806 |
| phi_n | 0.657909 | 61.084690 |
| sigma_p | 0.188540 | 14.936142 |
| sigma_n | 0.005200 | 0.004480 |

### Rank 17: Seed 27, Draw 33

- LogLik: `-150.926065`; AIC: `325.852130`; BIC: `366.299787`
- Max shape path: `164.582941`; max implied variance: `4.522415`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.657743)}{0.044211} + \underset{(14.096093)}{0.534263}\,\pi_t + \underset{(0.058219)}{0.008705}\,\pi_{t-1} + \underset{(0.511008)}{0.422497}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.887546)}{0.199773}\,\omega_{p,t} - \underset{(0.353330)}{0.071506}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(34.554966)}{1.893324} + \underset{(10.389262)}{0.428651}\,p_{t-1} + \frac{\underset{(0.571695)}{0.416522}}{2(\underset{(0.887546)}{0.199773})^2}\,u_{t-1}^2,\\
n_t &= \underset{(9.151124)}{4.471182} + \underset{(0.388173)}{0.654822}\,n_{t-1} + \frac{\underset{(0.436177)}{0.088077}}{2(\underset{(0.353330)}{0.071506})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.044211 | 0.657743 |
| rho_1 | 0.534263 | 14.096093 |
| rho_2 | 0.008705 | 0.058219 |
| phi_1 | 0.422497 | 0.511008 |
| p0 | 1.893324 | 34.554966 |
| n0 | 4.471182 | 9.151124 |
| rho_p | 0.428651 | 10.389262 |
| rho_n | 0.654822 | 0.388173 |
| phi_p | 0.416522 | 0.571695 |
| phi_n | 0.088077 | 0.436177 |
| sigma_p | 0.199773 | 0.887546 |
| sigma_n | 0.071506 | 0.353330 |

### Rank 18: Seed 4, Draw 33

- LogLik: `-152.543626`; AIC: `329.087252`; BIC: `369.534908`
- Max shape path: `134.632804`; max implied variance: `9.471505`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1.988128)}{-0.065762} + \underset{(0.452483)}{0.490099}\,\pi_t + \underset{(1.217975)}{0.288797}\,\pi_{t-1} + \underset{(1.163762)}{0.234986}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000010)}{0.106574}\,\omega_{p,t} - \underset{(0.001382)}{0.270768}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.532293)}{9.999977} + \underset{(0.608978)}{0.108186}\,p_{t-1} + \frac{\underset{(0.001422)}{0.147561}}{2(\underset{(0.000010)}{0.106574})^2}\,u_{t-1}^2,\\
n_t &= \underset{(13.645966)}{0.634078} + \underset{(0.552316)}{0.134067}\,n_{t-1} + \frac{\underset{(0.083621)}{0.829724}}{2(\underset{(0.001382)}{0.270768})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.065762 | 1.988128 |
| rho_1 | 0.490099 | 0.452483 |
| rho_2 | 0.288797 | 1.217975 |
| phi_1 | 0.234986 | 1.163762 |
| p0 | 9.999977 | 3.532293 |
| n0 | 0.634078 | 13.645966 |
| rho_p | 0.108186 | 0.608978 |
| rho_n | 0.134067 | 0.552316 |
| phi_p | 0.147561 | 0.001422 |
| phi_n | 0.829724 | 0.083621 |
| sigma_p | 0.106574 | 0.000010 |
| sigma_n | 0.270768 | 0.001382 |

### Rank 19: Seed 14, Draw 36

- LogLik: `-153.046409`; AIC: `330.092819`; BIC: `370.540475`
- Max shape path: `159.849241`; max implied variance: `7.752837`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(6.979700)}{0.110968} + \underset{(13.351217)}{0.257765}\,\pi_t + \underset{(37.751673)}{0.218227}\,\pi_{t-1} + \underset{(35.495428)}{0.440257}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.154298)}{0.104102}\,\omega_{p,t} - \underset{(0.635374)}{0.196265}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(26.051759)}{3.392496} + \underset{(2.008643)}{0.563391}\,p_{t-1} + \frac{\underset{(3.456848)}{0.154641}}{2(\underset{(0.154298)}{0.104102})^2}\,u_{t-1}^2,\\
n_t &= \underset{(12.267210)}{0.922572} + \underset{(21.065509)}{0.288735}\,n_{t-1} + \frac{\underset{(10.310256)}{0.635847}}{2(\underset{(0.635374)}{0.196265})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.110968 | 6.979700 |
| rho_1 | 0.257765 | 13.351217 |
| rho_2 | 0.218227 | 37.751673 |
| phi_1 | 0.440257 | 35.495428 |
| p0 | 3.392496 | 26.051759 |
| n0 | 0.922572 | 12.267210 |
| rho_p | 0.563391 | 2.008643 |
| rho_n | 0.288735 | 21.065509 |
| phi_p | 0.154641 | 3.456848 |
| phi_n | 0.635847 | 10.310256 |
| sigma_p | 0.104102 | 0.154298 |
| sigma_n | 0.196265 | 0.635374 |

### Rank 20: Seed 14, Draw 32

- LogLik: `-156.034070`; AIC: `336.068140`; BIC: `376.515796`
- Max shape path: `1983.525858`; max implied variance: `4.448180`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(6.495281)}{0.206511} + \underset{(7.255929)}{0.138601}\,\pi_t + \underset{(2.387225)}{0.008793}\,\pi_{t-1} + \underset{(9.504505)}{1.041140}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000202)}{0.019508}\,\omega_{p,t} - \underset{(0.252911)}{0.313079}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(184.185157)}{9.327102} + \underset{(1.438902)}{0.550701}\,p_{t-1} + \frac{\underset{(0.011250)}{0.077241}}{2(\underset{(0.000202)}{0.019508})^2}\,u_{t-1}^2,\\
n_t &= \underset{(48.936926)}{5.136997} + \underset{(3.893326)}{0.569601}\,n_{t-1} + \frac{\underset{(0.692006)}{0.260526}}{2(\underset{(0.252911)}{0.313079})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.206511 | 6.495281 |
| rho_1 | 0.138601 | 7.255929 |
| rho_2 | 0.008793 | 2.387225 |
| phi_1 | 1.041140 | 9.504505 |
| p0 | 9.327102 | 184.185157 |
| n0 | 5.136997 | 48.936926 |
| rho_p | 0.550701 | 1.438902 |
| rho_n | 0.569601 | 3.893326 |
| phi_p | 0.077241 | 0.011250 |
| phi_n | 0.260526 | 0.692006 |
| sigma_p | 0.019508 | 0.000202 |
| sigma_n | 0.313079 | 0.252911 |

## ARX(2,2)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 10 | 25 | -61.925184 | 149.850368 | 193.668662 | 262.085151 | 4.464018 | yes | `computed` |
| 2 | 26 | 19 | -66.378246 | 158.756492 | 202.574786 | 3813.897751 | 5.728508 | yes | `computed` |
| 3 | 24 | 7 | -79.902479 | 185.804957 | 229.623252 | 168.015386 | 2.812712 | yes | `computed` |
| 4 | 6 | 11 | -86.068763 | 198.137526 | 241.955820 | 280.228267 | 3.060894 | yes | `computed` |
| 5 | 17 | 33 | -86.625742 | 199.251484 | 243.069778 | 177.450296 | 6.169924 | yes | `computed` |
| 6 | 46 | 4 | -91.421677 | 208.843355 | 252.661649 | 154.152711 | 7.692550 | yes | `computed` |
| 7 | 36 | 22 | -99.201755 | 224.403511 | 268.221805 | 445.052793 | 7.553068 | yes | `computed` |
| 8 | 39 | 27 | -102.547309 | 231.094617 | 274.912912 | 31151.465436 | 4.034935 | yes | `computed` |
| 9 | 20 | 14 | -117.585779 | 261.171558 | 304.989852 | 597.245995 | 10.437536 | yes | `computed` |
| 10 | 11 | 14 | -120.045245 | 266.090490 | 309.908785 | 291.819055 | 4.796976 | yes | `computed` |
| 11 | 50 | 26 | -123.981773 | 273.963547 | 317.781841 | 698.742148 | 5.592177 | yes | `computed` |
| 12 | 25 | 19 | -130.884277 | 287.768554 | 331.586848 | 150.498831 | 15.603709 | yes | `computed` |
| 13 | 30 | 9 | -135.468167 | 296.936335 | 340.754629 | 179.662297 | 8.609947 | yes | `computed` |
| 14 | 50 | 20 | -139.418382 | 304.836765 | 348.655059 | 157.831324 | 8.068394 | yes | `computed` |
| 15 | 15 | 22 | -144.332504 | 314.665007 | 358.483302 | 1204.139790 | 4.852232 | yes | `computed` |
| 16 | 9 | 40 | -144.849358 | 315.698716 | 359.517010 | 5157.926556 | 5.824597 | yes | `computed` |
| 17 | 29 | 3 | -148.093319 | 322.186637 | 366.004932 | 140.394326 | 4.683188 | yes | `computed` |
| 18 | 24 | 12 | -153.321042 | 332.642083 | 376.460378 | 136.719025 | 8.664038 | no | `computed` |
| 19 | 18 | 25 | -154.524648 | 335.049295 | 378.867589 | 345.587163 | 7.933266 | no | `computed` |
| 20 | 31 | 28 | -158.351218 | 342.702436 | 386.520730 | 203.711424 | 6.420922 | no | `computed` |

### Rank 1: Seed 10, Draw 25

- LogLik: `-61.925184`; AIC: `149.850368`; BIC: `193.668662`
- Max shape path: `262.085151`; max implied variance: `4.464018`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(351.157967)}{0.073215} + \underset{(872.635148)}{0.369573}\,\pi_t + \underset{(88.947237)}{0.261927}\,\pi_{t-1} + \underset{(2400.667081)}{0.055788}\,SPF_t + \underset{(1690.171015)}{0.279174}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.991133)}{0.027712}\,\omega_{p,t} - \underset{(26.553305)}{0.150845}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1783.807750)}{5.013663} + \underset{(41.417833)}{0.964615}\,p_{t-1} + \frac{\underset{(41.930723)}{0.005810}}{2(\underset{(1.991133)}{0.027712})^2}\,u_{t-1}^2,\\
n_t &= \underset{(456.386425)}{2.925165} + \underset{(76.218189)}{0.249291}\,n_{t-1} + \frac{\underset{(48.690015)}{0.425049}}{2(\underset{(26.553305)}{0.150845})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.073215 | 351.157967 |
| rho_1 | 0.369573 | 872.635148 |
| rho_2 | 0.261927 | 88.947237 |
| phi_1 | 0.055788 | 2400.667081 |
| phi_2 | 0.279174 | 1690.171015 |
| p0 | 5.013663 | 1783.807750 |
| n0 | 2.925165 | 456.386425 |
| rho_p | 0.964615 | 41.417833 |
| rho_n | 0.249291 | 76.218189 |
| phi_p | 0.005810 | 41.930723 |
| phi_n | 0.425049 | 48.690015 |
| sigma_p | 0.027712 | 1.991133 |
| sigma_n | 0.150845 | 26.553305 |

### Rank 2: Seed 26, Draw 19

- LogLik: `-66.378246`; AIC: `158.756492`; BIC: `202.574786`
- Max shape path: `3813.897751`; max implied variance: `5.728508`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(7.455266)}{0.224682} + \underset{(5.100449)}{0.388085}\,\pi_t + \underset{(2.467503)}{0.142189}\,\pi_{t-1} + \underset{(12.692775)}{-0.509452}\,SPF_t + \underset{(1.881968)}{0.925216}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001624)}{0.028431}\,\omega_{p,t} - \underset{(8.199105)}{0.265060}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1371.061491)}{7.273837} + \underset{(4.628095)}{0.556710}\,p_{t-1} + \frac{\underset{(13.326087)}{0.308945}}{2(\underset{(0.001624)}{0.028431})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1470.102846)}{3.133774} + \underset{(56.862793)}{0.883901}\,n_{t-1} + \frac{\underset{(48.892659)}{0.065642}}{2(\underset{(8.199105)}{0.265060})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.224682 | 7.455266 |
| rho_1 | 0.388085 | 5.100449 |
| rho_2 | 0.142189 | 2.467503 |
| phi_1 | -0.509452 | 12.692775 |
| phi_2 | 0.925216 | 1.881968 |
| p0 | 7.273837 | 1371.061491 |
| n0 | 3.133774 | 1470.102846 |
| rho_p | 0.556710 | 4.628095 |
| rho_n | 0.883901 | 56.862793 |
| phi_p | 0.308945 | 13.326087 |
| phi_n | 0.065642 | 48.892659 |
| sigma_p | 0.028431 | 0.001624 |
| sigma_n | 0.265060 | 8.199105 |

### Rank 3: Seed 24, Draw 7

- LogLik: `-79.902479`; AIC: `185.804957`; BIC: `229.623252`
- Max shape path: `168.015386`; max implied variance: `2.812712`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(13.696260)}{0.250460} + \underset{(207.709963)}{0.390960}\,\pi_t + \underset{(21.724504)}{0.134068}\,\pi_{t-1} + \underset{(312.597959)}{0.402582}\,SPF_t + \underset{(115.825918)}{-0.139047}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(15.534110)}{0.177933}\,\omega_{p,t} - \underset{(0.080693)}{0.047487}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(178.667627)}{1.873331} + \underset{(61.284481)}{0.344976}\,p_{t-1} + \frac{\underset{(42.431561)}{0.246582}}{2(\underset{(15.534110)}{0.177933})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1374.738673)}{8.360788} + \underset{(34.383441)}{0.768672}\,n_{t-1} + \frac{\underset{(0.080973)}{0.029500}}{2(\underset{(0.080693)}{0.047487})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.250460 | 13.696260 |
| rho_1 | 0.390960 | 207.709963 |
| rho_2 | 0.134068 | 21.724504 |
| phi_1 | 0.402582 | 312.597959 |
| phi_2 | -0.139047 | 115.825918 |
| p0 | 1.873331 | 178.667627 |
| n0 | 8.360788 | 1374.738673 |
| rho_p | 0.344976 | 61.284481 |
| rho_n | 0.768672 | 34.383441 |
| phi_p | 0.246582 | 42.431561 |
| phi_n | 0.029500 | 0.080973 |
| sigma_p | 0.177933 | 15.534110 |
| sigma_n | 0.047487 | 0.080693 |

### Rank 4: Seed 6, Draw 11

- LogLik: `-86.068763`; AIC: `198.137526`; BIC: `241.955820`
- Max shape path: `280.228267`; max implied variance: `3.060894`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(330.786587)}{-0.080364} + \underset{(27.403983)}{0.229754}\,\pi_t + \underset{(0.000004)}{0.152536}\,\pi_{t-1} + \underset{(778.919686)}{0.132234}\,SPF_t + \underset{(1521.443616)}{0.486496}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000646)}{0.025256}\,\omega_{p,t} - \underset{(0.002715)}{0.121015}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1606.872873)}{6.830372} + \underset{(0.000021)}{0.763805}\,p_{t-1} + \frac{\underset{(0.000663)}{0.017249}}{2(\underset{(0.000646)}{0.025256})^2}\,u_{t-1}^2,\\
n_t &= \underset{(5542.649255)}{9.134264} + \underset{(0.000018)}{0.571020}\,n_{t-1} + \frac{\underset{(105.146911)}{0.290531}}{2(\underset{(0.002715)}{0.121015})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.080364 | 330.786587 |
| rho_1 | 0.229754 | 27.403983 |
| rho_2 | 0.152536 | 0.000004 |
| phi_1 | 0.132234 | 778.919686 |
| phi_2 | 0.486496 | 1521.443616 |
| p0 | 6.830372 | 1606.872873 |
| n0 | 9.134264 | 5542.649255 |
| rho_p | 0.763805 | 0.000021 |
| rho_n | 0.571020 | 0.000018 |
| phi_p | 0.017249 | 0.000663 |
| phi_n | 0.290531 | 105.146911 |
| sigma_p | 0.025256 | 0.000646 |
| sigma_n | 0.121015 | 0.002715 |

### Rank 5: Seed 17, Draw 33

- LogLik: `-86.625742`; AIC: `199.251484`; BIC: `243.069778`
- Max shape path: `177.450296`; max implied variance: `6.169924`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(16335.837201)}{0.442113} + \underset{(23673.312402)}{0.081556}\,\pi_t + \underset{(22029.351089)}{0.155466}\,\pi_{t-1} + \underset{(89751.677288)}{-0.000547}\,SPF_t + \underset{(69925.202082)}{0.495184}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.728504)}{0.044881}\,\omega_{p,t} - \underset{(2733.993149)}{0.450802}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(49934.412222)}{1.209099} + \underset{(0.218005)}{0.970162}\,p_{t-1} + \frac{\underset{(1.820400)}{0.018103}}{2(\underset{(1.728504)}{0.044881})^2}\,u_{t-1}^2,\\
n_t &= \underset{(8272.357267)}{0.000035} + \underset{(8666.725597)}{0.430336}\,n_{t-1} + \frac{\underset{(832.273543)}{0.569585}}{2(\underset{(2733.993149)}{0.450802})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.442113 | 16335.837201 |
| rho_1 | 0.081556 | 23673.312402 |
| rho_2 | 0.155466 | 22029.351089 |
| phi_1 | -0.000547 | 89751.677288 |
| phi_2 | 0.495184 | 69925.202082 |
| p0 | 1.209099 | 49934.412222 |
| n0 | 0.000035 | 8272.357267 |
| rho_p | 0.970162 | 0.218005 |
| rho_n | 0.430336 | 8666.725597 |
| phi_p | 0.018103 | 1.820400 |
| phi_n | 0.569585 | 832.273543 |
| sigma_p | 0.044881 | 1.728504 |
| sigma_n | 0.450802 | 2733.993149 |

### Rank 6: Seed 46, Draw 4

- LogLik: `-91.421677`; AIC: `208.843355`; BIC: `252.661649`
- Max shape path: `154.152711`; max implied variance: `7.692550`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.194730)}{0.318288} + \underset{(6.658136)}{0.079860}\,\pi_t + \underset{(1.845057)}{0.251467}\,\pi_{t-1} + \underset{(22.967472)}{0.255399}\,SPF_t + \underset{(7.082124)}{-0.029651}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.087266)}{0.077762}\,\omega_{p,t} - \underset{(0.484794)}{0.296822}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(2.128024)}{2.484072} + \underset{(14.146912)}{0.786802}\,p_{t-1} + \frac{\underset{(0.616114)}{0.083555}}{2(\underset{(0.087266)}{0.077762})^2}\,u_{t-1}^2,\\
n_t &= \underset{(52.500003)}{0.001846} + \underset{(11.703044)}{0.249348}\,n_{t-1} + \frac{\underset{(3.555594)}{0.697336}}{2(\underset{(0.484794)}{0.296822})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.318288 | 0.194730 |
| rho_1 | 0.079860 | 6.658136 |
| rho_2 | 0.251467 | 1.845057 |
| phi_1 | 0.255399 | 22.967472 |
| phi_2 | -0.029651 | 7.082124 |
| p0 | 2.484072 | 2.128024 |
| n0 | 0.001846 | 52.500003 |
| rho_p | 0.786802 | 14.146912 |
| rho_n | 0.249348 | 11.703044 |
| phi_p | 0.083555 | 0.616114 |
| phi_n | 0.697336 | 3.555594 |
| sigma_p | 0.077762 | 0.087266 |
| sigma_n | 0.296822 | 0.484794 |

### Rank 7: Seed 36, Draw 22

- LogLik: `-99.201755`; AIC: `224.403511`; BIC: `268.221805`
- Max shape path: `445.052793`; max implied variance: `7.553068`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(23.555919)}{0.127860} + \underset{(17.999377)}{0.338984}\,\pi_t + \underset{(15.779240)}{0.187070}\,\pi_{t-1} + \underset{(139.024260)}{0.040345}\,SPF_t + \underset{(112.157580)}{0.348515}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.002195)}{0.039692}\,\omega_{p,t} - \underset{(6.918621)}{0.192851}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(798.347630)}{7.650206} + \underset{(5.585462)}{0.843466}\,p_{t-1} + \frac{\underset{(0.000034)}{0.058963}}{2(\underset{(0.002195)}{0.039692})^2}\,u_{t-1}^2,\\
n_t &= \underset{(185.931920)}{1.032090} + \underset{(63.721531)}{0.244655}\,n_{t-1} + \frac{\underset{(78.442674)}{0.715666}}{2(\underset{(6.918621)}{0.192851})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.127860 | 23.555919 |
| rho_1 | 0.338984 | 17.999377 |
| rho_2 | 0.187070 | 15.779240 |
| phi_1 | 0.040345 | 139.024260 |
| phi_2 | 0.348515 | 112.157580 |
| p0 | 7.650206 | 798.347630 |
| n0 | 1.032090 | 185.931920 |
| rho_p | 0.843466 | 5.585462 |
| rho_n | 0.244655 | 63.721531 |
| phi_p | 0.058963 | 0.000034 |
| phi_n | 0.715666 | 78.442674 |
| sigma_p | 0.039692 | 0.002195 |
| sigma_n | 0.192851 | 6.918621 |

### Rank 8: Seed 39, Draw 27

- LogLik: `-102.547309`; AIC: `231.094617`; BIC: `274.912912`
- Max shape path: `31151.465436`; max implied variance: `4.034935`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(169.300261)}{0.204762} + \underset{(178.648898)}{0.475180}\,\pi_t + \underset{(64.560326)}{-0.011542}\,\pi_{t-1} + \underset{(1214.518792)}{0.145621}\,SPF_t + \underset{(1179.041758)}{0.185751}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(4.152515)}{0.067951}\,\omega_{p,t} - \underset{(0.006186)}{0.009203}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(536.578602)}{7.461907} + \underset{(23.239457)}{0.743011}\,p_{t-1} + \frac{\underset{(22.588462)}{0.138543}}{2(\underset{(4.152515)}{0.067951})^2}\,u_{t-1}^2,\\
n_t &= \underset{(3818.571296)}{0.905546} + \underset{(13.900836)}{0.419818}\,n_{t-1} + \frac{\underset{(15.948026)}{0.306278}}{2(\underset{(0.006186)}{0.009203})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.204762 | 169.300261 |
| rho_1 | 0.475180 | 178.648898 |
| rho_2 | -0.011542 | 64.560326 |
| phi_1 | 0.145621 | 1214.518792 |
| phi_2 | 0.185751 | 1179.041758 |
| p0 | 7.461907 | 536.578602 |
| n0 | 0.905546 | 3818.571296 |
| rho_p | 0.743011 | 23.239457 |
| rho_n | 0.419818 | 13.900836 |
| phi_p | 0.138543 | 22.588462 |
| phi_n | 0.306278 | 15.948026 |
| sigma_p | 0.067951 | 4.152515 |
| sigma_n | 0.009203 | 0.006186 |

### Rank 9: Seed 20, Draw 14

- LogLik: `-117.585779`; AIC: `261.171558`; BIC: `304.989852`
- Max shape path: `597.245995`; max implied variance: `10.437536`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(43.264480)}{0.039914} + \underset{(145.175396)}{0.445104}\,\pi_t + \underset{(16.506799)}{0.245302}\,\pi_{t-1} + \underset{(50.359533)}{-0.125617}\,SPF_t + \underset{(210.615097)}{0.455937}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000094)}{0.044820}\,\omega_{p,t} - \underset{(2.173480)}{0.227379}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(960.022184)}{0.258319} + \underset{(49.400636)}{0.801917}\,p_{t-1} + \frac{\underset{(3.175724)}{0.113367}}{2(\underset{(0.000094)}{0.044820})^2}\,u_{t-1}^2,\\
n_t &= \underset{(839.637149)}{1.133004} + \underset{(146.124275)}{0.003624}\,n_{t-1} + \frac{\underset{(72.044436)}{0.947671}}{2(\underset{(2.173480)}{0.227379})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039914 | 43.264480 |
| rho_1 | 0.445104 | 145.175396 |
| rho_2 | 0.245302 | 16.506799 |
| phi_1 | -0.125617 | 50.359533 |
| phi_2 | 0.455937 | 210.615097 |
| p0 | 0.258319 | 960.022184 |
| n0 | 1.133004 | 839.637149 |
| rho_p | 0.801917 | 49.400636 |
| rho_n | 0.003624 | 146.124275 |
| phi_p | 0.113367 | 3.175724 |
| phi_n | 0.947671 | 72.044436 |
| sigma_p | 0.044820 | 0.000094 |
| sigma_n | 0.227379 | 2.173480 |

### Rank 10: Seed 11, Draw 14

- LogLik: `-120.045245`; AIC: `266.090490`; BIC: `309.908785`
- Max shape path: `291.819055`; max implied variance: `4.796976`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(57.245960)}{0.131703} + \underset{(79.261494)}{0.340233}\,\pi_t + \underset{(206.336192)}{0.177592}\,\pi_{t-1} + \underset{(1158.552385)}{0.381721}\,SPF_t + \underset{(780.162438)}{0.071338}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000880)}{0.035746}\,\omega_{p,t} - \underset{(0.006462)}{0.123599}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(314.844365)}{4.757803} + \underset{(0.000000)}{0.924630}\,p_{t-1} + \frac{\underset{(0.004390)}{0.021060}}{2(\underset{(0.000880)}{0.035746})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1308.803545)}{2.685688} + \underset{(39.446685)}{0.428405}\,n_{t-1} + \frac{\underset{(17.352789)}{0.451560}}{2(\underset{(0.006462)}{0.123599})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.131703 | 57.245960 |
| rho_1 | 0.340233 | 79.261494 |
| rho_2 | 0.177592 | 206.336192 |
| phi_1 | 0.381721 | 1158.552385 |
| phi_2 | 0.071338 | 780.162438 |
| p0 | 4.757803 | 314.844365 |
| n0 | 2.685688 | 1308.803545 |
| rho_p | 0.924630 | 0.000000 |
| rho_n | 0.428405 | 39.446685 |
| phi_p | 0.021060 | 0.004390 |
| phi_n | 0.451560 | 17.352789 |
| sigma_p | 0.035746 | 0.000880 |
| sigma_n | 0.123599 | 0.006462 |

### Rank 11: Seed 50, Draw 26

- LogLik: `-123.981773`; AIC: `273.963547`; BIC: `317.781841`
- Max shape path: `698.742148`; max implied variance: `5.592177`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(231.028941)}{-0.041214} + \underset{(88.140641)}{0.521027}\,\pi_t + \underset{(40.954572)}{0.296423}\,\pi_{t-1} + \underset{(285.463188)}{-0.470487}\,SPF_t + \underset{(663.737392)}{0.753360}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001569)}{0.035860}\,\omega_{p,t} - \underset{(11.555425)}{0.164650}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(200.116103)}{1.592441} + \underset{(369.335313)}{0.124633}\,p_{t-1} + \frac{\underset{(6.993490)}{0.090771}}{2(\underset{(0.001569)}{0.035860})^2}\,u_{t-1}^2,\\
n_t &= \underset{(356.703393)}{9.774679} + \underset{(100.437422)}{0.000000}\,n_{t-1} + \frac{\underset{(31.287925)}{0.450240}}{2(\underset{(11.555425)}{0.164650})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.041214 | 231.028941 |
| rho_1 | 0.521027 | 88.140641 |
| rho_2 | 0.296423 | 40.954572 |
| phi_1 | -0.470487 | 285.463188 |
| phi_2 | 0.753360 | 663.737392 |
| p0 | 1.592441 | 200.116103 |
| n0 | 9.774679 | 356.703393 |
| rho_p | 0.124633 | 369.335313 |
| rho_n | 0.000000 | 100.437422 |
| phi_p | 0.090771 | 6.993490 |
| phi_n | 0.450240 | 31.287925 |
| sigma_p | 0.035860 | 0.001569 |
| sigma_n | 0.164650 | 11.555425 |

### Rank 12: Seed 25, Draw 19

- LogLik: `-130.884277`; AIC: `287.768554`; BIC: `331.586848`
- Max shape path: `150.498831`; max implied variance: `15.603709`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.031631)}{0.245820} + \underset{(501.809952)}{0.151641}\,\pi_t + \underset{(0.031600)}{0.383664}\,\pi_{t-1} + \underset{(1550.586506)}{-1.061084}\,SPF_t + \underset{(1254.575513)}{1.332050}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.031537)}{0.094191}\,\omega_{p,t} - \underset{(0.031597)}{0.411092}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(13.575697)}{1.590027} + \underset{(0.000493)}{0.605048}\,p_{t-1} + \frac{\underset{(0.031525)}{0.118982}}{2(\underset{(0.031537)}{0.094191})^2}\,u_{t-1}^2,\\
n_t &= \underset{(53.501320)}{0.021202} + \underset{(207.454569)}{0.001724}\,n_{t-1} + \frac{\underset{(205.783393)}{0.998046}}{2(\underset{(0.031597)}{0.411092})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.245820 | 0.031631 |
| rho_1 | 0.151641 | 501.809952 |
| rho_2 | 0.383664 | 0.031600 |
| phi_1 | -1.061084 | 1550.586506 |
| phi_2 | 1.332050 | 1254.575513 |
| p0 | 1.590027 | 13.575697 |
| n0 | 0.021202 | 53.501320 |
| rho_p | 0.605048 | 0.000493 |
| rho_n | 0.001724 | 207.454569 |
| phi_p | 0.118982 | 0.031525 |
| phi_n | 0.998046 | 205.783393 |
| sigma_p | 0.094191 | 0.031537 |
| sigma_n | 0.411092 | 0.031597 |

### Rank 13: Seed 30, Draw 9

- LogLik: `-135.468167`; AIC: `296.936335`; BIC: `340.754629`
- Max shape path: `179.662297`; max implied variance: `8.609947`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(332.577416)}{0.105934} + \underset{(128.281172)}{0.336151}\,\pi_t + \underset{(33.572941)}{0.191921}\,\pi_{t-1} + \underset{(1044.062738)}{0.004357}\,SPF_t + \underset{(436.737662)}{0.394522}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.027687)}{0.095693}\,\omega_{p,t} - \underset{(0.027674)}{0.198162}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(155.126488)}{7.408858} + \underset{(9.034663)}{0.372786}\,p_{t-1} + \frac{\underset{(0.000283)}{0.161495}}{2(\underset{(0.027687)}{0.095693})^2}\,u_{t-1}^2,\\
n_t &= \underset{(118.307624)}{1.227179} + \underset{(59.975768)}{0.199524}\,n_{t-1} + \frac{\underset{(4.941137)}{0.730829}}{2(\underset{(0.027674)}{0.198162})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.105934 | 332.577416 |
| rho_1 | 0.336151 | 128.281172 |
| rho_2 | 0.191921 | 33.572941 |
| phi_1 | 0.004357 | 1044.062738 |
| phi_2 | 0.394522 | 436.737662 |
| p0 | 7.408858 | 155.126488 |
| n0 | 1.227179 | 118.307624 |
| rho_p | 0.372786 | 9.034663 |
| rho_n | 0.199524 | 59.975768 |
| phi_p | 0.161495 | 0.000283 |
| phi_n | 0.730829 | 4.941137 |
| sigma_p | 0.095693 | 0.027687 |
| sigma_n | 0.198162 | 0.027674 |

### Rank 14: Seed 50, Draw 20

- LogLik: `-139.418382`; AIC: `304.836765`; BIC: `348.655059`
- Max shape path: `157.831324`; max implied variance: `8.068394`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.447401)}{0.210607} + \underset{(0.442507)}{-0.008834}\,\pi_t + \underset{(0.261389)}{0.292917}\,\pi_{t-1} + \underset{(1.073730)}{0.050941}\,SPF_t + \underset{(0.556591)}{0.059934}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.031680)}{0.079353}\,\omega_{p,t} - \underset{(0.108901)}{0.309882}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(5.035745)}{3.883159} + \underset{(0.279490)}{0.712439}\,p_{t-1} + \frac{\underset{(0.086669)}{0.094712}}{2(\underset{(0.031680)}{0.079353})^2}\,u_{t-1}^2,\\
n_t &= \underset{(1.233324)}{0.165107} + \underset{(0.625409)}{0.208191}\,n_{t-1} + \frac{\underset{(0.635717)}{0.782384}}{2(\underset{(0.108901)}{0.309882})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.210607 | 0.447401 |
| rho_1 | -0.008834 | 0.442507 |
| rho_2 | 0.292917 | 0.261389 |
| phi_1 | 0.050941 | 1.073730 |
| phi_2 | 0.059934 | 0.556591 |
| p0 | 3.883159 | 5.035745 |
| n0 | 0.165107 | 1.233324 |
| rho_p | 0.712439 | 0.279490 |
| rho_n | 0.208191 | 0.625409 |
| phi_p | 0.094712 | 0.086669 |
| phi_n | 0.782384 | 0.635717 |
| sigma_p | 0.079353 | 0.031680 |
| sigma_n | 0.309882 | 0.108901 |

### Rank 15: Seed 15, Draw 22

- LogLik: `-144.332504`; AIC: `314.665007`; BIC: `358.483302`
- Max shape path: `1204.139790`; max implied variance: `4.852232`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(59.413674)}{-0.138620} + \underset{(223.896995)}{0.123577}\,\pi_t + \underset{(314.417891)}{0.032240}\,\pi_{t-1} + \underset{(1838.977193)}{0.120776}\,SPF_t + \underset{(1018.157700)}{-0.886455}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.016070)}{0.027918}\,\omega_{p,t} - \underset{(2.344235)}{0.341736}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(8515.909354)}{8.181204} + \underset{(53.333404)}{0.786466}\,p_{t-1} + \frac{\underset{(0.013817)}{0.021692}}{2(\underset{(0.016070)}{0.027918})^2}\,u_{t-1}^2,\\
n_t &= \underset{(674.212667)}{4.707460} + \underset{(27.826222)}{0.817434}\,n_{t-1} + \frac{\underset{(5.022690)}{0.019538}}{2(\underset{(2.344235)}{0.341736})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.138620 | 59.413674 |
| rho_1 | 0.123577 | 223.896995 |
| rho_2 | 0.032240 | 314.417891 |
| phi_1 | 0.120776 | 1838.977193 |
| phi_2 | -0.886455 | 1018.157700 |
| p0 | 8.181204 | 8515.909354 |
| n0 | 4.707460 | 674.212667 |
| rho_p | 0.786466 | 53.333404 |
| rho_n | 0.817434 | 27.826222 |
| phi_p | 0.021692 | 0.013817 |
| phi_n | 0.019538 | 5.022690 |
| sigma_p | 0.027918 | 0.016070 |
| sigma_n | 0.341736 | 2.344235 |

### Rank 16: Seed 9, Draw 40

- LogLik: `-144.849358`; AIC: `315.698716`; BIC: `359.517010`
- Max shape path: `5157.926556`; max implied variance: `5.824597`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(716.229047)}{-0.374865} + \underset{(78.563931)}{0.005636}\,\pi_t + \underset{(123.312152)}{0.066333}\,\pi_{t-1} + \underset{(5134.960466)}{1.399950}\,SPF_t + \underset{(3857.138812)}{-0.115860}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000004)}{0.028727}\,\omega_{p,t} - \underset{(16.185136)}{0.077353}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(950.111850)}{5.407234} + \underset{(133.069726)}{0.386734}\,p_{t-1} + \frac{\underset{(8.990208)}{0.536091}}{2(\underset{(0.000004)}{0.028727})^2}\,u_{t-1}^2,\\
n_t &= \underset{(4055.000685)}{5.926176} + \underset{(116.892172)}{0.768225}\,n_{t-1} + \frac{\underset{(328.691584)}{0.160261}}{2(\underset{(16.185136)}{0.077353})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.374865 | 716.229047 |
| rho_1 | 0.005636 | 78.563931 |
| rho_2 | 0.066333 | 123.312152 |
| phi_1 | 1.399950 | 5134.960466 |
| phi_2 | -0.115860 | 3857.138812 |
| p0 | 5.407234 | 950.111850 |
| n0 | 5.926176 | 4055.000685 |
| rho_p | 0.386734 | 133.069726 |
| rho_n | 0.768225 | 116.892172 |
| phi_p | 0.536091 | 8.990208 |
| phi_n | 0.160261 | 328.691584 |
| sigma_p | 0.028727 | 0.000004 |
| sigma_n | 0.077353 | 16.185136 |

### Rank 17: Seed 29, Draw 3

- LogLik: `-148.093319`; AIC: `322.186637`; BIC: `366.004932`
- Max shape path: `140.394326`; max implied variance: `4.683188`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.769000)}{-0.103753} + \underset{(0.149733)}{0.520883}\,\pi_t + \underset{(0.473959)}{0.035450}\,\pi_{t-1} + \underset{(3.690307)}{-0.757089}\,SPF_t + \underset{(3.340415)}{1.131422}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001015)}{0.200278}\,\omega_{p,t} - \underset{(0.000006)}{0.079658}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(2.103789)}{1.681601} + \underset{(0.086751)}{0.229625}\,p_{t-1} + \frac{\underset{(0.011337)}{0.475139}}{2(\underset{(0.001015)}{0.200278})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.525086)}{2.360660} + \underset{(0.042930)}{0.741088}\,n_{t-1} + \frac{\underset{(0.001012)}{0.098044}}{2(\underset{(0.000006)}{0.079658})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.103753 | 0.769000 |
| rho_1 | 0.520883 | 0.149733 |
| rho_2 | 0.035450 | 0.473959 |
| phi_1 | -0.757089 | 3.690307 |
| phi_2 | 1.131422 | 3.340415 |
| p0 | 1.681601 | 2.103789 |
| n0 | 2.360660 | 0.525086 |
| rho_p | 0.229625 | 0.086751 |
| rho_n | 0.741088 | 0.042930 |
| phi_p | 0.475139 | 0.011337 |
| phi_n | 0.098044 | 0.001012 |
| sigma_p | 0.200278 | 0.001015 |
| sigma_n | 0.079658 | 0.000006 |

### Rank 18: Seed 24, Draw 12

- LogLik: `-153.321042`; AIC: `332.642083`; BIC: `376.460378`
- Max shape path: `136.719025`; max implied variance: `8.664038`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(18.943079)}{0.071234} + \underset{(31.027527)}{0.285902}\,\pi_t + \underset{(1.033113)}{0.126009}\,\pi_{t-1} + \underset{(141.115660)}{0.419607}\,SPF_t + \underset{(267.677147)}{0.133506}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001106)}{0.090949}\,\omega_{p,t} - \underset{(0.000000)}{0.347797}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1031.819722)}{4.391383} + \underset{(19.920482)}{0.705807}\,p_{t-1} + \frac{\underset{(0.001081)}{0.106508}}{2(\underset{(0.001106)}{0.090949})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.099835)}{0.000656} + \underset{(5.597524)}{0.164114}\,n_{t-1} + \frac{\underset{(5.615415)}{0.835749}}{2(\underset{(0.000000)}{0.347797})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.071234 | 18.943079 |
| rho_1 | 0.285902 | 31.027527 |
| rho_2 | 0.126009 | 1.033113 |
| phi_1 | 0.419607 | 141.115660 |
| phi_2 | 0.133506 | 267.677147 |
| p0 | 4.391383 | 1031.819722 |
| n0 | 0.000656 | 0.099835 |
| rho_p | 0.705807 | 19.920482 |
| rho_n | 0.164114 | 5.597524 |
| phi_p | 0.106508 | 0.001081 |
| phi_n | 0.835749 | 5.615415 |
| sigma_p | 0.090949 | 0.001106 |
| sigma_n | 0.347797 | 0.000000 |

### Rank 19: Seed 18, Draw 25

- LogLik: `-154.524648`; AIC: `335.049295`; BIC: `378.867589`
- Max shape path: `345.587163`; max implied variance: `7.933266`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(200.379525)}{0.111032} + \underset{(11.942187)}{0.319311}\,\pi_t + \underset{(249.511980)}{0.190955}\,\pi_{t-1} + \underset{(59.009462)}{0.159788}\,SPF_t + \underset{(19.109283)}{0.262557}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000243)}{0.051685}\,\omega_{p,t} - \underset{(0.402999)}{0.200874}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(314.059342)}{9.950038} + \underset{(8.185620)}{0.774763}\,p_{t-1} + \frac{\underset{(0.396930)}{0.079705}}{2(\underset{(0.000243)}{0.051685})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.355192)}{0.000102} + \underset{(0.626033)}{0.262009}\,n_{t-1} + \frac{\underset{(0.588901)}{0.737972}}{2(\underset{(0.402999)}{0.200874})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.111032 | 200.379525 |
| rho_1 | 0.319311 | 11.942187 |
| rho_2 | 0.190955 | 249.511980 |
| phi_1 | 0.159788 | 59.009462 |
| phi_2 | 0.262557 | 19.109283 |
| p0 | 9.950038 | 314.059342 |
| n0 | 0.000102 | 0.355192 |
| rho_p | 0.774763 | 8.185620 |
| rho_n | 0.262009 | 0.626033 |
| phi_p | 0.079705 | 0.396930 |
| phi_n | 0.737972 | 0.588901 |
| sigma_p | 0.051685 | 0.000243 |
| sigma_n | 0.200874 | 0.402999 |

### Rank 20: Seed 31, Draw 28

- LogLik: `-158.351218`; AIC: `342.702436`; BIC: `386.520730`
- Max shape path: `203.711424`; max implied variance: `6.420922`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.095387)}{0.081776} + \underset{(0.074719)}{0.308494}\,\pi_t + \underset{(0.116482)}{0.183020}\,\pi_{t-1} + \underset{(0.144020)}{0.351207}\,SPF_t + \underset{(0.001201)}{0.113942}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001179)}{0.025574}\,\omega_{p,t} - \underset{(0.013587)}{0.176103}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000247)}{0.926010} + \underset{(0.000013)}{0.994140}\,p_{t-1} + \frac{\underset{(0.000012)}{0.000000}}{2(\underset{(0.001179)}{0.025574})^2}\,u_{t-1}^2,\\
n_t &= \underset{(0.000014)}{0.000002} + \underset{(0.000017)}{0.329279}\,n_{t-1} + \frac{\underset{(0.000017)}{0.670720}}{2(\underset{(0.013587)}{0.176103})^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.081776 | 0.095387 |
| rho_1 | 0.308494 | 0.074719 |
| rho_2 | 0.183020 | 0.116482 |
| phi_1 | 0.351207 | 0.144020 |
| phi_2 | 0.113942 | 0.001201 |
| p0 | 0.926010 | 0.000247 |
| n0 | 0.000002 | 0.000014 |
| rho_p | 0.994140 | 0.000013 |
| rho_n | 0.329279 | 0.000017 |
| phi_p | 0.000000 | 0.000012 |
| phi_n | 0.670720 | 0.000017 |
| sigma_p | 0.025574 | 0.001179 |
| sigma_n | 0.176103 | 0.013587 |
