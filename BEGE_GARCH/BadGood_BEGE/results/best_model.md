```{raw:typst}
#set page(margin: auto)
```

# BadGood BEGE Best Model Summary

Generated: `2026-06-03T14:03:36`
Total estimations: `8000`
Converged estimations: `7841`
Eligible estimations for best-model selection: `7841`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `5` admissible estimates by corrected log likelihood.

```{note}
Flagged 107 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## constant

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 5 | 34 | -76.863954 | 169.727908 | 196.693012 | 867.260443 | 3.169626 | yes |
| 2 | 8 | 1 | -95.900687 | 207.801374 | 234.766478 | 1520.179511 | 5.579027 | yes |
| 3 | 47 | 13 | -97.335535 | 210.671070 | 237.636174 | 1083.813607 | 6.125048 | yes |
| 4 | 19 | 2 | -108.585176 | 233.170353 | 260.135457 | 948.187840 | 2.327896 | yes |
| 5 | 46 | 3 | -109.985324 | 235.970647 | 262.935752 | 61548.919879 | 3662.321713 | yes |

### Rank 1: Seed 5, Draw 34

- LogLik: `-76.863954`; AIC: `169.727908`; BIC: `196.693012`
- Max shape path: `867.260443`; max implied variance: `3.169626`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.031380\,\omega_{p,t} - 0.058727\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 2.270537 + 0.983074\,p_{t-1} + \frac{0.002637}{2(0.031380)^2}\,u_{t-1}^2,\\
n_t &= 2.637307 + 0.639193\,n_{t-1} + \frac{0.348569}{2(0.058727)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 2.270537 |
| n0 | 2.637307 |
| rho_p | 0.983074 |
| rho_n | 0.639193 |
| phi_p | 0.002637 |
| phi_n | 0.348569 |
| sigma_p | 0.031380 |
| sigma_n | 0.058727 |

### Rank 2: Seed 8, Draw 1

- LogLik: `-95.900687`; AIC: `207.801374`; BIC: `234.766478`
- Max shape path: `1520.179511`; max implied variance: `5.579027`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.022691\,\omega_{p,t} - 0.189781\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 7.851649 + 0.199140\,p_{t-1} + \frac{0.096206}{2(0.022691)^2}\,u_{t-1}^2,\\
n_t &= 9.784475 + 0.269556\,n_{t-1} + \frac{0.531636}{2(0.189781)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 7.851649 |
| n0 | 9.784475 |
| rho_p | 0.199140 |
| rho_n | 0.269556 |
| phi_p | 0.096206 |
| phi_n | 0.531636 |
| sigma_p | 0.022691 |
| sigma_n | 0.189781 |

### Rank 3: Seed 47, Draw 13

- LogLik: `-97.335535`; AIC: `210.671070`; BIC: `237.636174`
- Max shape path: `1083.813607`; max implied variance: `6.125048`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.032827\,\omega_{p,t} - 0.094885\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.919752 + 0.825077\,p_{t-1} + \frac{0.120216}{2(0.032827)^2}\,u_{t-1}^2,\\
n_t &= 3.864534 + 0.410286\,n_{t-1} + \frac{0.583847}{2(0.094885)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 9.919752 |
| n0 | 3.864534 |
| rho_p | 0.825077 |
| rho_n | 0.410286 |
| phi_p | 0.120216 |
| phi_n | 0.583847 |
| sigma_p | 0.032827 |
| sigma_n | 0.094885 |

### Rank 4: Seed 19, Draw 2

- LogLik: `-108.585176`; AIC: `233.170353`; BIC: `260.135457`
- Max shape path: `948.187840`; max implied variance: `2.327896`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.028391\,\omega_{p,t} - 0.105153\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 7.379104 + 0.821244\,p_{t-1} + \frac{0.079647}{2(0.028391)^2}\,u_{t-1}^2,\\
n_t &= 6.148920 + 0.739039\,n_{t-1} + \frac{0.148665}{2(0.105153)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 7.379104 |
| n0 | 6.148920 |
| rho_p | 0.821244 |
| rho_n | 0.739039 |
| phi_p | 0.079647 |
| phi_n | 0.148665 |
| sigma_p | 0.028391 |
| sigma_n | 0.105153 |

### Rank 5: Seed 46, Draw 3

- LogLik: `-109.985324`; AIC: `235.970647`; BIC: `262.935752`
- Max shape path: `61548.919879`; max implied variance: `3662.321713`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.074653\,\omega_{p,t} - 0.243931\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.220103 + 0.691252\,p_{t-1} + \frac{0.109814}{2(0.074653)^2}\,u_{t-1}^2,\\
n_t &= 0.693730 + 0.249436\,n_{t-1} + \frac{0.750553}{2(0.243931)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 0.220103 |
| n0 | 0.693730 |
| rho_p | 0.691252 |
| rho_n | 0.249436 |
| phi_p | 0.109814 |
| phi_n | 0.750553 |
| sigma_p | 0.074653 |
| sigma_n | 0.243931 |

## ARX(1,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 16 | 24 | 134.033609 | -246.067218 | -208.990199 | 3024.746008 | 11.591510 | yes |
| 2 | 25 | 21 | -45.494390 | 112.988780 | 150.065798 | 5826.439656 | 3.575903 | yes |
| 3 | 27 | 22 | -54.652832 | 131.305664 | 168.382682 | 726.516171 | 16.268748 | yes |
| 4 | 38 | 5 | -75.025748 | 172.051495 | 209.128514 | 169.896634 | 6.238326 | yes |
| 5 | 6 | 34 | -80.401075 | 182.802151 | 219.879169 | 8005.029925 | 3.056479 | yes |

### Rank 1: Seed 16, Draw 24

- LogLik: `134.033609`; AIC: `-246.067218`; BIC: `-208.990199`
- Max shape path: `3024.746008`; max implied variance: `11.591510`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.378918 + 0.445463\,\pi_t + 1.135712\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.271844\,\omega_{p,t} - 0.048968\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 4.246196 + 0.921285\,p_{t-1} + \frac{0.022016}{2(0.271844)^2}\,u_{t-1}^2,\\
n_t &= 9.681746 + 0.232777\,n_{t-1} + \frac{0.626957}{2(0.048968)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.378918 |
| rho_1 | 0.445463 |
| phi_1 | 1.135712 |
| p0 | 4.246196 |
| n0 | 9.681746 |
| rho_p | 0.921285 |
| rho_n | 0.232777 |
| phi_p | 0.022016 |
| phi_n | 0.626957 |
| sigma_p | 0.271844 |
| sigma_n | 0.048968 |

### Rank 2: Seed 25, Draw 21

- LogLik: `-45.494390`; AIC: `112.988780`; BIC: `150.065798`
- Max shape path: `5826.439656`; max implied variance: `3.575903`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.077521 + 0.382617\,\pi_t + 0.677602\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.016682\,\omega_{p,t} - 0.097515\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.421150 + 0.665092\,p_{t-1} + \frac{0.178494}{2(0.016682)^2}\,u_{t-1}^2,\\
n_t &= 6.048270 + 0.733906\,n_{t-1} + \frac{0.188749}{2(0.097515)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.077521 |
| rho_1 | 0.382617 |
| phi_1 | 0.677602 |
| p0 | 9.421150 |
| n0 | 6.048270 |
| rho_p | 0.665092 |
| rho_n | 0.733906 |
| phi_p | 0.178494 |
| phi_n | 0.188749 |
| sigma_p | 0.016682 |
| sigma_n | 0.097515 |

### Rank 3: Seed 27, Draw 22

- LogLik: `-54.652832`; AIC: `131.305664`; BIC: `168.382682`
- Max shape path: `726.516171`; max implied variance: `16.268748`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.086861 + 0.359913\,\pi_t + 0.609843\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.040600\,\omega_{p,t} - 0.148771\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.999913 + 0.852835\,p_{t-1} + \frac{0.059870}{2(0.040600)^2}\,u_{t-1}^2,\\
n_t &= 0.003265 + 0.250377\,n_{t-1} + \frac{0.749619}{2(0.148771)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.086861 |
| rho_1 | 0.359913 |
| phi_1 | 0.609843 |
| p0 | 9.999913 |
| n0 | 0.003265 |
| rho_p | 0.852835 |
| rho_n | 0.250377 |
| phi_p | 0.059870 |
| phi_n | 0.749619 |
| sigma_p | 0.040600 |
| sigma_n | 0.148771 |

### Rank 4: Seed 38, Draw 5

- LogLik: `-75.025748`; AIC: `172.051495`; BIC: `209.128514`
- Max shape path: `169.896634`; max implied variance: `6.238326`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.101891 + 0.367629\,\pi_t + 0.609300\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.215479\,\omega_{p,t} - 0.076014\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 2.382816 + 0.114423\,p_{t-1} + \frac{0.602979}{2(0.215479)^2}\,u_{t-1}^2,\\
n_t &= 4.925862 + 0.648157\,n_{t-1} + \frac{0.101198}{2(0.076014)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.101891 |
| rho_1 | 0.367629 |
| phi_1 | 0.609300 |
| p0 | 2.382816 |
| n0 | 4.925862 |
| rho_p | 0.114423 |
| rho_n | 0.648157 |
| phi_p | 0.602979 |
| phi_n | 0.101198 |
| sigma_p | 0.215479 |
| sigma_n | 0.076014 |

### Rank 5: Seed 6, Draw 34

- LogLik: `-80.401075`; AIC: `182.802151`; BIC: `219.879169`
- Max shape path: `8005.029925`; max implied variance: `3.056479`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.346872 + 0.499570\,\pi_t + 0.093800\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.210631\,\omega_{p,t} - 0.014850\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.822369 + 0.768891\,p_{t-1} + \frac{0.059272}{2(0.210631)^2}\,u_{t-1}^2,\\
n_t &= 8.400677 + 0.735106\,n_{t-1} + \frac{0.189896}{2(0.014850)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.346872 |
| rho_1 | 0.499570 |
| phi_1 | 0.093800 |
| p0 | 3.822369 |
| n0 | 8.400677 |
| rho_p | 0.768891 |
| rho_n | 0.735106 |
| phi_p | 0.059272 |
| phi_n | 0.189896 |
| sigma_p | 0.210631 |
| sigma_n | 0.014850 |

## ARX(2,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 34 | 2 | 65.425865 | -106.851731 | -66.404074 | 2681.319149 | 2.901267 | yes |
| 2 | 21 | 4 | -38.978702 | 101.957403 | 142.405060 | 3879.803639 | 7.343046 | yes |
| 3 | 34 | 11 | -66.762579 | 157.525158 | 197.972814 | 7029.672813 | 11.252888 | yes |
| 4 | 2 | 32 | -72.190224 | 168.380448 | 208.828105 | 178.018364 | 2.558309 | yes |
| 5 | 8 | 1 | -92.524490 | 209.048981 | 249.496637 | 1885.253562 | 6.440858 | yes |

### Rank 1: Seed 34, Draw 2

- LogLik: `65.425865`; AIC: `-106.851731`; BIC: `-66.404074`
- Max shape path: `2681.319149`; max implied variance: `2.901267`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.640609 + 0.568106\,\pi_t + 0.065923\,\pi_{t-1} + 1.628255\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.024174\,\omega_{p,t} - 0.097351\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 4.300604 + 0.863914\,p_{t-1} + \frac{0.083489}{2(0.024174)^2}\,u_{t-1}^2,\\
n_t &= 8.633965 + 0.871306\,n_{t-1} + \frac{0.036444}{2(0.097351)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.640609 |
| rho_1 | 0.568106 |
| rho_2 | 0.065923 |
| phi_1 | 1.628255 |
| p0 | 4.300604 |
| n0 | 8.633965 |
| rho_p | 0.863914 |
| rho_n | 0.871306 |
| phi_p | 0.083489 |
| phi_n | 0.036444 |
| sigma_p | 0.024174 |
| sigma_n | 0.097351 |

### Rank 2: Seed 21, Draw 4

- LogLik: `-38.978702`; AIC: `101.957403`; BIC: `142.405060`
- Max shape path: `3879.803639`; max implied variance: `7.343046`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.295516 + 0.188505\,\pi_t + 0.015717\,\pi_{t-1} + -0.008748\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.030375\,\omega_{p,t} - 0.375567\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.446004 + 0.467251\,p_{t-1} + \frac{0.328018}{2(0.030375)^2}\,u_{t-1}^2,\\
n_t &= 3.489810 + 0.854431\,n_{t-1} + \frac{0.015730}{2(0.375567)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.295516 |
| rho_1 | 0.188505 |
| rho_2 | 0.015717 |
| phi_1 | -0.008748 |
| p0 | 3.446004 |
| n0 | 3.489810 |
| rho_p | 0.467251 |
| rho_n | 0.854431 |
| phi_p | 0.328018 |
| phi_n | 0.015730 |
| sigma_p | 0.030375 |
| sigma_n | 0.375567 |

### Rank 3: Seed 34, Draw 11

- LogLik: `-66.762579`; AIC: `157.525158`; BIC: `197.972814`
- Max shape path: `7029.672813`; max implied variance: `11.252888`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.588458 + 0.424659\,\pi_t + 0.209958\,\pi_{t-1} + 0.122050\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.314423\,\omega_{p,t} - 0.033006\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 5.523491 + 0.817999\,p_{t-1} + \frac{0.047638}{2(0.314423)^2}\,u_{t-1}^2,\\
n_t &= 0.570582 + 0.007286\,n_{t-1} + \frac{0.678814}{2(0.033006)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.588458 |
| rho_1 | 0.424659 |
| rho_2 | 0.209958 |
| phi_1 | 0.122050 |
| p0 | 5.523491 |
| n0 | 0.570582 |
| rho_p | 0.817999 |
| rho_n | 0.007286 |
| phi_p | 0.047638 |
| phi_n | 0.678814 |
| sigma_p | 0.314423 |
| sigma_n | 0.033006 |

### Rank 4: Seed 2, Draw 32

- LogLik: `-72.190224`; AIC: `168.380448`; BIC: `208.828105`
- Max shape path: `178.018364`; max implied variance: `2.558309`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.133147 + 0.331288\,\pi_t + 0.149878\,\pi_{t-1} + 0.671700\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.040932\,\omega_{p,t} - 0.141988\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 2.891717 + 0.957764\,p_{t-1} + \frac{0.012760}{2(0.040932)^2}\,u_{t-1}^2,\\
n_t &= 5.914930 + 0.250195\,n_{t-1} + \frac{0.210882}{2(0.141988)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.133147 |
| rho_1 | 0.331288 |
| rho_2 | 0.149878 |
| phi_1 | 0.671700 |
| p0 | 2.891717 |
| n0 | 5.914930 |
| rho_p | 0.957764 |
| rho_n | 0.250195 |
| phi_p | 0.012760 |
| phi_n | 0.210882 |
| sigma_p | 0.040932 |
| sigma_n | 0.141988 |

### Rank 5: Seed 8, Draw 1

- LogLik: `-92.524490`; AIC: `209.048981`; BIC: `249.496637`
- Max shape path: `1885.253562`; max implied variance: `6.440858`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.212374 + 0.202864\,\pi_t + 0.035718\,\pi_{t-1} + 0.722230\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.022091\,\omega_{p,t} - 0.196116\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 8.014245 + 0.181151\,p_{t-1} + \frac{0.100607}{2(0.022091)^2}\,u_{t-1}^2,\\
n_t &= 9.777479 + 0.243178\,n_{t-1} + \frac{0.551071}{2(0.196116)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.212374 |
| rho_1 | 0.202864 |
| rho_2 | 0.035718 |
| phi_1 | 0.722230 |
| p0 | 8.014245 |
| n0 | 9.777479 |
| rho_p | 0.181151 |
| rho_n | 0.243178 |
| phi_p | 0.100607 |
| phi_n | 0.551071 |
| sigma_p | 0.022091 |
| sigma_n | 0.196116 |

## ARX(2,2)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 10 | 25 | -61.925184 | 149.850368 | 193.668662 | 262.085151 | 4.464018 | yes |
| 2 | 26 | 19 | -66.378246 | 158.756492 | 202.574786 | 3813.897751 | 5.728508 | yes |
| 3 | 24 | 7 | -79.902479 | 185.804957 | 229.623252 | 168.015386 | 2.812712 | yes |
| 4 | 6 | 11 | -86.068763 | 198.137526 | 241.955820 | 280.228267 | 3.060894 | yes |
| 5 | 17 | 33 | -86.625742 | 199.251484 | 243.069778 | 177.450296 | 6.169924 | yes |

### Rank 1: Seed 10, Draw 25

- LogLik: `-61.925184`; AIC: `149.850368`; BIC: `193.668662`
- Max shape path: `262.085151`; max implied variance: `4.464018`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.073215 + 0.369573\,\pi_t + 0.261927\,\pi_{t-1} + 0.055788\,SPF_t + 0.279174\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.027712\,\omega_{p,t} - 0.150845\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 5.013663 + 0.964615\,p_{t-1} + \frac{0.005810}{2(0.027712)^2}\,u_{t-1}^2,\\
n_t &= 2.925165 + 0.249291\,n_{t-1} + \frac{0.425049}{2(0.150845)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.073215 |
| rho_1 | 0.369573 |
| rho_2 | 0.261927 |
| phi_1 | 0.055788 |
| phi_2 | 0.279174 |
| p0 | 5.013663 |
| n0 | 2.925165 |
| rho_p | 0.964615 |
| rho_n | 0.249291 |
| phi_p | 0.005810 |
| phi_n | 0.425049 |
| sigma_p | 0.027712 |
| sigma_n | 0.150845 |

### Rank 2: Seed 26, Draw 19

- LogLik: `-66.378246`; AIC: `158.756492`; BIC: `202.574786`
- Max shape path: `3813.897751`; max implied variance: `5.728508`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.224682 + 0.388085\,\pi_t + 0.142189\,\pi_{t-1} + -0.509452\,SPF_t + 0.925216\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.028431\,\omega_{p,t} - 0.265060\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 7.273837 + 0.556710\,p_{t-1} + \frac{0.308945}{2(0.028431)^2}\,u_{t-1}^2,\\
n_t &= 3.133774 + 0.883901\,n_{t-1} + \frac{0.065642}{2(0.265060)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.224682 |
| rho_1 | 0.388085 |
| rho_2 | 0.142189 |
| phi_1 | -0.509452 |
| phi_2 | 0.925216 |
| p0 | 7.273837 |
| n0 | 3.133774 |
| rho_p | 0.556710 |
| rho_n | 0.883901 |
| phi_p | 0.308945 |
| phi_n | 0.065642 |
| sigma_p | 0.028431 |
| sigma_n | 0.265060 |

### Rank 3: Seed 24, Draw 7

- LogLik: `-79.902479`; AIC: `185.804957`; BIC: `229.623252`
- Max shape path: `168.015386`; max implied variance: `2.812712`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.250460 + 0.390960\,\pi_t + 0.134068\,\pi_{t-1} + 0.402582\,SPF_t + -0.139047\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.177933\,\omega_{p,t} - 0.047487\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.873331 + 0.344976\,p_{t-1} + \frac{0.246582}{2(0.177933)^2}\,u_{t-1}^2,\\
n_t &= 8.360788 + 0.768672\,n_{t-1} + \frac{0.029500}{2(0.047487)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.250460 |
| rho_1 | 0.390960 |
| rho_2 | 0.134068 |
| phi_1 | 0.402582 |
| phi_2 | -0.139047 |
| p0 | 1.873331 |
| n0 | 8.360788 |
| rho_p | 0.344976 |
| rho_n | 0.768672 |
| phi_p | 0.246582 |
| phi_n | 0.029500 |
| sigma_p | 0.177933 |
| sigma_n | 0.047487 |

### Rank 4: Seed 6, Draw 11

- LogLik: `-86.068763`; AIC: `198.137526`; BIC: `241.955820`
- Max shape path: `280.228267`; max implied variance: `3.060894`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.080364 + 0.229754\,\pi_t + 0.152536\,\pi_{t-1} + 0.132234\,SPF_t + 0.486496\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.025256\,\omega_{p,t} - 0.121015\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.830372 + 0.763805\,p_{t-1} + \frac{0.017249}{2(0.025256)^2}\,u_{t-1}^2,\\
n_t &= 9.134264 + 0.571020\,n_{t-1} + \frac{0.290531}{2(0.121015)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.080364 |
| rho_1 | 0.229754 |
| rho_2 | 0.152536 |
| phi_1 | 0.132234 |
| phi_2 | 0.486496 |
| p0 | 6.830372 |
| n0 | 9.134264 |
| rho_p | 0.763805 |
| rho_n | 0.571020 |
| phi_p | 0.017249 |
| phi_n | 0.290531 |
| sigma_p | 0.025256 |
| sigma_n | 0.121015 |

### Rank 5: Seed 17, Draw 33

- LogLik: `-86.625742`; AIC: `199.251484`; BIC: `243.069778`
- Max shape path: `177.450296`; max implied variance: `6.169924`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.442113 + 0.081556\,\pi_t + 0.155466\,\pi_{t-1} + -0.000547\,SPF_t + 0.495184\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.044881\,\omega_{p,t} - 0.450802\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.209099 + 0.970162\,p_{t-1} + \frac{0.018103}{2(0.044881)^2}\,u_{t-1}^2,\\
n_t &= 0.000035 + 0.430336\,n_{t-1} + \frac{0.569585}{2(0.450802)^2}\,u_{t-1}^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.442113 |
| rho_1 | 0.081556 |
| rho_2 | 0.155466 |
| phi_1 | -0.000547 |
| phi_2 | 0.495184 |
| p0 | 1.209099 |
| n0 | 0.000035 |
| rho_p | 0.970162 |
| rho_n | 0.430336 |
| phi_p | 0.018103 |
| phi_n | 0.569585 |
| sigma_p | 0.044881 |
| sigma_n | 0.450802 |
