```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-06-03T14:06:09`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7999`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `5` admissible estimates by corrected log likelihood.

```{note}
Flagged 54 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## constant

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 28 | 30 | -62.493516 | 138.987032 | 162.581498 | 11244.168351 | 2.215097 | yes |
| 2 | 28 | 29 | -86.297753 | 186.595506 | 210.189972 | 10058.659020 | 3.410985 | yes |
| 3 | 35 | 28 | -106.188964 | 226.377927 | 249.972393 | 1182.972538 | 2.380025 | yes |
| 4 | 1 | 3 | -106.822942 | 227.645884 | 251.240350 | 5204.688927 | 55.771103 | yes |
| 5 | 21 | 11 | -112.500921 | 239.001842 | 262.596309 | 1588.336820 | 2.705783 | yes |

### Rank 1: Seed 28, Draw 30

- LogLik: `-62.493516`; AIC: `138.987032`; BIC: `162.581498`
- Max shape path: `11244.168351`; max implied variance: `2.215097`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.009537\,\omega_{p,t} - 0.081106\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.256316 + 0.785845\,p_{t-1} + \frac{0.233777}{2(0.009537)^2}\,(u_{t-1}^+)^2 + \frac{0.000000}{2(0.009537)^2}\,(u_{t-1}^-)^2,\\
n_t &= 5.530094 + 0.785845\,n_{t-1} + \frac{0.233777}{2(0.081106)^2}\,(u_{t-1}^+)^2 + \frac{0.000000}{2(0.081106)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 0.256316 |
| n0 | 5.530094 |
| rho | 0.785845 |
| phi_plus | 0.233777 |
| phi_minus | 0.000000 |
| sigma_p | 0.009537 |
| sigma_n | 0.081106 |

### Rank 2: Seed 28, Draw 29

- LogLik: `-86.297753`; AIC: `186.595506`; BIC: `210.189972`
- Max shape path: `10058.659020`; max implied variance: `3.410985`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.110843\,\omega_{p,t} - 0.012538\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.390719 + 0.542217\,p_{t-1} + \frac{0.494590}{2(0.110843)^2}\,(u_{t-1}^+)^2 + \frac{0.081859}{2(0.110843)^2}\,(u_{t-1}^-)^2,\\
n_t &= 10.000000 + 0.542217\,n_{t-1} + \frac{0.494590}{2(0.012538)^2}\,(u_{t-1}^+)^2 + \frac{0.081859}{2(0.012538)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 9.390719 |
| n0 | 10.000000 |
| rho | 0.542217 |
| phi_plus | 0.494590 |
| phi_minus | 0.081859 |
| sigma_p | 0.110843 |
| sigma_n | 0.012538 |

### Rank 3: Seed 35, Draw 28

- LogLik: `-106.188964`; AIC: `226.377927`; BIC: `249.972393`
- Max shape path: `1182.972538`; max implied variance: `2.380025`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.030905\,\omega_{p,t} - 0.056737\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.002195 + 0.732299\,p_{t-1} + \frac{0.172029}{2(0.030905)^2}\,(u_{t-1}^+)^2 + \frac{0.125657}{2(0.030905)^2}\,(u_{t-1}^-)^2,\\
n_t &= 9.997876 + 0.732299\,n_{t-1} + \frac{0.172029}{2(0.056737)^2}\,(u_{t-1}^+)^2 + \frac{0.125657}{2(0.056737)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 0.002195 |
| n0 | 9.997876 |
| rho | 0.732299 |
| phi_plus | 0.172029 |
| phi_minus | 0.125657 |
| sigma_p | 0.030905 |
| sigma_n | 0.056737 |

### Rank 4: Seed 1, Draw 3

- LogLik: `-106.822942`; AIC: `227.645884`; BIC: `251.240350`
- Max shape path: `5204.688927`; max implied variance: `55.771103`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.103507\,\omega_{p,t} - 0.023368\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.883002 + 0.632256\,p_{t-1} + \frac{0.718574}{2(0.103507)^2}\,(u_{t-1}^+)^2 + \frac{0.013116}{2(0.103507)^2}\,(u_{t-1}^-)^2,\\
n_t &= 0.032333 + 0.632256\,n_{t-1} + \frac{0.718574}{2(0.023368)^2}\,(u_{t-1}^+)^2 + \frac{0.013116}{2(0.023368)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 9.883002 |
| n0 | 0.032333 |
| rho | 0.632256 |
| phi_plus | 0.718574 |
| phi_minus | 0.013116 |
| sigma_p | 0.103507 |
| sigma_n | 0.023368 |

### Rank 5: Seed 21, Draw 11

- LogLik: `-112.500921`; AIC: `239.001842`; BIC: `262.596309`
- Max shape path: `1588.336820`; max implied variance: `2.705783`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.085129\,\omega_{p,t} - 0.026699\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.937734 + 0.880627\,p_{t-1} + \frac{0.112556}{2(0.085129)^2}\,(u_{t-1}^+)^2 + \frac{0.115001}{2(0.085129)^2}\,(u_{t-1}^-)^2,\\
n_t &= 1.535849 + 0.880627\,n_{t-1} + \frac{0.112556}{2(0.026699)^2}\,(u_{t-1}^+)^2 + \frac{0.115001}{2(0.026699)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 1.937734 |
| n0 | 1.535849 |
| rho | 0.880627 |
| phi_plus | 0.112556 |
| phi_minus | 0.115001 |
| sigma_p | 0.085129 |
| sigma_n | 0.026699 |

## ARX(1,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 33 | 32 | 250.490969 | -480.981939 | -447.275558 | 333.583931 | 11.278357 | yes |
| 2 | 20 | 5 | -2.584402 | 25.168805 | 58.875185 | 1690.545280 | 4.211158 | yes |
| 3 | 42 | 4 | -16.634979 | 53.269958 | 86.976338 | 3803.017359 | 4.543585 | yes |
| 4 | 43 | 9 | -40.183729 | 100.367458 | 134.073838 | 10624.346451 | 3.317568 | yes |
| 5 | 8 | 9 | -94.216877 | 208.433754 | 242.140134 | 29093.994442 | 7.121775 | yes |

### Rank 1: Seed 33, Draw 32

- LogLik: `250.490969`; AIC: `-480.981939`; BIC: `-447.275558`
- Max shape path: `333.583931`; max implied variance: `11.278357`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.166410 + 0.176424\,\pi_t + 0.694792\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.450682\,\omega_{p,t} - 0.041885\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.417208 + 0.943264\,p_{t-1} + \frac{0.049721}{2(0.450682)^2}\,(u_{t-1}^+)^2 + \frac{0.010220}{2(0.450682)^2}\,(u_{t-1}^-)^2,\\
n_t &= 7.985130 + 0.943264\,n_{t-1} + \frac{0.049721}{2(0.041885)^2}\,(u_{t-1}^+)^2 + \frac{0.010220}{2(0.041885)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.166410 |
| rho_1 | 0.176424 |
| phi_1 | 0.694792 |
| p0 | 1.417208 |
| n0 | 7.985130 |
| rho | 0.943264 |
| phi_plus | 0.049721 |
| phi_minus | 0.010220 |
| sigma_p | 0.450682 |
| sigma_n | 0.041885 |

### Rank 2: Seed 20, Draw 5

- LogLik: `-2.584402`; AIC: `25.168805`; BIC: `58.875185`
- Max shape path: `1690.545280`; max implied variance: `4.211158`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.437768 + 0.082676\,\pi_t + 1.493895\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.303172\,\omega_{p,t} - 0.027871\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.736192 + 0.613442\,p_{t-1} + \frac{0.252264}{2(0.303172)^2}\,(u_{t-1}^+)^2 + \frac{0.152797}{2(0.303172)^2}\,(u_{t-1}^-)^2,\\
n_t &= 8.456699 + 0.613442\,n_{t-1} + \frac{0.252264}{2(0.027871)^2}\,(u_{t-1}^+)^2 + \frac{0.152797}{2(0.027871)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.437768 |
| rho_1 | 0.082676 |
| phi_1 | 1.493895 |
| p0 | 6.736192 |
| n0 | 8.456699 |
| rho | 0.613442 |
| phi_plus | 0.252264 |
| phi_minus | 0.152797 |
| sigma_p | 0.303172 |
| sigma_n | 0.027871 |

### Rank 3: Seed 42, Draw 4

- LogLik: `-16.634979`; AIC: `53.269958`; BIC: `86.976338`
- Max shape path: `3803.017359`; max implied variance: `4.543585`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.246654 + 0.409457\,\pi_t + 1.075944\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.023935\,\omega_{p,t} - 0.134657\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.292586 + 0.807807\,p_{t-1} + \frac{0.088379}{2(0.023935)^2}\,(u_{t-1}^+)^2 + \frac{0.243758}{2(0.023935)^2}\,(u_{t-1}^-)^2,\\
n_t &= 2.170878 + 0.807807\,n_{t-1} + \frac{0.088379}{2(0.134657)^2}\,(u_{t-1}^+)^2 + \frac{0.243758}{2(0.134657)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.246654 |
| rho_1 | 0.409457 |
| phi_1 | 1.075944 |
| p0 | 6.292586 |
| n0 | 2.170878 |
| rho | 0.807807 |
| phi_plus | 0.088379 |
| phi_minus | 0.243758 |
| sigma_p | 0.023935 |
| sigma_n | 0.134657 |

### Rank 4: Seed 43, Draw 9

- LogLik: `-40.183729`; AIC: `100.367458`; BIC: `134.073838`
- Max shape path: `10624.346451`; max implied variance: `3.317568`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.125728 + 0.195998\,\pi_t + 0.578810\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.011933\,\omega_{p,t} - 0.107328\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.521635 + 0.802419\,p_{t-1} + \frac{0.297975}{2(0.011933)^2}\,(u_{t-1}^+)^2 + \frac{0.000000}{2(0.011933)^2}\,(u_{t-1}^-)^2,\\
n_t &= 5.125495 + 0.802419\,n_{t-1} + \frac{0.297975}{2(0.107328)^2}\,(u_{t-1}^+)^2 + \frac{0.000000}{2(0.107328)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.125728 |
| rho_1 | 0.195998 |
| phi_1 | 0.578810 |
| p0 | 9.521635 |
| n0 | 5.125495 |
| rho | 0.802419 |
| phi_plus | 0.297975 |
| phi_minus | 0.000000 |
| sigma_p | 0.011933 |
| sigma_n | 0.107328 |

### Rank 5: Seed 8, Draw 9

- LogLik: `-94.216877`; AIC: `208.433754`; BIC: `242.140134`
- Max shape path: `29093.994442`; max implied variance: `7.121775`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.297414 + 0.244068\,\pi_t + 0.533529\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.010810\,\omega_{p,t} - 0.137945\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.642971 + 0.637719\,p_{t-1} + \frac{0.315531}{2(0.010810)^2}\,(u_{t-1}^+)^2 + \frac{0.374156}{2(0.010810)^2}\,(u_{t-1}^-)^2,\\
n_t &= 6.183895 + 0.637719\,n_{t-1} + \frac{0.315531}{2(0.137945)^2}\,(u_{t-1}^+)^2 + \frac{0.374156}{2(0.137945)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.297414 |
| rho_1 | 0.244068 |
| phi_1 | 0.533529 |
| p0 | 6.642971 |
| n0 | 6.183895 |
| rho | 0.637719 |
| phi_plus | 0.315531 |
| phi_minus | 0.374156 |
| sigma_p | 0.010810 |
| sigma_n | 0.137945 |

## ARX(2,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 21 | 40 | 253.476194 | -484.952389 | -447.875370 | 271.160360 | 23.471739 | yes |
| 2 | 21 | 39 | -20.703249 | 63.406498 | 100.483516 | 4013.468681 | 1.246669 | yes |
| 3 | 43 | 25 | -89.071219 | 200.142438 | 237.219456 | 2305.724576 | 5.611361 | yes |
| 4 | 27 | 8 | -99.205068 | 220.410136 | 257.487154 | 428.778316 | 2.364721 | yes |
| 5 | 18 | 32 | -108.208535 | 238.417070 | 275.494089 | 4468.230081 | 4.578289 | yes |

### Rank 1: Seed 21, Draw 40

- LogLik: `253.476194`; AIC: `-484.952389`; BIC: `-447.875370`
- Max shape path: `271.160360`; max implied variance: `23.471739`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.027867 + 0.408254\,\pi_t + -0.033101\,\pi_{t-1} + 0.665318\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.076122\,\omega_{p,t} - 0.495743\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 8.161163 + 0.939951\,p_{t-1} + \frac{0.050597}{2(0.076122)^2}\,(u_{t-1}^+)^2 + \frac{0.009308}{2(0.076122)^2}\,(u_{t-1}^-)^2,\\
n_t &= 2.682046 + 0.939951\,n_{t-1} + \frac{0.050597}{2(0.495743)^2}\,(u_{t-1}^+)^2 + \frac{0.009308}{2(0.495743)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.027867 |
| rho_1 | 0.408254 |
| rho_2 | -0.033101 |
| phi_1 | 0.665318 |
| p0 | 8.161163 |
| n0 | 2.682046 |
| rho | 0.939951 |
| phi_plus | 0.050597 |
| phi_minus | 0.009308 |
| sigma_p | 0.076122 |
| sigma_n | 0.495743 |

### Rank 2: Seed 21, Draw 39

- LogLik: `-20.703249`; AIC: `63.406498`; BIC: `100.483516`
- Max shape path: `4013.468681`; max implied variance: `1.246669`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.007187 + 0.227721\,\pi_t + 0.072011\,\pi_{t-1} + 0.807644\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.011845\,\omega_{p,t} - 0.046743\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.073296 + 0.874843\,p_{t-1} + \frac{0.180942}{2(0.011845)^2}\,(u_{t-1}^+)^2 + \frac{0.006684}{2(0.011845)^2}\,(u_{t-1}^-)^2,\\
n_t &= 7.094145 + 0.874843\,n_{t-1} + \frac{0.180942}{2(0.046743)^2}\,(u_{t-1}^+)^2 + \frac{0.006684}{2(0.046743)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.007187 |
| rho_1 | 0.227721 |
| rho_2 | 0.072011 |
| phi_1 | 0.807644 |
| p0 | 3.073296 |
| n0 | 7.094145 |
| rho | 0.874843 |
| phi_plus | 0.180942 |
| phi_minus | 0.006684 |
| sigma_p | 0.011845 |
| sigma_n | 0.046743 |

### Rank 3: Seed 43, Draw 25

- LogLik: `-89.071219`; AIC: `200.142438`; BIC: `237.219456`
- Max shape path: `2305.724576`; max implied variance: `5.611361`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.225251 + 0.135374\,\pi_t + -0.084017\,\pi_{t-1} + 0.688499\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.029275\,\omega_{p,t} - 0.266324\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 4.396554 + 0.756702\,p_{t-1} + \frac{0.033864}{2(0.029275)^2}\,(u_{t-1}^+)^2 + \frac{0.305846}{2(0.029275)^2}\,(u_{t-1}^-)^2,\\
n_t &= 5.744938 + 0.756702\,n_{t-1} + \frac{0.033864}{2(0.266324)^2}\,(u_{t-1}^+)^2 + \frac{0.305846}{2(0.266324)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.225251 |
| rho_1 | 0.135374 |
| rho_2 | -0.084017 |
| phi_1 | 0.688499 |
| p0 | 4.396554 |
| n0 | 5.744938 |
| rho | 0.756702 |
| phi_plus | 0.033864 |
| phi_minus | 0.305846 |
| sigma_p | 0.029275 |
| sigma_n | 0.266324 |

### Rank 4: Seed 27, Draw 8

- LogLik: `-99.205068`; AIC: `220.410136`; BIC: `257.487154`
- Max shape path: `428.778316`; max implied variance: `2.364721`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.071479 + 0.141478\,\pi_t + 0.145558\,\pi_{t-1} + 0.294803\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.240745\,\omega_{p,t} - 0.030387\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.316360 + 0.771484\,p_{t-1} + \frac{0.051195}{2(0.240745)^2}\,(u_{t-1}^+)^2 + \frac{0.004422}{2(0.240745)^2}\,(u_{t-1}^-)^2,\\
n_t &= 7.203270 + 0.771484\,n_{t-1} + \frac{0.051195}{2(0.030387)^2}\,(u_{t-1}^+)^2 + \frac{0.004422}{2(0.030387)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.071479 |
| rho_1 | 0.141478 |
| rho_2 | 0.145558 |
| phi_1 | 0.294803 |
| p0 | 6.316360 |
| n0 | 7.203270 |
| rho | 0.771484 |
| phi_plus | 0.051195 |
| phi_minus | 0.004422 |
| sigma_p | 0.240745 |
| sigma_n | 0.030387 |

### Rank 5: Seed 18, Draw 32

- LogLik: `-108.208535`; AIC: `238.417070`; BIC: `275.494089`
- Max shape path: `4468.230081`; max implied variance: `4.578289`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.056344 + 0.548729\,\pi_t + 0.003487\,\pi_{t-1} + 0.492159\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.192974\,\omega_{p,t} - 0.020563\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 8.931381 + 0.586700\,p_{t-1} + \frac{0.599083}{2(0.192974)^2}\,(u_{t-1}^+)^2 + \frac{0.069957}{2(0.192974)^2}\,(u_{t-1}^-)^2,\\
n_t &= 5.064279 + 0.586700\,n_{t-1} + \frac{0.599083}{2(0.020563)^2}\,(u_{t-1}^+)^2 + \frac{0.069957}{2(0.020563)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.056344 |
| rho_1 | 0.548729 |
| rho_2 | 0.003487 |
| phi_1 | 0.492159 |
| p0 | 8.931381 |
| n0 | 5.064279 |
| rho | 0.586700 |
| phi_plus | 0.599083 |
| phi_minus | 0.069957 |
| sigma_p | 0.192974 |
| sigma_n | 0.020563 |

## ARX(2,2)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 8 | 30 | 103.495171 | -182.990343 | -142.542687 | 1259.701093 | 21.519226 | yes |
| 2 | 30 | 18 | 90.936287 | -157.872573 | -117.424917 | 2667.006996 | 5.553789 | yes |
| 3 | 30 | 6 | -6.141744 | 36.283488 | 76.731144 | 574.261901 | 2.525784 | yes |
| 4 | 21 | 14 | -46.378541 | 116.757081 | 157.204738 | 7096.098750 | 1.589305 | yes |
| 5 | 50 | 7 | -66.282420 | 156.564840 | 197.012496 | 467.575642 | 6.034697 | yes |

### Rank 1: Seed 8, Draw 30

- LogLik: `103.495171`; AIC: `-182.990343`; BIC: `-142.542687`
- Max shape path: `1259.701093`; max implied variance: `21.519226`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.043346 + 0.184817\,\pi_t + -0.046132\,\pi_{t-1} + 0.499300\,SPF_t + 0.667900\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.181297\,\omega_{p,t} - 0.028670\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 8.031497 + 0.778679\,p_{t-1} + \frac{0.394227}{2(0.181297)^2}\,(u_{t-1}^+)^2 + \frac{0.023401}{2(0.181297)^2}\,(u_{t-1}^-)^2,\\
n_t &= 6.281045 + 0.778679\,n_{t-1} + \frac{0.394227}{2(0.028670)^2}\,(u_{t-1}^+)^2 + \frac{0.023401}{2(0.028670)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.043346 |
| rho_1 | 0.184817 |
| rho_2 | -0.046132 |
| phi_1 | 0.499300 |
| phi_2 | 0.667900 |
| p0 | 8.031497 |
| n0 | 6.281045 |
| rho | 0.778679 |
| phi_plus | 0.394227 |
| phi_minus | 0.023401 |
| sigma_p | 0.181297 |
| sigma_n | 0.028670 |

### Rank 2: Seed 30, Draw 18

- LogLik: `90.936287`; AIC: `-157.872573`; BIC: `-117.424917`
- Max shape path: `2667.006996`; max implied variance: `5.553789`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.565703 + -0.170832\,\pi_t + -0.071699\,\pi_{t-1} + 1.480241\,SPF_t + 0.876347\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.200273\,\omega_{p,t} - 0.028054\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 8.760070 + 0.742767\,p_{t-1} + \frac{0.135301}{2(0.200273)^2}\,(u_{t-1}^+)^2 + \frac{0.221380}{2(0.200273)^2}\,(u_{t-1}^-)^2,\\
n_t &= 3.324015 + 0.742767\,n_{t-1} + \frac{0.135301}{2(0.028054)^2}\,(u_{t-1}^+)^2 + \frac{0.221380}{2(0.028054)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.565703 |
| rho_1 | -0.170832 |
| rho_2 | -0.071699 |
| phi_1 | 1.480241 |
| phi_2 | 0.876347 |
| p0 | 8.760070 |
| n0 | 3.324015 |
| rho | 0.742767 |
| phi_plus | 0.135301 |
| phi_minus | 0.221380 |
| sigma_p | 0.200273 |
| sigma_n | 0.028054 |

### Rank 3: Seed 30, Draw 6

- LogLik: `-6.141744`; AIC: `36.283488`; BIC: `76.731144`
- Max shape path: `574.261901`; max implied variance: `2.525784`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.084629 + 0.236799\,\pi_t + 0.145062\,\pi_{t-1} + 0.548225\,SPF_t + 0.141850\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.030545\,\omega_{p,t} - 0.202755\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.735992 + 0.795260\,p_{t-1} + \frac{0.039509}{2(0.030545)^2}\,(u_{t-1}^+)^2 + \frac{0.055524}{2(0.030545)^2}\,(u_{t-1}^-)^2,\\
n_t &= 7.327373 + 0.795260\,n_{t-1} + \frac{0.039509}{2(0.202755)^2}\,(u_{t-1}^+)^2 + \frac{0.055524}{2(0.202755)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.084629 |
| rho_1 | 0.236799 |
| rho_2 | 0.145062 |
| phi_1 | 0.548225 |
| phi_2 | 0.141850 |
| p0 | 3.735992 |
| n0 | 7.327373 |
| rho | 0.795260 |
| phi_plus | 0.039509 |
| phi_minus | 0.055524 |
| sigma_p | 0.030545 |
| sigma_n | 0.202755 |

### Rank 4: Seed 21, Draw 14

- LogLik: `-46.378541`; AIC: `116.757081`; BIC: `157.204738`
- Max shape path: `7096.098750`; max implied variance: `1.589305`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.010316 + 0.237341\,\pi_t + 0.042893\,\pi_{t-1} + 0.536182\,SPF_t + 0.257009\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.010184\,\omega_{p,t} - 0.044505\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.000239 + 0.831275\,p_{t-1} + \frac{0.249302}{2(0.010184)^2}\,(u_{t-1}^+)^2 + \frac{0.000001}{2(0.010184)^2}\,(u_{t-1}^-)^2,\\
n_t &= 9.999756 + 0.831275\,n_{t-1} + \frac{0.249302}{2(0.044505)^2}\,(u_{t-1}^+)^2 + \frac{0.000001}{2(0.044505)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.010316 |
| rho_1 | 0.237341 |
| rho_2 | 0.042893 |
| phi_1 | 0.536182 |
| phi_2 | 0.257009 |
| p0 | 0.000239 |
| n0 | 9.999756 |
| rho | 0.831275 |
| phi_plus | 0.249302 |
| phi_minus | 0.000001 |
| sigma_p | 0.010184 |
| sigma_n | 0.044505 |

### Rank 5: Seed 50, Draw 7

- LogLik: `-66.282420`; AIC: `156.564840`; BIC: `197.012496`
- Max shape path: `467.575642`; max implied variance: `6.034697`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.105336 + -0.079264\,\pi_t + -0.163079\,\pi_{t-1} + -0.568168\,SPF_t + 1.346827\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.047203\,\omega_{p,t} - 0.275456\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.990700 + 0.890070\,p_{t-1} + \frac{0.046611}{2(0.047203)^2}\,(u_{t-1}^+)^2 + \frac{0.008521}{2(0.047203)^2}\,(u_{t-1}^-)^2,\\
n_t &= 5.772698 + 0.890070\,n_{t-1} + \frac{0.046611}{2(0.275456)^2}\,(u_{t-1}^+)^2 + \frac{0.008521}{2(0.275456)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.105336 |
| rho_1 | -0.079264 |
| rho_2 | -0.163079 |
| phi_1 | -0.568168 |
| phi_2 | 1.346827 |
| p0 | 1.990700 |
| n0 | 5.772698 |
| rho | 0.890070 |
| phi_plus | 0.046611 |
| phi_minus | 0.008521 |
| sigma_p | 0.047203 |
| sigma_n | 0.275456 |
