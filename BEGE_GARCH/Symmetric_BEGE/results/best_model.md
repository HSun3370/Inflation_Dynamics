```{raw:typst}
#set page(margin: auto)
```

# Symmetric BEGE Best Model Summary

Generated: `2026-06-02T13:20:12`
Total estimations: `8000`
Converged estimations: `8000`
Eligible estimations for best-model selection: `7999`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `20` admissible estimates by corrected log likelihood. Standard errors are shown below substituted equation coefficients in parentheses.

```{note}
Flagged 54 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## constant

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 28 | 30 | -62.493516 | 138.987032 | 162.581498 | 11244.168351 | 2.215097 | yes | `computed` |
| 2 | 28 | 29 | -86.297753 | 186.595506 | 210.189972 | 10058.659020 | 3.410985 | yes | `computed` |
| 3 | 35 | 28 | -106.188964 | 226.377927 | 249.972393 | 1182.972538 | 2.380025 | yes | `computed` |
| 4 | 1 | 3 | -106.822942 | 227.645884 | 251.240350 | 5204.688927 | 55.771103 | yes | `computed` |
| 5 | 21 | 11 | -112.500921 | 239.001842 | 262.596309 | 1588.336820 | 2.705783 | yes | `computed` |
| 6 | 9 | 40 | -116.060222 | 246.120445 | 269.714911 | 686.085700 | 1.693877 | yes | `computed` |
| 7 | 4 | 40 | -125.851486 | 265.702971 | 289.297437 | 2819.642465 | 1.673329 | yes | `computed` |
| 8 | 3 | 3 | -125.975205 | 265.950409 | 289.544875 | 13683.632131 | 7.797295 | yes | `computed` |
| 9 | 11 | 27 | -133.362159 | 280.724318 | 304.318785 | 22676.149392 | 2.918690 | yes | `computed` |
| 10 | 39 | 19 | -135.473482 | 284.946963 | 308.541429 | 2431.873841 | 1.912500 | yes | `computed` |
| 11 | 33 | 12 | -140.153227 | 294.306453 | 317.900919 | 3847.827002 | 2.729608 | yes | `computed` |
| 12 | 13 | 31 | -140.215971 | 294.431941 | 318.026407 | 3578.863569 | 3.946531 | yes | `computed` |
| 13 | 3 | 29 | -141.910333 | 297.820666 | 321.415132 | 16447.950499 | 6.807656 | yes | `computed` |
| 14 | 26 | 38 | -153.025733 | 320.051466 | 343.645932 | 1306.982704 | 17.187578 | no | `computed` |
| 15 | 21 | 17 | -157.423374 | 328.846748 | 352.441215 | 2236.620492 | 3.948726 | no | `computed` |
| 16 | 10 | 23 | -161.090651 | 336.181301 | 359.775767 | 6058.160148 | 9.841311 | no | `computed` |
| 17 | 10 | 34 | -161.489730 | 336.979460 | 360.573926 | 1065.053439 | 18.260138 | no | `computed` |
| 18 | 46 | 31 | -169.886850 | 353.773700 | 377.368166 | 3313.461977 | 3.164644 | no | `computed` |
| 19 | 30 | 22 | -174.048521 | 362.097043 | 385.691509 | 589.445116 | 156.597261 | no | `computed` |
| 20 | 21 | 33 | -176.123241 | 366.246483 | 389.840949 | 112.717874 | 4.452085 | no | `computed` |

### Rank 1: Seed 28, Draw 30

- LogLik: `-62.493516`; AIC: `138.987032`; BIC: `162.581498`
- Max shape path: `11244.168351`; max implied variance: `2.215097`
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
u_t &= \underset{(0.000029)}{0.009537}\,\omega_{p,t} - \underset{(0.268788)}{0.081106}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(351.637318)}{0.256316} + \underset{(0.431563)}{0.785845}\,p_{t-1} + \frac{\underset{(2.910382)}{0.233777}}{2(\underset{(0.000029)}{0.009537})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000723)}{0.000000}}{2(\underset{(0.000029)}{0.009537})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(3.608172)}{5.530094} + \underset{(0.431563)}{0.785845}\,n_{t-1} + \frac{\underset{(2.910382)}{0.233777}}{2(\underset{(0.268788)}{0.081106})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000723)}{0.000000}}{2(\underset{(0.268788)}{0.081106})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.256316 | 351.637318 |
| n0 | 5.530094 | 3.608172 |
| rho | 0.785845 | 0.431563 |
| phi_plus | 0.233777 | 2.910382 |
| phi_minus | 0.000000 | 0.000723 |
| sigma_p | 0.009537 | 0.000029 |
| sigma_n | 0.081106 | 0.268788 |

### Rank 2: Seed 28, Draw 29

- LogLik: `-86.297753`; AIC: `186.595506`; BIC: `210.189972`
- Max shape path: `10058.659020`; max implied variance: `3.410985`
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
u_t &= \underset{(0.553992)}{0.110843}\,\omega_{p,t} - \underset{(0.000282)}{0.012538}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.811094)}{9.390719} + \underset{(0.910202)}{0.542217}\,p_{t-1} + \frac{\underset{(2.208861)}{0.494590}}{2(\underset{(0.553992)}{0.110843})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.779647)}{0.081859}}{2(\underset{(0.553992)}{0.110843})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(3.815254)}{10.000000} + \underset{(0.910202)}{0.542217}\,n_{t-1} + \frac{\underset{(2.208861)}{0.494590}}{2(\underset{(0.000282)}{0.012538})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.779647)}{0.081859}}{2(\underset{(0.000282)}{0.012538})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.390719 | 1.811094 |
| n0 | 10.000000 | 3.815254 |
| rho | 0.542217 | 0.910202 |
| phi_plus | 0.494590 | 2.208861 |
| phi_minus | 0.081859 | 2.779647 |
| sigma_p | 0.110843 | 0.553992 |
| sigma_n | 0.012538 | 0.000282 |

### Rank 3: Seed 35, Draw 28

- LogLik: `-106.188964`; AIC: `226.377927`; BIC: `249.972393`
- Max shape path: `1182.972538`; max implied variance: `2.380025`
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
u_t &= \underset{(0.001467)}{0.030905}\,\omega_{p,t} - \underset{(0.028747)}{0.056737}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(20.997658)}{0.002195} + \underset{(0.159179)}{0.732299}\,p_{t-1} + \frac{\underset{(0.134906)}{0.172029}}{2(\underset{(0.001467)}{0.030905})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.725620)}{0.125657}}{2(\underset{(0.001467)}{0.030905})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(14.874776)}{9.997876} + \underset{(0.159179)}{0.732299}\,n_{t-1} + \frac{\underset{(0.134906)}{0.172029}}{2(\underset{(0.028747)}{0.056737})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.725620)}{0.125657}}{2(\underset{(0.028747)}{0.056737})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.002195 | 20.997658 |
| n0 | 9.997876 | 14.874776 |
| rho | 0.732299 | 0.159179 |
| phi_plus | 0.172029 | 0.134906 |
| phi_minus | 0.125657 | 0.725620 |
| sigma_p | 0.030905 | 0.001467 |
| sigma_n | 0.056737 | 0.028747 |

### Rank 4: Seed 1, Draw 3

- LogLik: `-106.822942`; AIC: `227.645884`; BIC: `251.240350`
- Max shape path: `5204.688927`; max implied variance: `55.771103`
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
u_t &= \underset{(0.194435)}{0.103507}\,\omega_{p,t} - \underset{(0.000448)}{0.023368}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(23.446918)}{9.883002} + \underset{(0.000468)}{0.632256}\,p_{t-1} + \frac{\underset{(0.071977)}{0.718574}}{2(\underset{(0.194435)}{0.103507})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.071524)}{0.013116}}{2(\underset{(0.194435)}{0.103507})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(13.789638)}{0.032333} + \underset{(0.000468)}{0.632256}\,n_{t-1} + \frac{\underset{(0.071977)}{0.718574}}{2(\underset{(0.000448)}{0.023368})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.071524)}{0.013116}}{2(\underset{(0.000448)}{0.023368})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.883002 | 23.446918 |
| n0 | 0.032333 | 13.789638 |
| rho | 0.632256 | 0.000468 |
| phi_plus | 0.718574 | 0.071977 |
| phi_minus | 0.013116 | 0.071524 |
| sigma_p | 0.103507 | 0.194435 |
| sigma_n | 0.023368 | 0.000448 |

### Rank 5: Seed 21, Draw 11

- LogLik: `-112.500921`; AIC: `239.001842`; BIC: `262.596309`
- Max shape path: `1588.336820`; max implied variance: `2.705783`
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
u_t &= \underset{(0.119041)}{0.085129}\,\omega_{p,t} - \underset{(0.000060)}{0.026699}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.658747)}{1.937734} + \underset{(0.000056)}{0.880627}\,p_{t-1} + \frac{\underset{(0.000035)}{0.112556}}{2(\underset{(0.119041)}{0.085129})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000040)}{0.115001}}{2(\underset{(0.119041)}{0.085129})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.058483)}{1.535849} + \underset{(0.000056)}{0.880627}\,n_{t-1} + \frac{\underset{(0.000035)}{0.112556}}{2(\underset{(0.000060)}{0.026699})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000040)}{0.115001}}{2(\underset{(0.000060)}{0.026699})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 1.937734 | 0.658747 |
| n0 | 1.535849 | 0.058483 |
| rho | 0.880627 | 0.000056 |
| phi_plus | 0.112556 | 0.000035 |
| phi_minus | 0.115001 | 0.000040 |
| sigma_p | 0.085129 | 0.119041 |
| sigma_n | 0.026699 | 0.000060 |

### Rank 6: Seed 9, Draw 40

- LogLik: `-116.060222`; AIC: `246.120445`; BIC: `269.714911`
- Max shape path: `686.085700`; max implied variance: `1.693877`
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
u_t &= \underset{(0.000034)}{0.034123}\,\omega_{p,t} - \underset{(0.000271)}{0.052907}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(17281.644059)}{0.003556} + \underset{(226.555467)}{0.708848}\,p_{t-1} + \frac{\underset{(66.953694)}{0.208104}}{2(\underset{(0.000034)}{0.034123})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(370.555094)}{0.027096}}{2(\underset{(0.000034)}{0.034123})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(14763.364389)}{9.998942} + \underset{(226.555467)}{0.708848}\,n_{t-1} + \frac{\underset{(66.953694)}{0.208104}}{2(\underset{(0.000271)}{0.052907})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(370.555094)}{0.027096}}{2(\underset{(0.000271)}{0.052907})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.003556 | 17281.644059 |
| n0 | 9.998942 | 14763.364389 |
| rho | 0.708848 | 226.555467 |
| phi_plus | 0.208104 | 66.953694 |
| phi_minus | 0.027096 | 370.555094 |
| sigma_p | 0.034123 | 0.000034 |
| sigma_n | 0.052907 | 0.000271 |

### Rank 7: Seed 4, Draw 40

- LogLik: `-125.851486`; AIC: `265.702971`; BIC: `289.297437`
- Max shape path: `2819.642465`; max implied variance: `1.673329`
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
u_t &= \underset{(0.000017)}{0.014626}\,\omega_{p,t} - \underset{(0.101144)}{0.166667}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.660966)}{0.175805} + \underset{(0.116667)}{0.429682}\,p_{t-1} + \frac{\underset{(0.039656)}{0.126789}}{2(\underset{(0.000017)}{0.014626})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000002)}{0.072091}}{2(\underset{(0.000017)}{0.014626})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(2.986559)}{9.589660} + \underset{(0.116667)}{0.429682}\,n_{t-1} + \frac{\underset{(0.039656)}{0.126789}}{2(\underset{(0.101144)}{0.166667})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000002)}{0.072091}}{2(\underset{(0.101144)}{0.166667})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.175805 | 1.660966 |
| n0 | 9.589660 | 2.986559 |
| rho | 0.429682 | 0.116667 |
| phi_plus | 0.126789 | 0.039656 |
| phi_minus | 0.072091 | 0.000002 |
| sigma_p | 0.014626 | 0.000017 |
| sigma_n | 0.166667 | 0.101144 |

### Rank 8: Seed 3, Draw 3

- LogLik: `-125.975205`; AIC: `265.950409`; BIC: `289.544875`
- Max shape path: `13683.632131`; max implied variance: `7.797295`
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
u_t &= \underset{(0.017275)}{0.264911}\,\omega_{p,t} - \underset{(0.000006)}{0.016016}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.122513)}{7.443839} + \underset{(0.008498)}{0.329639}\,p_{t-1} + \frac{\underset{(0.201561)}{0.487935}}{2(\underset{(0.017275)}{0.264911})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.184777)}{0.429724}}{2(\underset{(0.017275)}{0.264911})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(45.440865)}{4.968896} + \underset{(0.008498)}{0.329639}\,n_{t-1} + \frac{\underset{(0.201561)}{0.487935}}{2(\underset{(0.000006)}{0.016016})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.184777)}{0.429724}}{2(\underset{(0.000006)}{0.016016})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 7.443839 | 0.122513 |
| n0 | 4.968896 | 45.440865 |
| rho | 0.329639 | 0.008498 |
| phi_plus | 0.487935 | 0.201561 |
| phi_minus | 0.429724 | 0.184777 |
| sigma_p | 0.264911 | 0.017275 |
| sigma_n | 0.016016 | 0.000006 |

### Rank 9: Seed 11, Draw 27

- LogLik: `-133.362159`; AIC: `280.724318`; BIC: `304.318785`
- Max shape path: `22676.149392`; max implied variance: `2.918690`
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
u_t &= \underset{(0.000973)}{0.007843}\,\omega_{p,t} - \underset{(1.435141)}{0.058702}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(2913.584661)}{1.032286} + \underset{(0.524290)}{0.734898}\,p_{t-1} + \frac{\underset{(19.987007)}{0.350502}}{2(\underset{(0.000973)}{0.007843})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.447715)}{0.000141}}{2(\underset{(0.000973)}{0.007843})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(104.032068)}{9.932763} + \underset{(0.524290)}{0.734898}\,n_{t-1} + \frac{\underset{(19.987007)}{0.350502}}{2(\underset{(1.435141)}{0.058702})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.447715)}{0.000141}}{2(\underset{(1.435141)}{0.058702})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 1.032286 | 2913.584661 |
| n0 | 9.932763 | 104.032068 |
| rho | 0.734898 | 0.524290 |
| phi_plus | 0.350502 | 19.987007 |
| phi_minus | 0.000141 | 0.447715 |
| sigma_p | 0.007843 | 0.000973 |
| sigma_n | 0.058702 | 1.435141 |

### Rank 10: Seed 39, Draw 19

- LogLik: `-135.473482`; AIC: `284.946963`; BIC: `308.541429`
- Max shape path: `2431.873841`; max implied variance: `1.912500`
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
u_t &= \underset{(0.000000)}{0.016272}\,\omega_{p,t} - \underset{(0.071441)}{0.201882}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(8.280593)}{0.234421} + \underset{(0.255613)}{0.347713}\,p_{t-1} + \frac{\underset{(0.000102)}{0.233539}}{2(\underset{(0.000000)}{0.016272})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000453)}{0.000493}}{2(\underset{(0.000000)}{0.016272})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(2.532494)}{10.000000} + \underset{(0.255613)}{0.347713}\,n_{t-1} + \frac{\underset{(0.000102)}{0.233539}}{2(\underset{(0.071441)}{0.201882})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000453)}{0.000493}}{2(\underset{(0.071441)}{0.201882})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.234421 | 8.280593 |
| n0 | 10.000000 | 2.532494 |
| rho | 0.347713 | 0.255613 |
| phi_plus | 0.233539 | 0.000102 |
| phi_minus | 0.000493 | 0.000453 |
| sigma_p | 0.016272 | 0.000000 |
| sigma_n | 0.201882 | 0.071441 |

### Rank 11: Seed 33, Draw 12

- LogLik: `-140.153227`; AIC: `294.306453`; BIC: `317.900919`
- Max shape path: `3847.827002`; max implied variance: `2.729608`
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
u_t &= \underset{(0.001610)}{0.018618}\,\omega_{p,t} - \underset{(1.146689)}{0.037479}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(5020.371401)}{0.030849} + \underset{(66.397723)}{0.774273}\,p_{t-1} + \frac{\underset{(52.692793)}{0.311449}}{2(\underset{(0.001610)}{0.018618})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(78.762843)}{0.036573}}{2(\underset{(0.001610)}{0.018618})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(4570.606427)}{9.999463} + \underset{(66.397723)}{0.774273}\,n_{t-1} + \frac{\underset{(52.692793)}{0.311449}}{2(\underset{(1.146689)}{0.037479})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(78.762843)}{0.036573}}{2(\underset{(1.146689)}{0.037479})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.030849 | 5020.371401 |
| n0 | 9.999463 | 4570.606427 |
| rho | 0.774273 | 66.397723 |
| phi_plus | 0.311449 | 52.692793 |
| phi_minus | 0.036573 | 78.762843 |
| sigma_p | 0.018618 | 0.001610 |
| sigma_n | 0.037479 | 1.146689 |

### Rank 12: Seed 13, Draw 31

- LogLik: `-140.215971`; AIC: `294.431941`; BIC: `318.026407`
- Max shape path: `3578.863569`; max implied variance: `3.946531`
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
u_t &= \underset{(0.000234)}{0.023176}\,\omega_{p,t} - \underset{(0.000239)}{0.082163}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(51.736175)}{0.058518} + \underset{(0.000254)}{0.651602}\,p_{t-1} + \frac{\underset{(0.000242)}{0.537874}}{2(\underset{(0.000234)}{0.023176})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000242)}{0.066173}}{2(\underset{(0.000234)}{0.023176})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(51.735640)}{5.256491} + \underset{(0.000254)}{0.651602}\,n_{t-1} + \frac{\underset{(0.000242)}{0.537874}}{2(\underset{(0.000239)}{0.082163})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000242)}{0.066173}}{2(\underset{(0.000239)}{0.082163})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.058518 | 51.736175 |
| n0 | 5.256491 | 51.735640 |
| rho | 0.651602 | 0.000254 |
| phi_plus | 0.537874 | 0.000242 |
| phi_minus | 0.066173 | 0.000242 |
| sigma_p | 0.023176 | 0.000234 |
| sigma_n | 0.082163 | 0.000239 |

### Rank 13: Seed 3, Draw 29

- LogLik: `-141.910333`; AIC: `297.820666`; BIC: `321.415132`
- Max shape path: `16447.950499`; max implied variance: `6.807656`
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
u_t &= \underset{(0.317497)}{0.171597}\,\omega_{p,t} - \underset{(0.004551)}{0.013850}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(14.622771)}{8.285343} + \underset{(3.538773)}{0.510531}\,p_{t-1} + \frac{\underset{(33.441203)}{0.340854}}{2(\underset{(0.317497)}{0.171597})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(44.382730)}{0.380665}}{2(\underset{(0.317497)}{0.171597})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(6594.590186)}{2.547726} + \underset{(3.538773)}{0.510531}\,n_{t-1} + \frac{\underset{(33.441203)}{0.340854}}{2(\underset{(0.004551)}{0.013850})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(44.382730)}{0.380665}}{2(\underset{(0.004551)}{0.013850})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 8.285343 | 14.622771 |
| n0 | 2.547726 | 6594.590186 |
| rho | 0.510531 | 3.538773 |
| phi_plus | 0.340854 | 33.441203 |
| phi_minus | 0.380665 | 44.382730 |
| sigma_p | 0.171597 | 0.317497 |
| sigma_n | 0.013850 | 0.004551 |

### Rank 14: Seed 26, Draw 38

- LogLik: `-153.025733`; AIC: `320.051466`; BIC: `343.645932`
- Max shape path: `1306.982704`; max implied variance: `17.187578`
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
u_t &= \underset{(0.080102)}{0.496961}\,\omega_{p,t} - \underset{(0.024093)}{0.051066}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.004488)}{0.228359} + \underset{(0.000020)}{0.374166}\,p_{t-1} + \frac{\underset{(1.115157)}{1.214891}}{2(\underset{(0.080102)}{0.496961})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.115251)}{0.029954}}{2(\underset{(0.080102)}{0.496961})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(1.639473)}{0.855681} + \underset{(0.000020)}{0.374166}\,n_{t-1} + \frac{\underset{(1.115157)}{1.214891}}{2(\underset{(0.024093)}{0.051066})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.115251)}{0.029954}}{2(\underset{(0.024093)}{0.051066})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.228359 | 0.004488 |
| n0 | 0.855681 | 1.639473 |
| rho | 0.374166 | 0.000020 |
| phi_plus | 1.214891 | 1.115157 |
| phi_minus | 0.029954 | 1.115251 |
| sigma_p | 0.496961 | 0.080102 |
| sigma_n | 0.051066 | 0.024093 |

### Rank 15: Seed 21, Draw 17

- LogLik: `-157.423374`; AIC: `328.846748`; BIC: `352.441215`
- Max shape path: `2236.620492`; max implied variance: `3.948726`
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
u_t &= \underset{(0.000001)}{0.029180}\,\omega_{p,t} - \underset{(9.915332)}{0.083050}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(309.638872)}{1.694617} + \underset{(14.275192)}{0.672192}\,p_{t-1} + \frac{\underset{(22.625399)}{0.518708}}{2(\underset{(0.000001)}{0.029180})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.518366)}{0.000286}}{2(\underset{(0.000001)}{0.029180})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(556.624803)}{6.863340} + \underset{(14.275192)}{0.672192}\,n_{t-1} + \frac{\underset{(22.625399)}{0.518708}}{2(\underset{(9.915332)}{0.083050})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.518366)}{0.000286}}{2(\underset{(9.915332)}{0.083050})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 1.694617 | 309.638872 |
| n0 | 6.863340 | 556.624803 |
| rho | 0.672192 | 14.275192 |
| phi_plus | 0.518708 | 22.625399 |
| phi_minus | 0.000286 | 1.518366 |
| sigma_p | 0.029180 | 0.000001 |
| sigma_n | 0.083050 | 9.915332 |

### Rank 16: Seed 10, Draw 23

- LogLik: `-161.090651`; AIC: `336.181301`; BIC: `359.775767`
- Max shape path: `6058.160148`; max implied variance: `9.841311`
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
u_t &= \underset{(0.000094)}{0.025438}\,\omega_{p,t} - \underset{(0.002640)}{0.367477}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(128.118797)}{9.080310} + \underset{(0.000069)}{0.651129}\,p_{t-1} + \frac{\underset{(0.023902)}{0.001183}}{2(\underset{(0.000094)}{0.025438})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.335262)}{0.484657}}{2(\underset{(0.000094)}{0.025438})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.209148)}{5.213149} + \underset{(0.000069)}{0.651129}\,n_{t-1} + \frac{\underset{(0.023902)}{0.001183}}{2(\underset{(0.002640)}{0.367477})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.335262)}{0.484657}}{2(\underset{(0.002640)}{0.367477})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.080310 | 128.118797 |
| n0 | 5.213149 | 0.209148 |
| rho | 0.651129 | 0.000069 |
| phi_plus | 0.001183 | 0.023902 |
| phi_minus | 0.484657 | 1.335262 |
| sigma_p | 0.025438 | 0.000094 |
| sigma_n | 0.367477 | 0.002640 |

### Rank 17: Seed 10, Draw 34

- LogLik: `-161.489730`; AIC: `336.979460`; BIC: `360.573926`
- Max shape path: `1065.053439`; max implied variance: `18.260138`
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
u_t &= \underset{(11.018127)}{0.055132}\,\omega_{p,t} - \underset{(9.822951)}{0.509572}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(5928.371143)}{3.177795} + \underset{(1.221854)}{0.411192}\,p_{t-1} + \frac{\underset{(2853.196235)}{1.121326}}{2(\underset{(11.018127)}{0.055132})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2930.724881)}{0.031203}}{2(\underset{(11.018127)}{0.055132})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(2425.735888)}{0.844893} + \underset{(1.221854)}{0.411192}\,n_{t-1} + \frac{\underset{(2853.196235)}{1.121326}}{2(\underset{(9.822951)}{0.509572})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2930.724881)}{0.031203}}{2(\underset{(9.822951)}{0.509572})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.177795 | 5928.371143 |
| n0 | 0.844893 | 2425.735888 |
| rho | 0.411192 | 1.221854 |
| phi_plus | 1.121326 | 2853.196235 |
| phi_minus | 0.031203 | 2930.724881 |
| sigma_p | 0.055132 | 11.018127 |
| sigma_n | 0.509572 | 9.822951 |

### Rank 18: Seed 46, Draw 31

- LogLik: `-169.886850`; AIC: `353.773700`; BIC: `377.368166`
- Max shape path: `3313.461977`; max implied variance: `3.164644`
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
u_t &= \underset{(0.000032)}{0.021574}\,\omega_{p,t} - \underset{(0.033524)}{0.052705}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(68.743029)}{9.519960} + \underset{(0.035918)}{0.709998}\,p_{t-1} + \frac{\underset{(0.822081)}{0.397191}}{2(\underset{(0.000032)}{0.021574})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.273339)}{0.015416}}{2(\underset{(0.000032)}{0.021574})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(15.495571)}{9.956749} + \underset{(0.035918)}{0.709998}\,n_{t-1} + \frac{\underset{(0.822081)}{0.397191}}{2(\underset{(0.033524)}{0.052705})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.273339)}{0.015416}}{2(\underset{(0.033524)}{0.052705})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 9.519960 | 68.743029 |
| n0 | 9.956749 | 15.495571 |
| rho | 0.709998 | 0.035918 |
| phi_plus | 0.397191 | 0.822081 |
| phi_minus | 0.015416 | 0.273339 |
| sigma_p | 0.021574 | 0.000032 |
| sigma_n | 0.052705 | 0.033524 |

### Rank 19: Seed 30, Draw 22

- LogLik: `-174.048521`; AIC: `362.097043`; BIC: `385.691509`
- Max shape path: `589.445116`; max implied variance: `156.597261`
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
u_t &= \underset{(0.018919)}{0.208929}\,\omega_{p,t} - \underset{(0.006794)}{0.552880}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.002945)}{0.253115} + \underset{(0.000036)}{0.594119}\,p_{t-1} + \frac{\underset{(0.001821)}{0.534813}}{2(\underset{(0.018919)}{0.208929})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.001819)}{0.276090}}{2(\underset{(0.018919)}{0.208929})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.000028)}{0.183842} + \underset{(0.000036)}{0.594119}\,n_{t-1} + \frac{\underset{(0.001821)}{0.534813}}{2(\underset{(0.006794)}{0.552880})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.001819)}{0.276090}}{2(\underset{(0.006794)}{0.552880})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 0.253115 | 0.002945 |
| n0 | 0.183842 | 0.000028 |
| rho | 0.594119 | 0.000036 |
| phi_plus | 0.534813 | 0.001821 |
| phi_minus | 0.276090 | 0.001819 |
| sigma_p | 0.208929 | 0.018919 |
| sigma_n | 0.552880 | 0.006794 |

### Rank 20: Seed 21, Draw 33

- LogLik: `-176.123241`; AIC: `366.246483`; BIC: `389.840949`
- Max shape path: `112.717874`; max implied variance: `4.452085`
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
u_t &= \underset{(0.048021)}{0.139871}\,\omega_{p,t} - \underset{(1.292038)}{1.496046}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.609784)}{2.877569} + \underset{(0.135362)}{0.423344}\,p_{t-1} + \frac{\underset{(0.201198)}{0.727508}}{2(\underset{(0.048021)}{0.139871})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.123303)}{0.134979}}{2(\underset{(0.048021)}{0.139871})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.030062)}{0.035896} + \underset{(0.135362)}{0.423344}\,n_{t-1} + \frac{\underset{(0.201198)}{0.727508}}{2(\underset{(1.292038)}{1.496046})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.123303)}{0.134979}}{2(\underset{(1.292038)}{1.496046})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 2.877569 | 1.609784 |
| n0 | 0.035896 | 0.030062 |
| rho | 0.423344 | 0.135362 |
| phi_plus | 0.727508 | 0.201198 |
| phi_minus | 0.134979 | 0.123303 |
| sigma_p | 0.139871 | 0.048021 |
| sigma_n | 1.496046 | 1.292038 |

## ARX(1,1)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 33 | 32 | 250.490969 | -480.981939 | -447.275558 | 333.583931 | 11.278357 | yes | `computed` |
| 2 | 20 | 5 | -2.584402 | 25.168805 | 58.875185 | 1690.545280 | 4.211158 | yes | `computed` |
| 3 | 42 | 4 | -16.634979 | 53.269958 | 86.976338 | 3803.017359 | 4.543585 | yes | `computed` |
| 4 | 43 | 9 | -40.183729 | 100.367458 | 134.073838 | 10624.346451 | 3.317568 | yes | `computed` |
| 5 | 8 | 9 | -94.216877 | 208.433754 | 242.140134 | 29093.994442 | 7.121775 | yes | `computed` |
| 6 | 40 | 35 | -103.729216 | 227.458432 | 261.164812 | 31113.806343 | 3.927631 | yes | `computed` |
| 7 | 47 | 31 | -109.917679 | 239.835359 | 273.541739 | 2220.085146 | 4.273532 | yes | `computed` |
| 8 | 33 | 1 | -110.700824 | 241.401648 | 275.108028 | 950.488040 | 49.639436 | yes | `computed` |
| 9 | 20 | 30 | -113.652118 | 247.304236 | 281.010616 | 1418.023476 | 11.761217 | yes | `computed` |
| 10 | 26 | 3 | -126.373740 | 272.747480 | 306.453861 | 990.074774 | 35.411762 | yes | `computed` |
| 11 | 15 | 37 | -131.051663 | 282.103326 | 315.809706 | 583.241014 | 19.491296 | yes | `computed` |
| 12 | 11 | 21 | -134.377082 | 288.754164 | 322.460544 | 791.922237 | 34.196184 | yes | `computed` |
| 13 | 38 | 26 | -138.229694 | 296.459388 | 330.165769 | 1068.681196 | 1.266514 | yes | `computed` |
| 14 | 7 | 20 | -142.606992 | 305.213984 | 338.920364 | 15080.298303 | 3.965045 | yes | `computed` |
| 15 | 32 | 28 | -144.250493 | 308.500985 | 342.207365 | 232.568370 | 59.355582 | yes | `computed` |
| 16 | 22 | 10 | -150.796535 | 321.593071 | 355.299451 | 1222.103049 | 3.117990 | no | `computed` |
| 17 | 43 | 23 | -156.060344 | 332.120687 | 365.827068 | 3375.447265 | 1.843938 | no | `computed` |
| 18 | 43 | 32 | -156.258443 | 332.516886 | 366.223267 | 1498.324789 | 22.990855 | no | `computed` |
| 19 | 17 | 25 | -157.980768 | 335.961537 | 369.667917 | 1719.322902 | 5.254270 | no | `computed` |
| 20 | 27 | 17 | -158.835804 | 337.671609 | 371.377989 | 752438.582007 | 26826.806675 | no | `computed` |

### Rank 1: Seed 33, Draw 32

- LogLik: `250.490969`; AIC: `-480.981939`; BIC: `-447.275558`
- Max shape path: `333.583931`; max implied variance: `11.278357`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(5.487320)}{0.166410} + \underset{(17.532256)}{0.176424}\,\pi_t + \underset{(28.348040)}{0.694792}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.223743)}{0.450682}\,\omega_{p,t} - \underset{(0.000014)}{0.041885}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.270248)}{1.417208} + \underset{(0.000005)}{0.943264}\,p_{t-1} + \frac{\underset{(0.443328)}{0.049721}}{2(\underset{(0.223743)}{0.450682})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.155421)}{0.010220}}{2(\underset{(0.223743)}{0.450682})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(8.269118)}{7.985130} + \underset{(0.000005)}{0.943264}\,n_{t-1} + \frac{\underset{(0.443328)}{0.049721}}{2(\underset{(0.000014)}{0.041885})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.155421)}{0.010220}}{2(\underset{(0.000014)}{0.041885})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.166410 | 5.487320 |
| rho_1 | 0.176424 | 17.532256 |
| phi_1 | 0.694792 | 28.348040 |
| p0 | 1.417208 | 0.270248 |
| n0 | 7.985130 | 8.269118 |
| rho | 0.943264 | 0.000005 |
| phi_plus | 0.049721 | 0.443328 |
| phi_minus | 0.010220 | 0.155421 |
| sigma_p | 0.450682 | 0.223743 |
| sigma_n | 0.041885 | 0.000014 |

### Rank 2: Seed 20, Draw 5

- LogLik: `-2.584402`; AIC: `25.168805`; BIC: `58.875185`
- Max shape path: `1690.545280`; max implied variance: `4.211158`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.239741)}{-0.437768} + \underset{(0.058984)}{0.082676}\,\pi_t + \underset{(0.122867)}{1.493895}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.721428)}{0.303172}\,\omega_{p,t} - \underset{(0.000249)}{0.027871}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(14.198452)}{6.736192} + \underset{(0.074522)}{0.613442}\,p_{t-1} + \frac{\underset{(0.109957)}{0.252264}}{2(\underset{(0.721428)}{0.303172})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.238306)}{0.152797}}{2(\underset{(0.721428)}{0.303172})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(36.574012)}{8.456699} + \underset{(0.074522)}{0.613442}\,n_{t-1} + \frac{\underset{(0.109957)}{0.252264}}{2(\underset{(0.000249)}{0.027871})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.238306)}{0.152797}}{2(\underset{(0.000249)}{0.027871})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.437768 | 0.239741 |
| rho_1 | 0.082676 | 0.058984 |
| phi_1 | 1.493895 | 0.122867 |
| p0 | 6.736192 | 14.198452 |
| n0 | 8.456699 | 36.574012 |
| rho | 0.613442 | 0.074522 |
| phi_plus | 0.252264 | 0.109957 |
| phi_minus | 0.152797 | 0.238306 |
| sigma_p | 0.303172 | 0.721428 |
| sigma_n | 0.027871 | 0.000249 |

### Rank 3: Seed 42, Draw 4

- LogLik: `-16.634979`; AIC: `53.269958`; BIC: `86.976338`
- Max shape path: `3803.017359`; max implied variance: `4.543585`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(552.413967)}{-0.246654} + \underset{(88.937016)}{0.409457}\,\pi_t + \underset{(546.602459)}{1.075944}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000065)}{0.023935}\,\omega_{p,t} - \underset{(1.166879)}{0.134657}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(17.262885)}{6.292586} + \underset{(0.000016)}{0.807807}\,p_{t-1} + \frac{\underset{(0.005533)}{0.088379}}{2(\underset{(0.000065)}{0.023935})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.005758)}{0.243758}}{2(\underset{(0.000065)}{0.023935})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(2.956304)}{2.170878} + \underset{(0.000016)}{0.807807}\,n_{t-1} + \frac{\underset{(0.005533)}{0.088379}}{2(\underset{(1.166879)}{0.134657})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.005758)}{0.243758}}{2(\underset{(1.166879)}{0.134657})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.246654 | 552.413967 |
| rho_1 | 0.409457 | 88.937016 |
| phi_1 | 1.075944 | 546.602459 |
| p0 | 6.292586 | 17.262885 |
| n0 | 2.170878 | 2.956304 |
| rho | 0.807807 | 0.000016 |
| phi_plus | 0.088379 | 0.005533 |
| phi_minus | 0.243758 | 0.005758 |
| sigma_p | 0.023935 | 0.000065 |
| sigma_n | 0.134657 | 1.166879 |

### Rank 4: Seed 43, Draw 9

- LogLik: `-40.183729`; AIC: `100.367458`; BIC: `134.073838`
- Max shape path: `10624.346451`; max implied variance: `3.317568`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000027)}{0.125728} + \underset{(0.000002)}{0.195998}\,\pi_t + \underset{(0.000024)}{0.578810}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000768)}{0.011933}\,\omega_{p,t} - \underset{(0.002653)}{0.107328}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(38.382162)}{9.521635} + \underset{(0.002645)}{0.802419}\,p_{t-1} + \frac{\underset{(0.000010)}{0.297975}}{2(\underset{(0.000768)}{0.011933})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000001)}{0.000000}}{2(\underset{(0.000768)}{0.011933})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(66.237377)}{5.125495} + \underset{(0.002645)}{0.802419}\,n_{t-1} + \frac{\underset{(0.000010)}{0.297975}}{2(\underset{(0.002653)}{0.107328})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000001)}{0.000000}}{2(\underset{(0.002653)}{0.107328})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.125728 | 0.000027 |
| rho_1 | 0.195998 | 0.000002 |
| phi_1 | 0.578810 | 0.000024 |
| p0 | 9.521635 | 38.382162 |
| n0 | 5.125495 | 66.237377 |
| rho | 0.802419 | 0.002645 |
| phi_plus | 0.297975 | 0.000010 |
| phi_minus | 0.000000 | 0.000001 |
| sigma_p | 0.011933 | 0.000768 |
| sigma_n | 0.107328 | 0.002653 |

### Rank 5: Seed 8, Draw 9

- LogLik: `-94.216877`; AIC: `208.433754`; BIC: `242.140134`
- Max shape path: `29093.994442`; max implied variance: `7.121775`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(4.034333)}{0.297414} + \underset{(17.909277)}{0.244068}\,\pi_t + \underset{(16.375663)}{0.533529}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000032)}{0.010810}\,\omega_{p,t} - \underset{(0.077807)}{0.137945}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(133.333314)}{6.642971} + \underset{(2.518750)}{0.637719}\,p_{t-1} + \frac{\underset{(1.103147)}{0.315531}}{2(\underset{(0.000032)}{0.010810})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(3.751198)}{0.374156}}{2(\underset{(0.000032)}{0.010810})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(43.281980)}{6.183895} + \underset{(2.518750)}{0.637719}\,n_{t-1} + \frac{\underset{(1.103147)}{0.315531}}{2(\underset{(0.077807)}{0.137945})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(3.751198)}{0.374156}}{2(\underset{(0.077807)}{0.137945})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.297414 | 4.034333 |
| rho_1 | 0.244068 | 17.909277 |
| phi_1 | 0.533529 | 16.375663 |
| p0 | 6.642971 | 133.333314 |
| n0 | 6.183895 | 43.281980 |
| rho | 0.637719 | 2.518750 |
| phi_plus | 0.315531 | 1.103147 |
| phi_minus | 0.374156 | 3.751198 |
| sigma_p | 0.010810 | 0.000032 |
| sigma_n | 0.137945 | 0.077807 |

### Rank 6: Seed 40, Draw 35

- LogLik: `-103.729216`; AIC: `227.458432`; BIC: `261.164812`
- Max shape path: `31113.806343`; max implied variance: `3.927631`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.823737)}{0.074059} + \underset{(1.698029)}{0.203251}\,\pi_t + \underset{(2.009566)}{0.817088}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.102946)}{0.119447}\,\omega_{p,t} - \underset{(0.000041)}{0.007755}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(6.675442)}{7.321162} + \underset{(0.748048)}{0.437730}\,p_{t-1} + \frac{\underset{(1.299076)}{0.865283}}{2(\underset{(0.102946)}{0.119447})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.742321)}{0.161371}}{2(\underset{(0.102946)}{0.119447})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(78.289715)}{4.255975} + \underset{(0.748048)}{0.437730}\,n_{t-1} + \frac{\underset{(1.299076)}{0.865283}}{2(\underset{(0.000041)}{0.007755})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.742321)}{0.161371}}{2(\underset{(0.000041)}{0.007755})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.074059 | 0.823737 |
| rho_1 | 0.203251 | 1.698029 |
| phi_1 | 0.817088 | 2.009566 |
| p0 | 7.321162 | 6.675442 |
| n0 | 4.255975 | 78.289715 |
| rho | 0.437730 | 0.748048 |
| phi_plus | 0.865283 | 1.299076 |
| phi_minus | 0.161371 | 0.742321 |
| sigma_p | 0.119447 | 0.102946 |
| sigma_n | 0.007755 | 0.000041 |

### Rank 7: Seed 47, Draw 31

- LogLik: `-109.917679`; AIC: `239.835359`; BIC: `273.541739`
- Max shape path: `2220.085146`; max implied variance: `4.273532`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(368.623500)}{0.115597} + \underset{(332.740760)}{0.235815}\,\pi_t + \underset{(111.038366)}{0.586525}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001485)}{0.030487}\,\omega_{p,t} - \underset{(0.000012)}{0.077489}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(10711.688761)}{0.391148} + \underset{(1.460180)}{0.752190}\,p_{t-1} + \frac{\underset{(1.457980)}{0.149529}}{2(\underset{(0.001485)}{0.030487})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(147.291553)}{0.239734}}{2(\underset{(0.001485)}{0.030487})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(1630.189944)}{6.112799} + \underset{(1.460180)}{0.752190}\,n_{t-1} + \frac{\underset{(1.457980)}{0.149529}}{2(\underset{(0.000012)}{0.077489})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(147.291553)}{0.239734}}{2(\underset{(0.000012)}{0.077489})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.115597 | 368.623500 |
| rho_1 | 0.235815 | 332.740760 |
| phi_1 | 0.586525 | 111.038366 |
| p0 | 0.391148 | 10711.688761 |
| n0 | 6.112799 | 1630.189944 |
| rho | 0.752190 | 1.460180 |
| phi_plus | 0.149529 | 1.457980 |
| phi_minus | 0.239734 | 147.291553 |
| sigma_p | 0.030487 | 0.001485 |
| sigma_n | 0.077489 | 0.000012 |

### Rank 8: Seed 33, Draw 1

- LogLik: `-110.700824`; AIC: `241.401648`; BIC: `275.108028`
- Max shape path: `950.488040`; max implied variance: `49.639436`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(793.090968)}{0.095584} + \underset{(2223.761342)}{0.385110}\,\pi_t + \underset{(2545.519538)}{0.781777}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(215.797815)}{0.298706}\,\omega_{p,t} - \underset{(27.880292)}{0.079193}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(18.885443)}{1.852064} + \underset{(0.008012)}{0.540883}\,p_{t-1} + \frac{\underset{(3.322562)}{0.514627}}{2(\underset{(215.797815)}{0.298706})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(3.338504)}{0.396039}}{2(\underset{(215.797815)}{0.298706})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(163.272409)}{3.596029} + \underset{(0.008012)}{0.540883}\,n_{t-1} + \frac{\underset{(3.322562)}{0.514627}}{2(\underset{(27.880292)}{0.079193})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(3.338504)}{0.396039}}{2(\underset{(27.880292)}{0.079193})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.095584 | 793.090968 |
| rho_1 | 0.385110 | 2223.761342 |
| phi_1 | 0.781777 | 2545.519538 |
| p0 | 1.852064 | 18.885443 |
| n0 | 3.596029 | 163.272409 |
| rho | 0.540883 | 0.008012 |
| phi_plus | 0.514627 | 3.322562 |
| phi_minus | 0.396039 | 3.338504 |
| sigma_p | 0.298706 | 215.797815 |
| sigma_n | 0.079193 | 27.880292 |

### Rank 9: Seed 20, Draw 30

- LogLik: `-113.652118`; AIC: `247.304236`; BIC: `281.010616`
- Max shape path: `1418.023476`; max implied variance: `11.761217`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.395060)}{0.143062} + \underset{(0.261224)}{0.187840}\,\pi_t + \underset{(0.503076)}{0.688160}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.071714)}{0.063970}\,\omega_{p,t} - \underset{(0.255578)}{0.267925}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.034050)}{4.276481} + \underset{(0.000005)}{0.232058}\,p_{t-1} + \frac{\underset{(1.127216)}{0.791153}}{2(\underset{(0.071714)}{0.063970})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.127216)}{0.686343}}{2(\underset{(0.071714)}{0.063970})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.034056)}{1.909375} + \underset{(0.000005)}{0.232058}\,n_{t-1} + \frac{\underset{(1.127216)}{0.791153}}{2(\underset{(0.255578)}{0.267925})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(1.127216)}{0.686343}}{2(\underset{(0.255578)}{0.267925})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.143062 | 0.395060 |
| rho_1 | 0.187840 | 0.261224 |
| phi_1 | 0.688160 | 0.503076 |
| p0 | 4.276481 | 0.034050 |
| n0 | 1.909375 | 0.034056 |
| rho | 0.232058 | 0.000005 |
| phi_plus | 0.791153 | 1.127216 |
| phi_minus | 0.686343 | 1.127216 |
| sigma_p | 0.063970 | 0.071714 |
| sigma_n | 0.267925 | 0.255578 |

### Rank 10: Seed 26, Draw 3

- LogLik: `-126.373740`; AIC: `272.747480`; BIC: `306.453861`
- Max shape path: `990.074774`; max implied variance: `35.411762`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.077710} + \underset{(0.004734)}{0.216376}\,\pi_t + \underset{(0.000111)}{0.405685}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000588)}{0.037903}\,\omega_{p,t} - \underset{(0.879113)}{0.448782}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.952061)}{8.886394} + \underset{(0.000902)}{0.846857}\,p_{t-1} + \frac{\underset{(0.000126)}{0.162692}}{2(\underset{(0.000588)}{0.037903})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000113)}{0.104602}}{2(\underset{(0.000588)}{0.037903})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.977246)}{3.364610} + \underset{(0.000902)}{0.846857}\,n_{t-1} + \frac{\underset{(0.000126)}{0.162692}}{2(\underset{(0.879113)}{0.448782})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000113)}{0.104602}}{2(\underset{(0.879113)}{0.448782})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.077710 | 0.000002 |
| rho_1 | 0.216376 | 0.004734 |
| phi_1 | 0.405685 | 0.000111 |
| p0 | 8.886394 | 0.952061 |
| n0 | 3.364610 | 0.977246 |
| rho | 0.846857 | 0.000902 |
| phi_plus | 0.162692 | 0.000126 |
| phi_minus | 0.104602 | 0.000113 |
| sigma_p | 0.037903 | 0.000588 |
| sigma_n | 0.448782 | 0.879113 |

### Rank 11: Seed 15, Draw 37

- LogLik: `-131.051663`; AIC: `282.103326`; BIC: `315.809706`
- Max shape path: `583.241014`; max implied variance: `19.491296`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1019.352872)}{0.122462} + \underset{(1551.073528)}{0.376972}\,\pi_t + \underset{(1468.112138)}{0.812427}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(316.165618)}{0.483107}\,\omega_{p,t} - \underset{(34.001738)}{0.065884}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(8.783395)}{2.314242} + \underset{(0.004136)}{0.511077}\,p_{t-1} + \frac{\underset{(118.463480)}{0.662432}}{2(\underset{(316.165618)}{0.483107})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(118.456136)}{0.255648}}{2(\underset{(316.165618)}{0.483107})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(50.315770)}{9.755830} + \underset{(0.004136)}{0.511077}\,n_{t-1} + \frac{\underset{(118.463480)}{0.662432}}{2(\underset{(34.001738)}{0.065884})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(118.456136)}{0.255648}}{2(\underset{(34.001738)}{0.065884})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.122462 | 1019.352872 |
| rho_1 | 0.376972 | 1551.073528 |
| phi_1 | 0.812427 | 1468.112138 |
| p0 | 2.314242 | 8.783395 |
| n0 | 9.755830 | 50.315770 |
| rho | 0.511077 | 0.004136 |
| phi_plus | 0.662432 | 118.463480 |
| phi_minus | 0.255648 | 118.456136 |
| sigma_p | 0.483107 | 316.165618 |
| sigma_n | 0.065884 | 34.001738 |

### Rank 12: Seed 11, Draw 21

- LogLik: `-134.377082`; AIC: `288.754164`; BIC: `322.460544`
- Max shape path: `791.922237`; max implied variance: `34.196184`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(58.411070)}{0.056975} + \underset{(59.990466)}{0.302110}\,\pi_t + \underset{(97.946086)}{0.643917}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(5.657056)}{0.094022}\,\omega_{p,t} - \underset{(16.528523)}{0.269771}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(99.119557)}{6.676627} + \underset{(0.000480)}{0.257479}\,p_{t-1} + \frac{\underset{(21.479499)}{0.623867}}{2(\underset{(5.657056)}{0.094022})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(21.479662)}{0.837271}}{2(\underset{(5.657056)}{0.094022})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(11.874010)}{4.804783} + \underset{(0.000480)}{0.257479}\,n_{t-1} + \frac{\underset{(21.479499)}{0.623867}}{2(\underset{(16.528523)}{0.269771})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(21.479662)}{0.837271}}{2(\underset{(16.528523)}{0.269771})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056975 | 58.411070 |
| rho_1 | 0.302110 | 59.990466 |
| phi_1 | 0.643917 | 97.946086 |
| p0 | 6.676627 | 99.119557 |
| n0 | 4.804783 | 11.874010 |
| rho | 0.257479 | 0.000480 |
| phi_plus | 0.623867 | 21.479499 |
| phi_minus | 0.837271 | 21.479662 |
| sigma_p | 0.094022 | 5.657056 |
| sigma_n | 0.269771 | 16.528523 |

### Rank 13: Seed 38, Draw 26

- LogLik: `-138.229694`; AIC: `296.459388`; BIC: `330.165769`
- Max shape path: `1068.681196`; max implied variance: `1.266514`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.033254)}{0.017199} + \underset{(0.023905)}{0.227890}\,\pi_t + \underset{(0.083035)}{0.913425}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000008)}{0.023202}\,\omega_{p,t} - \underset{(0.000564)}{0.061321}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.058132)}{6.077891} + \underset{(0.000021)}{0.860440}\,p_{t-1} + \frac{\underset{(0.000589)}{0.176376}}{2(\underset{(0.000008)}{0.023202})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000614)}{0.002381}}{2(\underset{(0.000008)}{0.023202})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.007459)}{5.170331} + \underset{(0.000021)}{0.860440}\,n_{t-1} + \frac{\underset{(0.000589)}{0.176376}}{2(\underset{(0.000564)}{0.061321})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000614)}{0.002381}}{2(\underset{(0.000564)}{0.061321})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.017199 | 0.033254 |
| rho_1 | 0.227890 | 0.023905 |
| phi_1 | 0.913425 | 0.083035 |
| p0 | 6.077891 | 0.058132 |
| n0 | 5.170331 | 0.007459 |
| rho | 0.860440 | 0.000021 |
| phi_plus | 0.176376 | 0.000589 |
| phi_minus | 0.002381 | 0.000614 |
| sigma_p | 0.023202 | 0.000008 |
| sigma_n | 0.061321 | 0.000564 |

### Rank 14: Seed 7, Draw 20

- LogLik: `-142.606992`; AIC: `305.213984`; BIC: `338.920364`
- Max shape path: `15080.298303`; max implied variance: `3.965045`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000004)}{0.311365} + \underset{(0.000004)}{0.102319}\,\pi_t + \underset{(0.000006)}{0.664216}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.147600}\,\omega_{p,t} - \underset{(0.000049)}{0.010998}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(110.383557)}{8.678840} + \underset{(8.955796)}{0.402947}\,p_{t-1} + \frac{\underset{(159.162730)}{0.620485}}{2(\underset{(0.000002)}{0.147600})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000002)}{0.200105}}{2(\underset{(0.000002)}{0.147600})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(891.183086)}{0.130357} + \underset{(8.955796)}{0.402947}\,n_{t-1} + \frac{\underset{(159.162730)}{0.620485}}{2(\underset{(0.000049)}{0.010998})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000002)}{0.200105}}{2(\underset{(0.000049)}{0.010998})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.311365 | 0.000004 |
| rho_1 | 0.102319 | 0.000004 |
| phi_1 | 0.664216 | 0.000006 |
| p0 | 8.678840 | 110.383557 |
| n0 | 0.130357 | 891.183086 |
| rho | 0.402947 | 8.955796 |
| phi_plus | 0.620485 | 159.162730 |
| phi_minus | 0.200105 | 0.000002 |
| sigma_p | 0.147600 | 0.000002 |
| sigma_n | 0.010998 | 0.000049 |

### Rank 15: Seed 32, Draw 28

- LogLik: `-144.250493`; AIC: `308.500985`; BIC: `342.207365`
- Max shape path: `232.568370`; max implied variance: `59.355582`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.378214)}{0.243238} + \underset{(0.865307)}{0.131004}\,\pi_t + \underset{(0.882099)}{0.945519}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.294890)}{0.678907}\,\omega_{p,t} - \underset{(0.386097)}{0.182702}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.002328)}{0.305326} + \underset{(0.000019)}{0.681758}\,p_{t-1} + \frac{\underset{(0.447267)}{0.446660}}{2(\underset{(1.294890)}{0.678907})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.447277)}{0.184370}}{2(\underset{(1.294890)}{0.678907})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.000697)}{0.634379} + \underset{(0.000019)}{0.681758}\,n_{t-1} + \frac{\underset{(0.447267)}{0.446660}}{2(\underset{(0.386097)}{0.182702})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.447277)}{0.184370}}{2(\underset{(0.386097)}{0.182702})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.243238 | 0.378214 |
| rho_1 | 0.131004 | 0.865307 |
| phi_1 | 0.945519 | 0.882099 |
| p0 | 0.305326 | 0.002328 |
| n0 | 0.634379 | 0.000697 |
| rho | 0.681758 | 0.000019 |
| phi_plus | 0.446660 | 0.447267 |
| phi_minus | 0.184370 | 0.447277 |
| sigma_p | 0.678907 | 1.294890 |
| sigma_n | 0.182702 | 0.386097 |

### Rank 16: Seed 22, Draw 10

- LogLik: `-150.796535`; AIC: `321.593071`; BIC: `355.299451`
- Max shape path: `1222.103049`; max implied variance: `3.117990`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000001)}{0.317056} + \underset{(0.000000)}{0.146297}\,\pi_t + \underset{(0.000000)}{0.587983}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.034346}\,\omega_{p,t} - \underset{(0.000002)}{0.127753}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000000)}{7.229358} + \underset{(0.000001)}{0.534329}\,p_{t-1} + \frac{\underset{(0.000000)}{0.433031}}{2(\underset{(0.000002)}{0.034346})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000001)}{0.019709}}{2(\underset{(0.000002)}{0.034346})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.002920)}{7.217665} + \underset{(0.000001)}{0.534329}\,n_{t-1} + \frac{\underset{(0.000000)}{0.433031}}{2(\underset{(0.000002)}{0.127753})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000001)}{0.019709}}{2(\underset{(0.000002)}{0.127753})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.317056 | 0.000001 |
| rho_1 | 0.146297 | 0.000000 |
| phi_1 | 0.587983 | 0.000000 |
| p0 | 7.229358 | 0.000000 |
| n0 | 7.217665 | 0.002920 |
| rho | 0.534329 | 0.000001 |
| phi_plus | 0.433031 | 0.000000 |
| phi_minus | 0.019709 | 0.000001 |
| sigma_p | 0.034346 | 0.000002 |
| sigma_n | 0.127753 | 0.000002 |

### Rank 17: Seed 43, Draw 23

- LogLik: `-156.060344`; AIC: `332.120687`; BIC: `365.827068`
- Max shape path: `3375.447265`; max implied variance: `1.843938`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(9.545444)}{0.016108} + \underset{(10.642839)}{0.222905}\,\pi_t + \underset{(57.701868)}{0.848418}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000001)}{0.016081}\,\omega_{p,t} - \underset{(3.932737)}{0.059224}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1114.636451)}{10.000000} + \underset{(0.002664)}{0.797477}\,p_{t-1} + \frac{\underset{(7.580777)}{0.297692}}{2(\underset{(0.000001)}{0.016081})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.963138)}{0.000000}}{2(\underset{(0.000001)}{0.016081})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(63.553848)}{6.404265} + \underset{(0.002664)}{0.797477}\,n_{t-1} + \frac{\underset{(7.580777)}{0.297692}}{2(\underset{(3.932737)}{0.059224})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.963138)}{0.000000}}{2(\underset{(3.932737)}{0.059224})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.016108 | 9.545444 |
| rho_1 | 0.222905 | 10.642839 |
| phi_1 | 0.848418 | 57.701868 |
| p0 | 10.000000 | 1114.636451 |
| n0 | 6.404265 | 63.553848 |
| rho | 0.797477 | 0.002664 |
| phi_plus | 0.297692 | 7.580777 |
| phi_minus | 0.000000 | 2.963138 |
| sigma_p | 0.016081 | 0.000001 |
| sigma_n | 0.059224 | 3.932737 |

### Rank 18: Seed 43, Draw 32

- LogLik: `-156.258443`; AIC: `332.516886`; BIC: `366.223267`
- Max shape path: `1498.324789`; max implied variance: `22.990855`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(78.541523)}{0.624755} + \underset{(19.950499)}{-0.022209}\,\pi_t + \underset{(58.594872)}{-0.087380}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(10.120562)}{0.203289}\,\omega_{p,t} - \underset{(0.000024)}{0.070993}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(79.729132)}{4.915233} + \underset{(0.000006)}{0.751658}\,p_{t-1} + \frac{\underset{(0.000433)}{0.451474}}{2(\underset{(10.120562)}{0.203289})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.002195)}{0.025981}}{2(\underset{(10.120562)}{0.203289})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(119.647889)}{3.556328} + \underset{(0.000006)}{0.751658}\,n_{t-1} + \frac{\underset{(0.000433)}{0.451474}}{2(\underset{(0.000024)}{0.070993})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.002195)}{0.025981}}{2(\underset{(0.000024)}{0.070993})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.624755 | 78.541523 |
| rho_1 | -0.022209 | 19.950499 |
| phi_1 | -0.087380 | 58.594872 |
| p0 | 4.915233 | 79.729132 |
| n0 | 3.556328 | 119.647889 |
| rho | 0.751658 | 0.000006 |
| phi_plus | 0.451474 | 0.000433 |
| phi_minus | 0.025981 | 0.002195 |
| sigma_p | 0.203289 | 10.120562 |
| sigma_n | 0.070993 | 0.000024 |

### Rank 19: Seed 17, Draw 25

- LogLik: `-157.980768`; AIC: `335.961537`; BIC: `369.667917`
- Max shape path: `1719.322902`; max implied variance: `5.254270`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000016)}{0.088276} + \underset{(0.000530)}{0.260865}\,\pi_t + \underset{(0.000000)}{0.751949}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000001)}{0.038541}\,\omega_{p,t} - \underset{(0.000002)}{0.109950}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.019054)}{2.612776} + \underset{(0.000002)}{0.372682}\,p_{t-1} + \frac{\underset{(0.000016)}{0.650864}}{2(\underset{(0.000001)}{0.038541})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000530)}{0.291800}}{2(\underset{(0.000001)}{0.038541})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.016297)}{7.925652} + \underset{(0.000002)}{0.372682}\,n_{t-1} + \frac{\underset{(0.000016)}{0.650864}}{2(\underset{(0.000002)}{0.109950})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000530)}{0.291800}}{2(\underset{(0.000002)}{0.109950})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.088276 | 0.000016 |
| rho_1 | 0.260865 | 0.000530 |
| phi_1 | 0.751949 | 0.000000 |
| p0 | 2.612776 | 0.019054 |
| n0 | 7.925652 | 0.016297 |
| rho | 0.372682 | 0.000002 |
| phi_plus | 0.650864 | 0.000016 |
| phi_minus | 0.291800 | 0.000530 |
| sigma_p | 0.038541 | 0.000001 |
| sigma_n | 0.109950 | 0.000002 |

### Rank 20: Seed 27, Draw 17

- LogLik: `-158.835804`; AIC: `337.671609`; BIC: `371.377989`
- Max shape path: `752438.582007`; max implied variance: `26826.806675`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1.317824)}{-0.057638} + \underset{(2.907729)}{0.345429}\,\pi_t + \underset{(12.159811)}{0.722459}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(4.920634)}{0.248407}\,\omega_{p,t} - \underset{(1.246515)}{0.067986}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.120472)}{0.378391} + \underset{(0.000000)}{0.547098}\,p_{t-1} + \frac{\underset{(0.092003)}{0.700953}}{2(\underset{(4.920634)}{0.248407})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.091937)}{0.204849}}{2(\underset{(4.920634)}{0.248407})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(1.339321)}{0.752439} + \underset{(0.000000)}{0.547098}\,n_{t-1} + \frac{\underset{(0.092003)}{0.700953}}{2(\underset{(1.246515)}{0.067986})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.091937)}{0.204849}}{2(\underset{(1.246515)}{0.067986})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.057638 | 1.317824 |
| rho_1 | 0.345429 | 2.907729 |
| phi_1 | 0.722459 | 12.159811 |
| p0 | 0.378391 | 0.120472 |
| n0 | 0.752439 | 1.339321 |
| rho | 0.547098 | 0.000000 |
| phi_plus | 0.700953 | 0.092003 |
| phi_minus | 0.204849 | 0.091937 |
| sigma_p | 0.248407 | 4.920634 |
| sigma_n | 0.067986 | 1.246515 |

## ARX(2,1)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 21 | 40 | 253.476194 | -484.952389 | -447.875370 | 271.160360 | 23.471739 | yes | `computed` |
| 2 | 21 | 39 | -20.703249 | 63.406498 | 100.483516 | 4013.468681 | 1.246669 | yes | `computed` |
| 3 | 43 | 25 | -89.071219 | 200.142438 | 237.219456 | 2305.724576 | 5.611361 | yes | `computed` |
| 4 | 27 | 8 | -99.205068 | 220.410136 | 257.487154 | 428.778316 | 2.364721 | yes | `computed` |
| 5 | 18 | 32 | -108.208535 | 238.417070 | 275.494089 | 4468.230081 | 4.578289 | yes | `computed` |
| 6 | 29 | 7 | -110.828684 | 243.657368 | 280.734386 | 58035.160089 | 14.717785 | yes | `computed` |
| 7 | 8 | 26 | -123.753325 | 269.506650 | 306.583668 | 351.550369 | 7.498781 | yes | `computed` |
| 8 | 42 | 23 | -133.015952 | 288.031903 | 325.108922 | 1253.812395 | 9.908656 | yes | `computed` |
| 9 | 1 | 27 | -137.090582 | 296.181165 | 333.258183 | 3169.108831 | 7.169017 | yes | `computed` |
| 10 | 41 | 32 | -144.233056 | 310.466112 | 347.543130 | 179.315781 | 76.484555 | yes | `computed` |
| 11 | 27 | 37 | -147.617926 | 317.235852 | 354.312870 | 28505.541845 | 4.216636 | yes | `computed` |
| 12 | 6 | 7 | -149.098987 | 320.197975 | 357.274993 | 396.943733 | 1.411474 | yes | `computed` |
| 13 | 23 | 4 | -149.915059 | 321.830118 | 358.907137 | 6869.146863 | 8.220368 | yes | `computed` |
| 14 | 18 | 13 | -157.390206 | 336.780412 | 373.857430 | 358.485927 | 1.328582 | no | `computed` |
| 15 | 22 | 2 | -157.819369 | 337.638738 | 374.715756 | 8315.894717 | 4.433770 | no | `computed` |
| 16 | 14 | 33 | -159.451118 | 340.902235 | 377.979254 | 4196.116535 | 2.436154 | no | `computed` |
| 17 | 29 | 36 | -165.415635 | 352.831270 | 389.908288 | 3862.496623 | 4.627832 | no | `computed` |
| 18 | 6 | 28 | -167.158491 | 356.316983 | 393.394001 | 788.761257 | 1.240715 | no | `computed` |
| 19 | 9 | 29 | -167.964823 | 357.929645 | 395.006663 | 511.306262 | 6.072363 | no | `computed` |
| 20 | 13 | 40 | -168.212202 | 358.424403 | 395.501421 | 8.001020 | 2.650826 | no | `computed` |

### Rank 1: Seed 21, Draw 40

- LogLik: `253.476194`; AIC: `-484.952389`; BIC: `-447.875370`
- Max shape path: `271.160360`; max implied variance: `23.471739`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(43.835665)}{-0.027867} + \underset{(0.009603)}{0.408254}\,\pi_t + \underset{(35.235111)}{-0.033101}\,\pi_{t-1} + \underset{(49.672520)}{0.665318}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(2.206097)}{0.076122}\,\omega_{p,t} - \underset{(8.026198)}{0.495743}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(86.681238)}{8.161163} + \underset{(0.000001)}{0.939951}\,p_{t-1} + \frac{\underset{(0.142799)}{0.050597}}{2(\underset{(2.206097)}{0.076122})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.062691)}{0.009308}}{2(\underset{(2.206097)}{0.076122})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(11.136273)}{2.682046} + \underset{(0.000001)}{0.939951}\,n_{t-1} + \frac{\underset{(0.142799)}{0.050597}}{2(\underset{(8.026198)}{0.495743})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.062691)}{0.009308}}{2(\underset{(8.026198)}{0.495743})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.027867 | 43.835665 |
| rho_1 | 0.408254 | 0.009603 |
| rho_2 | -0.033101 | 35.235111 |
| phi_1 | 0.665318 | 49.672520 |
| p0 | 8.161163 | 86.681238 |
| n0 | 2.682046 | 11.136273 |
| rho | 0.939951 | 0.000001 |
| phi_plus | 0.050597 | 0.142799 |
| phi_minus | 0.009308 | 2.062691 |
| sigma_p | 0.076122 | 2.206097 |
| sigma_n | 0.495743 | 8.026198 |

### Rank 2: Seed 21, Draw 39

- LogLik: `-20.703249`; AIC: `63.406498`; BIC: `100.483516`
- Max shape path: `4013.468681`; max implied variance: `1.246669`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(19.683454)}{0.007187} + \underset{(74.732822)}{0.227721}\,\pi_t + \underset{(50.612646)}{0.072011}\,\pi_{t-1} + \underset{(8.200904)}{0.807644}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000032)}{0.011845}\,\omega_{p,t} - \underset{(1.136664)}{0.046743}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(498.423533)}{3.073296} + \underset{(0.154537)}{0.874843}\,p_{t-1} + \frac{\underset{(4.093747)}{0.180942}}{2(\underset{(0.000032)}{0.011845})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.942686)}{0.006684}}{2(\underset{(0.000032)}{0.011845})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(100.262954)}{7.094145} + \underset{(0.154537)}{0.874843}\,n_{t-1} + \frac{\underset{(4.093747)}{0.180942}}{2(\underset{(1.136664)}{0.046743})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.942686)}{0.006684}}{2(\underset{(1.136664)}{0.046743})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.007187 | 19.683454 |
| rho_1 | 0.227721 | 74.732822 |
| rho_2 | 0.072011 | 50.612646 |
| phi_1 | 0.807644 | 8.200904 |
| p0 | 3.073296 | 498.423533 |
| n0 | 7.094145 | 100.262954 |
| rho | 0.874843 | 0.154537 |
| phi_plus | 0.180942 | 4.093747 |
| phi_minus | 0.006684 | 0.942686 |
| sigma_p | 0.011845 | 0.000032 |
| sigma_n | 0.046743 | 1.136664 |

### Rank 3: Seed 43, Draw 25

- LogLik: `-89.071219`; AIC: `200.142438`; BIC: `237.219456`
- Max shape path: `2305.724576`; max implied variance: `5.611361`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.054429)}{-0.225251} + \underset{(0.024607)}{0.135374}\,\pi_t + \underset{(0.005743)}{-0.084017}\,\pi_{t-1} + \underset{(0.023505)}{0.688499}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000003)}{0.029275}\,\omega_{p,t} - \underset{(0.007875)}{0.266324}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(2.204377)}{4.396554} + \underset{(0.008240)}{0.756702}\,p_{t-1} + \frac{\underset{(0.000002)}{0.033864}}{2(\underset{(0.000003)}{0.029275})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.034988)}{0.305846}}{2(\underset{(0.000003)}{0.029275})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.206514)}{5.744938} + \underset{(0.008240)}{0.756702}\,n_{t-1} + \frac{\underset{(0.000002)}{0.033864}}{2(\underset{(0.007875)}{0.266324})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.034988)}{0.305846}}{2(\underset{(0.007875)}{0.266324})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.225251 | 0.054429 |
| rho_1 | 0.135374 | 0.024607 |
| rho_2 | -0.084017 | 0.005743 |
| phi_1 | 0.688499 | 0.023505 |
| p0 | 4.396554 | 2.204377 |
| n0 | 5.744938 | 0.206514 |
| rho | 0.756702 | 0.008240 |
| phi_plus | 0.033864 | 0.000002 |
| phi_minus | 0.305846 | 0.034988 |
| sigma_p | 0.029275 | 0.000003 |
| sigma_n | 0.266324 | 0.007875 |

### Rank 4: Seed 27, Draw 8

- LogLik: `-99.205068`; AIC: `220.410136`; BIC: `257.487154`
- Max shape path: `428.778316`; max implied variance: `2.364721`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(176.901183)}{0.071479} + \underset{(229.050283)}{0.141478}\,\pi_t + \underset{(447.640166)}{0.145558}\,\pi_{t-1} + \underset{(463.476290)}{0.294803}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(15.146565)}{0.240745}\,\omega_{p,t} - \underset{(0.007017)}{0.030387}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(326.906957)}{6.316360} + \underset{(7.862294)}{0.771484}\,p_{t-1} + \frac{\underset{(0.008506)}{0.051195}}{2(\underset{(15.146565)}{0.240745})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(64.674100)}{0.004422}}{2(\underset{(15.146565)}{0.240745})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(3153.404412)}{7.203270} + \underset{(7.862294)}{0.771484}\,n_{t-1} + \frac{\underset{(0.008506)}{0.051195}}{2(\underset{(0.007017)}{0.030387})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(64.674100)}{0.004422}}{2(\underset{(0.007017)}{0.030387})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.071479 | 176.901183 |
| rho_1 | 0.141478 | 229.050283 |
| rho_2 | 0.145558 | 447.640166 |
| phi_1 | 0.294803 | 463.476290 |
| p0 | 6.316360 | 326.906957 |
| n0 | 7.203270 | 3153.404412 |
| rho | 0.771484 | 7.862294 |
| phi_plus | 0.051195 | 0.008506 |
| phi_minus | 0.004422 | 64.674100 |
| sigma_p | 0.240745 | 15.146565 |
| sigma_n | 0.030387 | 0.007017 |

### Rank 5: Seed 18, Draw 32

- LogLik: `-108.208535`; AIC: `238.417070`; BIC: `275.494089`
- Max shape path: `4468.230081`; max implied variance: `4.578289`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000009)}{-0.056344} + \underset{(3.258619)}{0.548729}\,\pi_t + \underset{(5.587995)}{0.003487}\,\pi_{t-1} + \underset{(2.080404)}{0.492159}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(7.398763)}{0.192974}\,\omega_{p,t} - \underset{(0.003607)}{0.020563}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(29.046576)}{8.931381} + \underset{(2.332611)}{0.586700}\,p_{t-1} + \frac{\underset{(2.080463)}{0.599083}}{2(\underset{(7.398763)}{0.192974})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.587043)}{0.069957}}{2(\underset{(7.398763)}{0.192974})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(159.779666)}{5.064279} + \underset{(2.332611)}{0.586700}\,n_{t-1} + \frac{\underset{(2.080463)}{0.599083}}{2(\underset{(0.003607)}{0.020563})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.587043)}{0.069957}}{2(\underset{(0.003607)}{0.020563})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.056344 | 0.000009 |
| rho_1 | 0.548729 | 3.258619 |
| rho_2 | 0.003487 | 5.587995 |
| phi_1 | 0.492159 | 2.080404 |
| p0 | 8.931381 | 29.046576 |
| n0 | 5.064279 | 159.779666 |
| rho | 0.586700 | 2.332611 |
| phi_plus | 0.599083 | 2.080463 |
| phi_minus | 0.069957 | 0.587043 |
| sigma_p | 0.192974 | 7.398763 |
| sigma_n | 0.020563 | 0.003607 |

### Rank 6: Seed 29, Draw 7

- LogLik: `-110.828684`; AIC: `243.657368`; BIC: `280.734386`
- Max shape path: `58035.160089`; max implied variance: `14.717785`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(210.391598)}{0.088249} + \underset{(94.736049)}{0.393739}\,\pi_t + \underset{(119.203746)}{0.168936}\,\pi_{t-1} + \underset{(372.369307)}{0.556708}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000090)}{0.011175}\,\omega_{p,t} - \underset{(8.110577)}{0.127401}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(17278.074449)}{6.555633} + \underset{(10.323290)}{0.337734}\,p_{t-1} + \frac{\underset{(20.812743)}{0.138392}}{2(\underset{(0.000090)}{0.011175})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(88.197588)}{0.738718}}{2(\underset{(0.000090)}{0.011175})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(165.553268)}{9.171709} + \underset{(10.323290)}{0.337734}\,n_{t-1} + \frac{\underset{(20.812743)}{0.138392}}{2(\underset{(8.110577)}{0.127401})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(88.197588)}{0.738718}}{2(\underset{(8.110577)}{0.127401})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.088249 | 210.391598 |
| rho_1 | 0.393739 | 94.736049 |
| rho_2 | 0.168936 | 119.203746 |
| phi_1 | 0.556708 | 372.369307 |
| p0 | 6.555633 | 17278.074449 |
| n0 | 9.171709 | 165.553268 |
| rho | 0.337734 | 10.323290 |
| phi_plus | 0.138392 | 20.812743 |
| phi_minus | 0.738718 | 88.197588 |
| sigma_p | 0.011175 | 0.000090 |
| sigma_n | 0.127401 | 8.110577 |

### Rank 7: Seed 8, Draw 26

- LogLik: `-123.753325`; AIC: `269.506650`; BIC: `306.583668`
- Max shape path: `351.550369`; max implied variance: `7.498781`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(17832.130375)}{-0.037034} + \underset{(8477.861707)}{0.126063}\,\pi_t + \underset{(8618.896539)}{-0.196997}\,\pi_{t-1} + \underset{(33778.118167)}{1.308881}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(5441.917393)}{0.209352}\,\omega_{p,t} - \underset{(2137.518491)}{0.086800}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(317.709552)}{2.075765} + \underset{(0.974221)}{0.374577}\,p_{t-1} + \frac{\underset{(9214.892513)}{1.147907}}{2(\underset{(5441.917393)}{0.209352})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(9213.938029)}{0.073376}}{2(\underset{(5441.917393)}{0.209352})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(588.171071)}{2.637330} + \underset{(0.974221)}{0.374577}\,n_{t-1} + \frac{\underset{(9214.892513)}{1.147907}}{2(\underset{(2137.518491)}{0.086800})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(9213.938029)}{0.073376}}{2(\underset{(2137.518491)}{0.086800})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.037034 | 17832.130375 |
| rho_1 | 0.126063 | 8477.861707 |
| rho_2 | -0.196997 | 8618.896539 |
| phi_1 | 1.308881 | 33778.118167 |
| p0 | 2.075765 | 317.709552 |
| n0 | 2.637330 | 588.171071 |
| rho | 0.374577 | 0.974221 |
| phi_plus | 1.147907 | 9214.892513 |
| phi_minus | 0.073376 | 9213.938029 |
| sigma_p | 0.209352 | 5441.917393 |
| sigma_n | 0.086800 | 2137.518491 |

### Rank 8: Seed 42, Draw 23

- LogLik: `-133.015952`; AIC: `288.031903`; BIC: `325.108922`
- Max shape path: `1253.812395`; max implied variance: `9.908656`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(194.054827)}{0.008552} + \underset{(324.480883)}{0.293716}\,\pi_t + \underset{(194.597580)}{0.090795}\,\pi_{t-1} + \underset{(156.546866)}{0.838048}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(94.151070)}{0.437707}\,\omega_{p,t} - \underset{(10.785335)}{0.062119}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(10.622096)}{1.099507} + \underset{(0.000336)}{0.173193}\,p_{t-1} + \frac{\underset{(69.676451)}{1.070034}}{2(\underset{(94.151070)}{0.437707})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(69.672082)}{0.521414}}{2(\underset{(94.151070)}{0.437707})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(10.426623)}{4.838947} + \underset{(0.000336)}{0.173193}\,n_{t-1} + \frac{\underset{(69.676451)}{1.070034}}{2(\underset{(10.785335)}{0.062119})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(69.672082)}{0.521414}}{2(\underset{(10.785335)}{0.062119})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.008552 | 194.054827 |
| rho_1 | 0.293716 | 324.480883 |
| rho_2 | 0.090795 | 194.597580 |
| phi_1 | 0.838048 | 156.546866 |
| p0 | 1.099507 | 10.622096 |
| n0 | 4.838947 | 10.426623 |
| rho | 0.173193 | 0.000336 |
| phi_plus | 1.070034 | 69.676451 |
| phi_minus | 0.521414 | 69.672082 |
| sigma_p | 0.437707 | 94.151070 |
| sigma_n | 0.062119 | 10.785335 |

### Rank 9: Seed 1, Draw 27

- LogLik: `-137.090582`; AIC: `296.181165`; BIC: `333.258183`
- Max shape path: `3169.108831`; max implied variance: `7.169017`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(3.327190)}{0.148259} + \underset{(4.067133)}{0.352052}\,\pi_t + \underset{(4.180474)}{0.222409}\,\pi_{t-1} + \underset{(0.668723)}{0.379614}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.016065)}{0.033324}\,\omega_{p,t} - \underset{(0.229028)}{0.110592}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(23.496257)}{5.056291} + \underset{(3.361143)}{0.397778}\,p_{t-1} + \frac{\underset{(0.298112)}{0.541080}}{2(\underset{(0.016065)}{0.033324})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.417123)}{0.350216}}{2(\underset{(0.016065)}{0.033324})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.860193)}{6.881645} + \underset{(3.361143)}{0.397778}\,n_{t-1} + \frac{\underset{(0.298112)}{0.541080}}{2(\underset{(0.229028)}{0.110592})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.417123)}{0.350216}}{2(\underset{(0.229028)}{0.110592})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.148259 | 3.327190 |
| rho_1 | 0.352052 | 4.067133 |
| rho_2 | 0.222409 | 4.180474 |
| phi_1 | 0.379614 | 0.668723 |
| p0 | 5.056291 | 23.496257 |
| n0 | 6.881645 | 0.860193 |
| rho | 0.397778 | 3.361143 |
| phi_plus | 0.541080 | 0.298112 |
| phi_minus | 0.350216 | 0.417123 |
| sigma_p | 0.033324 | 0.016065 |
| sigma_n | 0.110592 | 0.229028 |

### Rank 10: Seed 41, Draw 32

- LogLik: `-144.233056`; AIC: `310.466112`; BIC: `347.543130`
- Max shape path: `179.315781`; max implied variance: `76.484555`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(10.349401)}{-0.080534} + \underset{(190.215989)}{0.237631}\,\pi_t + \underset{(102.479327)}{0.103556}\,\pi_{t-1} + \underset{(298.523345)}{0.539678}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(4.847759)}{0.205567}\,\omega_{p,t} - \underset{(35.125995)}{1.143348}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.060343)}{0.669929} + \underset{(0.000712)}{0.290328}\,p_{t-1} + \frac{\underset{(112.725270)}{1.411872}}{2(\underset{(4.847759)}{0.205567})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(112.725237)}{0.000000}}{2(\underset{(4.847759)}{0.205567})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.060568)}{0.196932} + \underset{(0.000712)}{0.290328}\,n_{t-1} + \frac{\underset{(112.725270)}{1.411872}}{2(\underset{(35.125995)}{1.143348})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(112.725237)}{0.000000}}{2(\underset{(35.125995)}{1.143348})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.080534 | 10.349401 |
| rho_1 | 0.237631 | 190.215989 |
| rho_2 | 0.103556 | 102.479327 |
| phi_1 | 0.539678 | 298.523345 |
| p0 | 0.669929 | 0.060343 |
| n0 | 0.196932 | 0.060568 |
| rho | 0.290328 | 0.000712 |
| phi_plus | 1.411872 | 112.725270 |
| phi_minus | 0.000000 | 112.725237 |
| sigma_p | 0.205567 | 4.847759 |
| sigma_n | 1.143348 | 35.125995 |

### Rank 11: Seed 27, Draw 37

- LogLik: `-147.617926`; AIC: `317.235852`; BIC: `354.312870`
- Max shape path: `28505.541845`; max implied variance: `4.216636`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.002369)}{0.154357} + \underset{(0.000000)}{0.116413}\,\pi_t + \underset{(0.000000)}{0.118401}\,\pi_{t-1} + \underset{(0.000029)}{0.620412}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000930)}{0.134261}\,\omega_{p,t} - \underset{(0.000001)}{0.008307}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.070257)}{9.058846} + \underset{(0.002373)}{0.422494}\,p_{t-1} + \frac{\underset{(0.000031)}{0.786887}}{2(\underset{(0.000930)}{0.134261})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000005)}{0.000039}}{2(\underset{(0.000930)}{0.134261})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.104247)}{2.876943} + \underset{(0.002373)}{0.422494}\,n_{t-1} + \frac{\underset{(0.000031)}{0.786887}}{2(\underset{(0.000001)}{0.008307})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000005)}{0.000039}}{2(\underset{(0.000001)}{0.008307})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.154357 | 0.002369 |
| rho_1 | 0.116413 | 0.000000 |
| rho_2 | 0.118401 | 0.000000 |
| phi_1 | 0.620412 | 0.000029 |
| p0 | 9.058846 | 0.070257 |
| n0 | 2.876943 | 0.104247 |
| rho | 0.422494 | 0.002373 |
| phi_plus | 0.786887 | 0.000031 |
| phi_minus | 0.000039 | 0.000005 |
| sigma_p | 0.134261 | 0.000930 |
| sigma_n | 0.008307 | 0.000001 |

### Rank 12: Seed 6, Draw 7

- LogLik: `-149.098987`; AIC: `320.197975`; BIC: `357.274993`
- Max shape path: `396.943733`; max implied variance: `1.411474`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1015.016685)}{0.015791} + \underset{(243.143707)}{0.212081}\,\pi_t + \underset{(29.854864)}{0.094414}\,\pi_{t-1} + \underset{(1058.163450)}{0.871128}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.003193)}{0.041146}\,\omega_{p,t} - \underset{(0.002312)}{0.045153}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3053.663014)}{2.462455} + \underset{(0.000061)}{0.844176}\,p_{t-1} + \frac{\underset{(100.829925)}{0.251229}}{2(\underset{(0.003193)}{0.041146})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(12.502203)}{0.000600}}{2(\underset{(0.003193)}{0.041146})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(2452.477411)}{7.200577} + \underset{(0.000061)}{0.844176}\,n_{t-1} + \frac{\underset{(100.829925)}{0.251229}}{2(\underset{(0.002312)}{0.045153})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(12.502203)}{0.000600}}{2(\underset{(0.002312)}{0.045153})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.015791 | 1015.016685 |
| rho_1 | 0.212081 | 243.143707 |
| rho_2 | 0.094414 | 29.854864 |
| phi_1 | 0.871128 | 1058.163450 |
| p0 | 2.462455 | 3053.663014 |
| n0 | 7.200577 | 2452.477411 |
| rho | 0.844176 | 0.000061 |
| phi_plus | 0.251229 | 100.829925 |
| phi_minus | 0.000600 | 12.502203 |
| sigma_p | 0.041146 | 0.003193 |
| sigma_n | 0.045153 | 0.002312 |

### Rank 13: Seed 23, Draw 4

- LogLik: `-149.915059`; AIC: `321.830118`; BIC: `358.907137`
- Max shape path: `6869.146863`; max implied variance: `8.220368`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000079)}{0.390049} + \underset{(0.000004)}{-0.066289}\,\pi_t + \underset{(0.000002)}{-0.035998}\,\pi_{t-1} + \underset{(0.000029)}{0.878312}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.813786)}{0.137843}\,\omega_{p,t} - \underset{(0.000700)}{0.024018}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(355.176667)}{9.559048} + \underset{(0.000001)}{0.394447}\,p_{t-1} + \frac{\underset{(0.914238)}{0.633019}}{2(\underset{(1.813786)}{0.137843})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(8.910662)}{0.437431}}{2(\underset{(1.813786)}{0.137843})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(123.426959)}{4.879062} + \underset{(0.000001)}{0.394447}\,n_{t-1} + \frac{\underset{(0.914238)}{0.633019}}{2(\underset{(0.000700)}{0.024018})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(8.910662)}{0.437431}}{2(\underset{(0.000700)}{0.024018})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.390049 | 0.000079 |
| rho_1 | -0.066289 | 0.000004 |
| rho_2 | -0.035998 | 0.000002 |
| phi_1 | 0.878312 | 0.000029 |
| p0 | 9.559048 | 355.176667 |
| n0 | 4.879062 | 123.426959 |
| rho | 0.394447 | 0.000001 |
| phi_plus | 0.633019 | 0.914238 |
| phi_minus | 0.437431 | 8.910662 |
| sigma_p | 0.137843 | 1.813786 |
| sigma_n | 0.024018 | 0.000700 |

### Rank 14: Seed 18, Draw 13

- LogLik: `-157.390206`; AIC: `336.780412`; BIC: `373.857430`
- Max shape path: `358.485927`; max implied variance: `1.328582`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.051350} + \underset{(0.000001)}{0.259806}\,\pi_t + \underset{(0.000145)}{0.090988}\,\pi_{t-1} + \underset{(0.000147)}{0.665047}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000035)}{0.041865}\,\omega_{p,t} - \underset{(0.000138)}{0.092073}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(62.601683)}{8.006328} + \underset{(0.000134)}{0.716682}\,p_{t-1} + \frac{\underset{(0.000137)}{0.253845}}{2(\underset{(0.000035)}{0.041865})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000000)}{0.043991}}{2(\underset{(0.000035)}{0.041865})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.000151)}{4.060656} + \underset{(0.000134)}{0.716682}\,n_{t-1} + \frac{\underset{(0.000137)}{0.253845}}{2(\underset{(0.000138)}{0.092073})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000000)}{0.043991}}{2(\underset{(0.000138)}{0.092073})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.051350 | 0.000002 |
| rho_1 | 0.259806 | 0.000001 |
| rho_2 | 0.090988 | 0.000145 |
| phi_1 | 0.665047 | 0.000147 |
| p0 | 8.006328 | 62.601683 |
| n0 | 4.060656 | 0.000151 |
| rho | 0.716682 | 0.000134 |
| phi_plus | 0.253845 | 0.000137 |
| phi_minus | 0.043991 | 0.000000 |
| sigma_p | 0.041865 | 0.000035 |
| sigma_n | 0.092073 | 0.000138 |

### Rank 15: Seed 22, Draw 2

- LogLik: `-157.819369`; AIC: `337.638738`; BIC: `374.715756`
- Max shape path: `8315.894717`; max implied variance: `4.433770`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.822256)}{0.200962} + \underset{(0.713604)}{0.157299}\,\pi_t + \underset{(0.577991)}{0.116848}\,\pi_{t-1} + \underset{(0.306991)}{0.388845}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.030756}\,\omega_{p,t} - \underset{(0.000007)}{0.016299}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(4.976564)}{6.677585} + \underset{(0.000019)}{0.623485}\,p_{t-1} + \frac{\underset{(0.126577)}{0.367726}}{2(\underset{(0.000002)}{0.030756})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.138898)}{0.239605}}{2(\underset{(0.000002)}{0.030756})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(17.472776)}{1.600630} + \underset{(0.000019)}{0.623485}\,n_{t-1} + \frac{\underset{(0.126577)}{0.367726}}{2(\underset{(0.000007)}{0.016299})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.138898)}{0.239605}}{2(\underset{(0.000007)}{0.016299})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.200962 | 0.822256 |
| rho_1 | 0.157299 | 0.713604 |
| rho_2 | 0.116848 | 0.577991 |
| phi_1 | 0.388845 | 0.306991 |
| p0 | 6.677585 | 4.976564 |
| n0 | 1.600630 | 17.472776 |
| rho | 0.623485 | 0.000019 |
| phi_plus | 0.367726 | 0.126577 |
| phi_minus | 0.239605 | 0.138898 |
| sigma_p | 0.030756 | 0.000002 |
| sigma_n | 0.016299 | 0.000007 |

### Rank 16: Seed 14, Draw 33

- LogLik: `-159.451118`; AIC: `340.902235`; BIC: `377.979254`
- Max shape path: `4196.116535`; max implied variance: `2.436154`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(35.769313)}{0.077220} + \underset{(54.359703)}{0.241514}\,\pi_t + \underset{(31.996978)}{-0.000556}\,\pi_{t-1} + \underset{(143.758144)}{0.767430}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(2.098449)}{0.056991}\,\omega_{p,t} - \underset{(0.000000)}{0.016678}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(527.253074)}{9.999683} + \underset{(4.371933)}{0.683944}\,p_{t-1} + \frac{\underset{(43.536730)}{0.515620}}{2(\underset{(2.098449)}{0.056991})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000840)}{0.000070}}{2(\underset{(2.098449)}{0.056991})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(6488.489626)}{1.103892} + \underset{(4.371933)}{0.683944}\,n_{t-1} + \frac{\underset{(43.536730)}{0.515620}}{2(\underset{(0.000000)}{0.016678})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000840)}{0.000070}}{2(\underset{(0.000000)}{0.016678})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.077220 | 35.769313 |
| rho_1 | 0.241514 | 54.359703 |
| rho_2 | -0.000556 | 31.996978 |
| phi_1 | 0.767430 | 143.758144 |
| p0 | 9.999683 | 527.253074 |
| n0 | 1.103892 | 6488.489626 |
| rho | 0.683944 | 4.371933 |
| phi_plus | 0.515620 | 43.536730 |
| phi_minus | 0.000070 | 0.000840 |
| sigma_p | 0.056991 | 2.098449 |
| sigma_n | 0.016678 | 0.000000 |

### Rank 17: Seed 29, Draw 36

- LogLik: `-165.415635`; AIC: `352.831270`; BIC: `389.908288`
- Max shape path: `3862.496623`; max implied variance: `4.627832`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(3.181123)}{0.379105} + \underset{(6.788560)}{0.105315}\,\pi_t + \underset{(23.615908)}{0.049703}\,\pi_{t-1} + \underset{(36.490498)}{0.798460}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001399)}{0.021817}\,\omega_{p,t} - \underset{(0.348670)}{0.269793}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1381.148377)}{0.509180} + \underset{(4.445777)}{0.700016}\,p_{t-1} + \frac{\underset{(3.085736)}{0.296889}}{2(\underset{(0.001399)}{0.021817})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.804690)}{0.177622}}{2(\underset{(0.001399)}{0.021817})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(67.434928)}{3.922673} + \underset{(4.445777)}{0.700016}\,n_{t-1} + \frac{\underset{(3.085736)}{0.296889}}{2(\underset{(0.348670)}{0.269793})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(2.804690)}{0.177622}}{2(\underset{(0.348670)}{0.269793})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.379105 | 3.181123 |
| rho_1 | 0.105315 | 6.788560 |
| rho_2 | 0.049703 | 23.615908 |
| phi_1 | 0.798460 | 36.490498 |
| p0 | 0.509180 | 1381.148377 |
| n0 | 3.922673 | 67.434928 |
| rho | 0.700016 | 4.445777 |
| phi_plus | 0.296889 | 3.085736 |
| phi_minus | 0.177622 | 2.804690 |
| sigma_p | 0.021817 | 0.001399 |
| sigma_n | 0.269793 | 0.348670 |

### Rank 18: Seed 6, Draw 28

- LogLik: `-167.158491`; AIC: `356.316983`; BIC: `393.394001`
- Max shape path: `788.761257`; max implied variance: `1.240715`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.003627)}{0.157691} + \underset{(0.000933)}{0.223885}\,\pi_t + \underset{(0.000094)}{0.084327}\,\pi_{t-1} + \underset{(0.068578)}{0.477315}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.037316}\,\omega_{p,t} - \underset{(0.001369)}{0.027338}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(65.532118)}{6.283930} + \underset{(0.034244)}{0.908422}\,p_{t-1} + \frac{\underset{(0.000713)}{0.096445}}{2(\underset{(0.000002)}{0.037316})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000049)}{0.031399}}{2(\underset{(0.000002)}{0.037316})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(65.400577)}{4.391603} + \underset{(0.034244)}{0.908422}\,n_{t-1} + \frac{\underset{(0.000713)}{0.096445}}{2(\underset{(0.001369)}{0.027338})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000049)}{0.031399}}{2(\underset{(0.001369)}{0.027338})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.157691 | 0.003627 |
| rho_1 | 0.223885 | 0.000933 |
| rho_2 | 0.084327 | 0.000094 |
| phi_1 | 0.477315 | 0.068578 |
| p0 | 6.283930 | 65.532118 |
| n0 | 4.391603 | 65.400577 |
| rho | 0.908422 | 0.034244 |
| phi_plus | 0.096445 | 0.000713 |
| phi_minus | 0.031399 | 0.000049 |
| sigma_p | 0.037316 | 0.000002 |
| sigma_n | 0.027338 | 0.001369 |

### Rank 19: Seed 9, Draw 29

- LogLik: `-167.964823`; AIC: `357.929645`; BIC: `395.006663`
- Max shape path: `511.306262`; max implied variance: `6.072363`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(8.573400)}{-0.097650} + \underset{(4.782305)}{0.160759}\,\pi_t + \underset{(4.752016)}{-0.003350}\,\pi_{t-1} + \underset{(2.497593)}{0.275927}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001227)}{0.035326}\,\omega_{p,t} - \underset{(1.560831)}{0.277803}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(103.125770)}{1.854100} + \underset{(0.000475)}{0.908090}\,p_{t-1} + \frac{\underset{(0.000748)}{0.024514}}{2(\underset{(0.001227)}{0.035326})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.467276)}{0.078277}}{2(\underset{(0.001227)}{0.035326})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(4.560234)}{3.157858} + \underset{(0.000475)}{0.908090}\,n_{t-1} + \frac{\underset{(0.000748)}{0.024514}}{2(\underset{(1.560831)}{0.277803})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.467276)}{0.078277}}{2(\underset{(1.560831)}{0.277803})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.097650 | 8.573400 |
| rho_1 | 0.160759 | 4.782305 |
| rho_2 | -0.003350 | 4.752016 |
| phi_1 | 0.275927 | 2.497593 |
| p0 | 1.854100 | 103.125770 |
| n0 | 3.157858 | 4.560234 |
| rho | 0.908090 | 0.000475 |
| phi_plus | 0.024514 | 0.000748 |
| phi_minus | 0.078277 | 0.467276 |
| sigma_p | 0.035326 | 0.001227 |
| sigma_n | 0.277803 | 1.560831 |

### Rank 20: Seed 13, Draw 40

- LogLik: `-168.212202`; AIC: `358.424403`; BIC: `395.501421`
- Max shape path: `8.001020`; max implied variance: `2.650826`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000022)}{0.188944} + \underset{(0.001835)}{0.193535}\,\pi_t + \underset{(0.000261)}{0.243289}\,\pi_{t-1} + \underset{(0.006861)}{0.321614}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.110122)}{0.405307}\,\omega_{p,t} - \underset{(0.172658)}{0.603839}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000074)}{0.182003} + \underset{(0.013069)}{0.627211}\,p_{t-1} + \frac{\underset{(0.022910)}{0.436921}}{2(\underset{(0.110122)}{0.405307})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.048933)}{0.105308}}{2(\underset{(0.110122)}{0.405307})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.000001)}{0.104610} + \underset{(0.013069)}{0.627211}\,n_{t-1} + \frac{\underset{(0.022910)}{0.436921}}{2(\underset{(0.172658)}{0.603839})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.048933)}{0.105308}}{2(\underset{(0.172658)}{0.603839})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.188944 | 0.000022 |
| rho_1 | 0.193535 | 0.001835 |
| rho_2 | 0.243289 | 0.000261 |
| phi_1 | 0.321614 | 0.006861 |
| p0 | 0.182003 | 0.000074 |
| n0 | 0.104610 | 0.000001 |
| rho | 0.627211 | 0.013069 |
| phi_plus | 0.436921 | 0.022910 |
| phi_minus | 0.105308 | 0.048933 |
| sigma_p | 0.405307 | 0.110122 |
| sigma_n | 0.603839 | 0.172658 |

## ARX(2,2)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 8 | 30 | 103.495171 | -182.990343 | -142.542687 | 1259.701093 | 21.519226 | yes | `computed` |
| 2 | 30 | 18 | 90.936287 | -157.872573 | -117.424917 | 2667.006996 | 5.553789 | yes | `computed` |
| 3 | 30 | 6 | -6.141744 | 36.283488 | 76.731144 | 574.261901 | 2.525784 | yes | `computed` |
| 4 | 21 | 14 | -46.378541 | 116.757081 | 157.204738 | 7096.098750 | 1.589305 | yes | `computed` |
| 5 | 50 | 7 | -66.282420 | 156.564840 | 197.012496 | 467.575642 | 6.034697 | yes | `computed` |
| 6 | 37 | 32 | -101.270558 | 226.541115 | 266.988771 | 7134.355369 | 2.404286 | yes | `computed` |
| 7 | 32 | 3 | -116.085629 | 256.171258 | 296.618915 | 486.298026 | 63.408959 | yes | `computed` |
| 8 | 1 | 3 | -117.666868 | 259.333737 | 299.781393 | 789.280551 | 5.078037 | yes | `computed` |
| 9 | 10 | 11 | -120.717350 | 265.434699 | 305.882356 | 1522.542735 | 4.171985 | yes | `computed` |
| 10 | 50 | 38 | -133.828366 | 291.656732 | 332.104388 | 935.186578 | 56.156244 | yes | `computed` |
| 11 | 46 | 1 | -137.884950 | 299.769901 | 340.217557 | 1065.000192 | 3.527382 | yes | `computed` |
| 12 | 17 | 31 | -148.780573 | 321.561147 | 362.008803 | 2922.932419 | 4.610658 | yes | `computed` |
| 13 | 10 | 38 | -150.078170 | 324.156339 | 364.603995 | 944.933418 | 3.932327 | no | `computed` |
| 14 | 41 | 37 | -157.215114 | 338.430229 | 378.877885 | 1735.241427 | 15.920894 | no | `computed` |
| 15 | 45 | 14 | -158.161416 | 340.322832 | 380.770489 | 4828.251225 | 2.202414 | no | `computed` |
| 16 | 5 | 7 | -159.021889 | 342.043777 | 382.491434 | 12730.655593 | 4.779221 | no | `computed` |
| 17 | 21 | 24 | -160.459730 | 344.919460 | 385.367116 | 212.717795 | 1.310455 | no | `computed` |
| 18 | 25 | 9 | -162.078426 | 348.156853 | 388.604509 | 27336.120557 | 4.691223 | no | `computed` |
| 19 | 10 | 26 | -163.378955 | 350.757910 | 391.205566 | 2822.105088 | 3.372188 | no | `computed` |
| 20 | 30 | 40 | -166.629932 | 357.259864 | 397.707520 | 898.602978 | 1.918708 | no | `computed` |

### Rank 1: Seed 8, Draw 30

- LogLik: `103.495171`; AIC: `-182.990343`; BIC: `-142.542687`
- Max shape path: `1259.701093`; max implied variance: `21.519226`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000003)}{0.043346} + \underset{(0.002508)}{0.184817}\,\pi_t + \underset{(0.000197)}{-0.046132}\,\pi_{t-1} + \underset{(0.021313)}{0.499300}\,SPF_t + \underset{(0.023041)}{0.667900}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.091960)}{0.181297}\,\omega_{p,t} - \underset{(0.000011)}{0.028670}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.673841)}{8.031497} + \underset{(0.000002)}{0.778679}\,p_{t-1} + \frac{\underset{(0.000005)}{0.394227}}{2(\underset{(0.091960)}{0.181297})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000191)}{0.023401}}{2(\underset{(0.091960)}{0.181297})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.125732)}{6.281045} + \underset{(0.000002)}{0.778679}\,n_{t-1} + \frac{\underset{(0.000005)}{0.394227}}{2(\underset{(0.000011)}{0.028670})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000191)}{0.023401}}{2(\underset{(0.000011)}{0.028670})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.043346 | 0.000003 |
| rho_1 | 0.184817 | 0.002508 |
| rho_2 | -0.046132 | 0.000197 |
| phi_1 | 0.499300 | 0.021313 |
| phi_2 | 0.667900 | 0.023041 |
| p0 | 8.031497 | 0.673841 |
| n0 | 6.281045 | 0.125732 |
| rho | 0.778679 | 0.000002 |
| phi_plus | 0.394227 | 0.000005 |
| phi_minus | 0.023401 | 0.000191 |
| sigma_p | 0.181297 | 0.091960 |
| sigma_n | 0.028670 | 0.000011 |

### Rank 2: Seed 30, Draw 18

- LogLik: `90.936287`; AIC: `-157.872573`; BIC: `-117.424917`
- Max shape path: `2667.006996`; max implied variance: `5.553789`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000005)}{-0.565703} + \underset{(0.000003)}{-0.170832}\,\pi_t + \underset{(0.000006)}{-0.071699}\,\pi_{t-1} + \underset{(0.000005)}{1.480241}\,SPF_t + \underset{(0.000003)}{0.876347}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000009)}{0.200273}\,\omega_{p,t} - \underset{(0.000005)}{0.028054}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.941851)}{8.760070} + \underset{(0.000003)}{0.742767}\,p_{t-1} + \frac{\underset{(0.000013)}{0.135301}}{2(\underset{(0.000009)}{0.200273})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000000)}{0.221380}}{2(\underset{(0.000009)}{0.200273})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.943685)}{3.324015} + \underset{(0.000003)}{0.742767}\,n_{t-1} + \frac{\underset{(0.000013)}{0.135301}}{2(\underset{(0.000005)}{0.028054})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000000)}{0.221380}}{2(\underset{(0.000005)}{0.028054})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.565703 | 0.000005 |
| rho_1 | -0.170832 | 0.000003 |
| rho_2 | -0.071699 | 0.000006 |
| phi_1 | 1.480241 | 0.000005 |
| phi_2 | 0.876347 | 0.000003 |
| p0 | 8.760070 | 0.941851 |
| n0 | 3.324015 | 0.943685 |
| rho | 0.742767 | 0.000003 |
| phi_plus | 0.135301 | 0.000013 |
| phi_minus | 0.221380 | 0.000000 |
| sigma_p | 0.200273 | 0.000009 |
| sigma_n | 0.028054 | 0.000005 |

### Rank 3: Seed 30, Draw 6

- LogLik: `-6.141744`; AIC: `36.283488`; BIC: `76.731144`
- Max shape path: `574.261901`; max implied variance: `2.525784`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.048439)}{-0.084629} + \underset{(0.448419)}{0.236799}\,\pi_t + \underset{(0.015836)}{0.145062}\,\pi_{t-1} + \underset{(1.086324)}{0.548225}\,SPF_t + \underset{(0.670162)}{0.141850}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000044)}{0.030545}\,\omega_{p,t} - \underset{(0.177746)}{0.202755}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.979714)}{3.735992} + \underset{(0.000044)}{0.795260}\,p_{t-1} + \frac{\underset{(0.000041)}{0.039509}}{2(\underset{(0.000044)}{0.030545})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000131)}{0.055524}}{2(\underset{(0.000044)}{0.030545})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(3.325383)}{7.327373} + \underset{(0.000044)}{0.795260}\,n_{t-1} + \frac{\underset{(0.000041)}{0.039509}}{2(\underset{(0.177746)}{0.202755})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000131)}{0.055524}}{2(\underset{(0.177746)}{0.202755})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.084629 | 0.048439 |
| rho_1 | 0.236799 | 0.448419 |
| rho_2 | 0.145062 | 0.015836 |
| phi_1 | 0.548225 | 1.086324 |
| phi_2 | 0.141850 | 0.670162 |
| p0 | 3.735992 | 1.979714 |
| n0 | 7.327373 | 3.325383 |
| rho | 0.795260 | 0.000044 |
| phi_plus | 0.039509 | 0.000041 |
| phi_minus | 0.055524 | 0.000131 |
| sigma_p | 0.030545 | 0.000044 |
| sigma_n | 0.202755 | 0.177746 |

### Rank 4: Seed 21, Draw 14

- LogLik: `-46.378541`; AIC: `116.757081`; BIC: `157.204738`
- Max shape path: `7096.098750`; max implied variance: `1.589305`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000000)}{0.010316} + \underset{(0.013472)}{0.237341}\,\pi_t + \underset{(0.038817)}{0.042893}\,\pi_{t-1} + \underset{(0.000005)}{0.536182}\,SPF_t + \underset{(0.066558)}{0.257009}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000001)}{0.010184}\,\omega_{p,t} - \underset{(0.000005)}{0.044505}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.082115)}{0.000239} + \underset{(0.000006)}{0.831275}\,p_{t-1} + \frac{\underset{(0.014278)}{0.249302}}{2(\underset{(0.000001)}{0.010184})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000001)}{0.000001}}{2(\underset{(0.000001)}{0.010184})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.082233)}{9.999756} + \underset{(0.000006)}{0.831275}\,n_{t-1} + \frac{\underset{(0.014278)}{0.249302}}{2(\underset{(0.000005)}{0.044505})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000001)}{0.000001}}{2(\underset{(0.000005)}{0.044505})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.010316 | 0.000000 |
| rho_1 | 0.237341 | 0.013472 |
| rho_2 | 0.042893 | 0.038817 |
| phi_1 | 0.536182 | 0.000005 |
| phi_2 | 0.257009 | 0.066558 |
| p0 | 0.000239 | 0.082115 |
| n0 | 9.999756 | 0.082233 |
| rho | 0.831275 | 0.000006 |
| phi_plus | 0.249302 | 0.014278 |
| phi_minus | 0.000001 | 0.000001 |
| sigma_p | 0.010184 | 0.000001 |
| sigma_n | 0.044505 | 0.000005 |

### Rank 5: Seed 50, Draw 7

- LogLik: `-66.282420`; AIC: `156.564840`; BIC: `197.012496`
- Max shape path: `467.575642`; max implied variance: `6.034697`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(26.978568)}{-0.105336} + \underset{(293.315640)}{-0.079264}\,\pi_t + \underset{(308.794494)}{-0.163079}\,\pi_{t-1} + \underset{(350.274994)}{-0.568168}\,SPF_t + \underset{(365.568875)}{1.346827}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000770)}{0.047203}\,\omega_{p,t} - \underset{(0.637689)}{0.275456}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(733.863466)}{1.990700} + \underset{(0.000040)}{0.890070}\,p_{t-1} + \frac{\underset{(0.000914)}{0.046611}}{2(\underset{(0.000770)}{0.047203})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(4.697691)}{0.008521}}{2(\underset{(0.000770)}{0.047203})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(21.252768)}{5.772698} + \underset{(0.000040)}{0.890070}\,n_{t-1} + \frac{\underset{(0.000914)}{0.046611}}{2(\underset{(0.637689)}{0.275456})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(4.697691)}{0.008521}}{2(\underset{(0.637689)}{0.275456})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.105336 | 26.978568 |
| rho_1 | -0.079264 | 293.315640 |
| rho_2 | -0.163079 | 308.794494 |
| phi_1 | -0.568168 | 350.274994 |
| phi_2 | 1.346827 | 365.568875 |
| p0 | 1.990700 | 733.863466 |
| n0 | 5.772698 | 21.252768 |
| rho | 0.890070 | 0.000040 |
| phi_plus | 0.046611 | 0.000914 |
| phi_minus | 0.008521 | 4.697691 |
| sigma_p | 0.047203 | 0.000770 |
| sigma_n | 0.275456 | 0.637689 |

### Rank 6: Seed 37, Draw 32

- LogLik: `-101.270558`; AIC: `226.541115`; BIC: `266.988771`
- Max shape path: `7134.355369`; max implied variance: `2.404286`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(51.100577)}{0.095908} + \underset{(29.445750)}{0.298398}\,\pi_t + \underset{(76.928425)}{0.062221}\,\pi_{t-1} + \underset{(194.891321)}{0.649963}\,SPF_t + \underset{(3.927320)}{-0.070761}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000591)}{0.012735}\,\omega_{p,t} - \underset{(1.175633)}{0.095142}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(6085.613571)}{5.695515} + \underset{(12.610667)}{0.778054}\,p_{t-1} + \frac{\underset{(92.914693)}{0.235541}}{2(\underset{(0.000591)}{0.012735})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(9.023288)}{0.114413}}{2(\underset{(0.000591)}{0.012735})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(1367.412825)}{2.309401} + \underset{(12.610667)}{0.778054}\,n_{t-1} + \frac{\underset{(92.914693)}{0.235541}}{2(\underset{(1.175633)}{0.095142})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(9.023288)}{0.114413}}{2(\underset{(1.175633)}{0.095142})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.095908 | 51.100577 |
| rho_1 | 0.298398 | 29.445750 |
| rho_2 | 0.062221 | 76.928425 |
| phi_1 | 0.649963 | 194.891321 |
| phi_2 | -0.070761 | 3.927320 |
| p0 | 5.695515 | 6085.613571 |
| n0 | 2.309401 | 1367.412825 |
| rho | 0.778054 | 12.610667 |
| phi_plus | 0.235541 | 92.914693 |
| phi_minus | 0.114413 | 9.023288 |
| sigma_p | 0.012735 | 0.000591 |
| sigma_n | 0.095142 | 1.175633 |

### Rank 7: Seed 32, Draw 3

- LogLik: `-116.085629`; AIC: `256.171258`; BIC: `296.618915`
- Max shape path: `486.298026`; max implied variance: `63.408959`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000003)}{0.236912} + \underset{(0.000026)}{0.148612}\,\pi_t + \underset{(0.000002)}{0.033056}\,\pi_{t-1} + \underset{(0.000002)}{-0.036242}\,SPF_t + \underset{(0.000023)}{0.145318}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000017)}{0.089209}\,\omega_{p,t} - \underset{(0.000850)}{0.687749}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000002)}{5.334682} + \underset{(0.000013)}{0.962430}\,p_{t-1} + \frac{\underset{(0.000001)}{0.030417}}{2(\underset{(0.000017)}{0.089209})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000003)}{0.022783}}{2(\underset{(0.000017)}{0.089209})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.000025)}{1.380849} + \underset{(0.000013)}{0.962430}\,n_{t-1} + \frac{\underset{(0.000001)}{0.030417}}{2(\underset{(0.000850)}{0.687749})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000003)}{0.022783}}{2(\underset{(0.000850)}{0.687749})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.236912 | 0.000003 |
| rho_1 | 0.148612 | 0.000026 |
| rho_2 | 0.033056 | 0.000002 |
| phi_1 | -0.036242 | 0.000002 |
| phi_2 | 0.145318 | 0.000023 |
| p0 | 5.334682 | 0.000002 |
| n0 | 1.380849 | 0.000025 |
| rho | 0.962430 | 0.000013 |
| phi_plus | 0.030417 | 0.000001 |
| phi_minus | 0.022783 | 0.000003 |
| sigma_p | 0.089209 | 0.000017 |
| sigma_n | 0.687749 | 0.000850 |

### Rank 8: Seed 1, Draw 3

- LogLik: `-117.666868`; AIC: `259.333737`; BIC: `299.781393`
- Max shape path: `789.280551`; max implied variance: `5.078037`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(12.976339)}{0.277192} + \underset{(39.220559)}{0.259529}\,\pi_t + \underset{(3.240816)}{0.048807}\,\pi_{t-1} + \underset{(65.303085)}{0.643976}\,SPF_t + \underset{(38.254105)}{0.449520}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(1.078301)}{0.244281}\,\omega_{p,t} - \underset{(0.000066)}{0.023706}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(2.810635)}{3.992379} + \underset{(0.001003)}{0.797935}\,p_{t-1} + \frac{\underset{(11.924610)}{0.282353}}{2(\underset{(1.078301)}{0.244281})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000047)}{0.026469}}{2(\underset{(1.078301)}{0.244281})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(35.285642)}{6.673997} + \underset{(0.001003)}{0.797935}\,n_{t-1} + \frac{\underset{(11.924610)}{0.282353}}{2(\underset{(0.000066)}{0.023706})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000047)}{0.026469}}{2(\underset{(0.000066)}{0.023706})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.277192 | 12.976339 |
| rho_1 | 0.259529 | 39.220559 |
| rho_2 | 0.048807 | 3.240816 |
| phi_1 | 0.643976 | 65.303085 |
| phi_2 | 0.449520 | 38.254105 |
| p0 | 3.992379 | 2.810635 |
| n0 | 6.673997 | 35.285642 |
| rho | 0.797935 | 0.001003 |
| phi_plus | 0.282353 | 11.924610 |
| phi_minus | 0.026469 | 0.000047 |
| sigma_p | 0.244281 | 1.078301 |
| sigma_n | 0.023706 | 0.000066 |

### Rank 9: Seed 10, Draw 11

- LogLik: `-120.717350`; AIC: `265.434699`; BIC: `305.882356`
- Max shape path: `1522.542735`; max implied variance: `4.171985`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.122727)}{-0.045710} + \underset{(0.195811)}{0.575529}\,\pi_t + \underset{(0.474644)}{-0.160319}\,\pi_{t-1} + \underset{(0.732458)}{1.703687}\,SPF_t + \underset{(0.851773)}{-1.080814}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.059999)}{0.090842}\,\omega_{p,t} - \underset{(0.000059)}{0.036869}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.067184)}{0.970990} + \underset{(0.000000)}{0.799197}\,p_{t-1} + \frac{\underset{(0.000003)}{0.147906}}{2(\underset{(0.059999)}{0.090842})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000002)}{0.247001}}{2(\underset{(0.059999)}{0.090842})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.027882)}{1.050636} + \underset{(0.000000)}{0.799197}\,n_{t-1} + \frac{\underset{(0.000003)}{0.147906}}{2(\underset{(0.000059)}{0.036869})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000002)}{0.247001}}{2(\underset{(0.000059)}{0.036869})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.045710 | 0.122727 |
| rho_1 | 0.575529 | 0.195811 |
| rho_2 | -0.160319 | 0.474644 |
| phi_1 | 1.703687 | 0.732458 |
| phi_2 | -1.080814 | 0.851773 |
| p0 | 0.970990 | 0.067184 |
| n0 | 1.050636 | 0.027882 |
| rho | 0.799197 | 0.000000 |
| phi_plus | 0.147906 | 0.000003 |
| phi_minus | 0.247001 | 0.000002 |
| sigma_p | 0.090842 | 0.059999 |
| sigma_n | 0.036869 | 0.000059 |

### Rank 10: Seed 50, Draw 38

- LogLik: `-133.828366`; AIC: `291.656732`; BIC: `332.104388`
- Max shape path: `935.186578`; max implied variance: `56.156244`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(135.659818)}{0.193387} + \underset{(17.198396)}{0.115846}\,\pi_t + \underset{(69.479760)}{-0.052193}\,\pi_{t-1} + \underset{(350.852798)}{0.487999}\,SPF_t + \underset{(19.320235)}{0.575355}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(3.123096)}{1.561747}\,\omega_{p,t} - \underset{(0.000002)}{0.123389}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.173456)}{0.195533} + \underset{(0.000290)}{0.100841}\,p_{t-1} + \frac{\underset{(78.802508)}{0.182527}}{2(\underset{(3.123096)}{1.561747})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(78.802701)}{1.597948}}{2(\underset{(3.123096)}{1.561747})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(1.254380)}{1.582319} + \underset{(0.000290)}{0.100841}\,n_{t-1} + \frac{\underset{(78.802508)}{0.182527}}{2(\underset{(0.000002)}{0.123389})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(78.802701)}{1.597948}}{2(\underset{(0.000002)}{0.123389})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.193387 | 135.659818 |
| rho_1 | 0.115846 | 17.198396 |
| rho_2 | -0.052193 | 69.479760 |
| phi_1 | 0.487999 | 350.852798 |
| phi_2 | 0.575355 | 19.320235 |
| p0 | 0.195533 | 0.173456 |
| n0 | 1.582319 | 1.254380 |
| rho | 0.100841 | 0.000290 |
| phi_plus | 0.182527 | 78.802508 |
| phi_minus | 1.597948 | 78.802701 |
| sigma_p | 1.561747 | 3.123096 |
| sigma_n | 0.123389 | 0.000002 |

### Rank 11: Seed 46, Draw 1

- LogLik: `-137.884950`; AIC: `299.769901`; BIC: `340.217557`
- Max shape path: `1065.000192`; max implied variance: `3.527382`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(3923.857627)}{0.108483} + \underset{(3477.604216)}{0.212902}\,\pi_t + \underset{(3926.058928)}{0.157874}\,\pi_{t-1} + \underset{(68907.299045)}{0.588676}\,SPF_t + \underset{(60883.353567)}{-0.007502}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.074377)}{0.040397}\,\omega_{p,t} - \underset{(0.000026)}{0.070009}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(64629.871578)}{9.794632} + \underset{(0.000082)}{0.430754}\,p_{t-1} + \frac{\underset{(234.141331)}{0.670107}}{2(\underset{(0.074377)}{0.040397})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(626.674284)}{0.176117}}{2(\underset{(0.074377)}{0.040397})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(21380.302142)}{9.229934} + \underset{(0.000082)}{0.430754}\,n_{t-1} + \frac{\underset{(234.141331)}{0.670107}}{2(\underset{(0.000026)}{0.070009})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(626.674284)}{0.176117}}{2(\underset{(0.000026)}{0.070009})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.108483 | 3923.857627 |
| rho_1 | 0.212902 | 3477.604216 |
| rho_2 | 0.157874 | 3926.058928 |
| phi_1 | 0.588676 | 68907.299045 |
| phi_2 | -0.007502 | 60883.353567 |
| p0 | 9.794632 | 64629.871578 |
| n0 | 9.229934 | 21380.302142 |
| rho | 0.430754 | 0.000082 |
| phi_plus | 0.670107 | 234.141331 |
| phi_minus | 0.176117 | 626.674284 |
| sigma_p | 0.040397 | 0.074377 |
| sigma_n | 0.070009 | 0.000026 |

### Rank 12: Seed 17, Draw 31

- LogLik: `-148.780573`; AIC: `321.561147`; BIC: `362.008803`
- Max shape path: `2922.932419`; max implied variance: `4.610658`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(321.266735)}{-0.016700} + \underset{(1084.071549)}{0.151698}\,\pi_t + \underset{(1207.795635)}{0.095135}\,\pi_{t-1} + \underset{(7322.946155)}{0.552378}\,SPF_t + \underset{(7671.250301)}{0.022577}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(5.312307)}{0.138571}\,\omega_{p,t} - \underset{(0.001368)}{0.026781}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1598.635905)}{5.292497} + \underset{(0.000486)}{0.761112}\,p_{t-1} + \frac{\underset{(0.002125)}{0.155351}}{2(\underset{(5.312307)}{0.138571})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(32.135583)}{0.242060}}{2(\underset{(5.312307)}{0.138571})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(4400.429816)}{2.522151} + \underset{(0.000486)}{0.761112}\,n_{t-1} + \frac{\underset{(0.002125)}{0.155351}}{2(\underset{(0.001368)}{0.026781})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(32.135583)}{0.242060}}{2(\underset{(0.001368)}{0.026781})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.016700 | 321.266735 |
| rho_1 | 0.151698 | 1084.071549 |
| rho_2 | 0.095135 | 1207.795635 |
| phi_1 | 0.552378 | 7322.946155 |
| phi_2 | 0.022577 | 7671.250301 |
| p0 | 5.292497 | 1598.635905 |
| n0 | 2.522151 | 4400.429816 |
| rho | 0.761112 | 0.000486 |
| phi_plus | 0.155351 | 0.002125 |
| phi_minus | 0.242060 | 32.135583 |
| sigma_p | 0.138571 | 5.312307 |
| sigma_n | 0.026781 | 0.001368 |

### Rank 13: Seed 10, Draw 38

- LogLik: `-150.078170`; AIC: `324.156339`; BIC: `364.603995`
- Max shape path: `944.933418`; max implied variance: `3.932327`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(697.168533)}{0.105549} + \underset{(361.613681)}{0.246211}\,\pi_t + \underset{(98.211741)}{0.060730}\,\pi_{t-1} + \underset{(76.765988)}{0.589309}\,SPF_t + \underset{(1152.677294)}{0.030801}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.761605)}{0.099267}\,\omega_{p,t} - \underset{(7.607059)}{0.045159}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(6432.648848)}{5.517741} + \underset{(15.501734)}{0.478619}\,p_{t-1} + \frac{\underset{(947.857352)}{0.866941}}{2(\underset{(0.761605)}{0.099267})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(801.569247)}{0.077667}}{2(\underset{(0.761605)}{0.099267})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(7890.268578)}{6.662755} + \underset{(15.501734)}{0.478619}\,n_{t-1} + \frac{\underset{(947.857352)}{0.866941}}{2(\underset{(7.607059)}{0.045159})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(801.569247)}{0.077667}}{2(\underset{(7.607059)}{0.045159})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.105549 | 697.168533 |
| rho_1 | 0.246211 | 361.613681 |
| rho_2 | 0.060730 | 98.211741 |
| phi_1 | 0.589309 | 76.765988 |
| phi_2 | 0.030801 | 1152.677294 |
| p0 | 5.517741 | 6432.648848 |
| n0 | 6.662755 | 7890.268578 |
| rho | 0.478619 | 15.501734 |
| phi_plus | 0.866941 | 947.857352 |
| phi_minus | 0.077667 | 801.569247 |
| sigma_p | 0.099267 | 0.761605 |
| sigma_n | 0.045159 | 7.607059 |

### Rank 14: Seed 41, Draw 37

- LogLik: `-157.215114`; AIC: `338.430229`; BIC: `378.877885`
- Max shape path: `1735.241427`; max implied variance: `15.920894`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1.418577)}{0.185173} + \underset{(12.712527)}{0.138210}\,\pi_t + \underset{(2.786035)}{0.092281}\,\pi_{t-1} + \underset{(9.846125)}{1.159452}\,SPF_t + \underset{(10.789035)}{-0.298861}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(17.476154)}{0.529318}\,\omega_{p,t} - \underset{(2.200142)}{0.066481}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(4.155671)}{1.828209} + \underset{(0.000357)}{0.189533}\,p_{t-1} + \frac{\underset{(13.616544)}{0.711513}}{2(\underset{(17.476154)}{0.529318})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(13.616899)}{0.792038}}{2(\underset{(17.476154)}{0.529318})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(3.846247)}{9.149534} + \underset{(0.000357)}{0.189533}\,n_{t-1} + \frac{\underset{(13.616544)}{0.711513}}{2(\underset{(2.200142)}{0.066481})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(13.616899)}{0.792038}}{2(\underset{(2.200142)}{0.066481})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.185173 | 1.418577 |
| rho_1 | 0.138210 | 12.712527 |
| rho_2 | 0.092281 | 2.786035 |
| phi_1 | 1.159452 | 9.846125 |
| phi_2 | -0.298861 | 10.789035 |
| p0 | 1.828209 | 4.155671 |
| n0 | 9.149534 | 3.846247 |
| rho | 0.189533 | 0.000357 |
| phi_plus | 0.711513 | 13.616544 |
| phi_minus | 0.792038 | 13.616899 |
| sigma_p | 0.529318 | 17.476154 |
| sigma_n | 0.066481 | 2.200142 |

### Rank 15: Seed 45, Draw 14

- LogLik: `-158.161416`; AIC: `340.322832`; BIC: `380.770489`
- Max shape path: `4828.251225`; max implied variance: `2.202414`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(1.632398)}{0.095748} + \underset{(1.064294)}{0.431278}\,\pi_t + \underset{(1.224278)}{0.059042}\,\pi_{t-1} + \underset{(3.355842)}{0.155632}\,SPF_t + \underset{(0.516458)}{0.331465}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000007)}{0.013621}\,\omega_{p,t} - \underset{(0.030987)}{0.155409}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(125.838288)}{9.041025} + \underset{(0.298421)}{0.504323}\,p_{t-1} + \frac{\underset{(0.487126)}{0.293058}}{2(\underset{(0.000007)}{0.013621})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.305122)}{0.095728}}{2(\underset{(0.000007)}{0.013621})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(4.163755)}{8.499491} + \underset{(0.298421)}{0.504323}\,n_{t-1} + \frac{\underset{(0.487126)}{0.293058}}{2(\underset{(0.030987)}{0.155409})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.305122)}{0.095728}}{2(\underset{(0.030987)}{0.155409})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.095748 | 1.632398 |
| rho_1 | 0.431278 | 1.064294 |
| rho_2 | 0.059042 | 1.224278 |
| phi_1 | 0.155632 | 3.355842 |
| phi_2 | 0.331465 | 0.516458 |
| p0 | 9.041025 | 125.838288 |
| n0 | 8.499491 | 4.163755 |
| rho | 0.504323 | 0.298421 |
| phi_plus | 0.293058 | 0.487126 |
| phi_minus | 0.095728 | 0.305122 |
| sigma_p | 0.013621 | 0.000007 |
| sigma_n | 0.155409 | 0.030987 |

### Rank 16: Seed 5, Draw 7

- LogLik: `-159.021889`; AIC: `342.043777`; BIC: `382.491434`
- Max shape path: `12730.655593`; max implied variance: `4.779221`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(31.280569)}{-0.031678} + \underset{(208.778541)}{0.265646}\,\pi_t + \underset{(185.517818)}{0.060554}\,\pi_{t-1} + \underset{(146.387685)}{0.388166}\,SPF_t + \underset{(141.537118)}{0.460330}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.002349)}{0.013609}\,\omega_{p,t} - \underset{(3.823173)}{0.093082}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1748.088904)}{1.605857} + \underset{(48.709613)}{0.456586}\,p_{t-1} + \frac{\underset{(87.958656)}{0.717412}}{2(\underset{(0.002349)}{0.013609})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(46.146226)}{0.258686}}{2(\underset{(0.002349)}{0.013609})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(844.656645)}{4.015641} + \underset{(48.709613)}{0.456586}\,n_{t-1} + \frac{\underset{(87.958656)}{0.717412}}{2(\underset{(3.823173)}{0.093082})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(46.146226)}{0.258686}}{2(\underset{(3.823173)}{0.093082})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.031678 | 31.280569 |
| rho_1 | 0.265646 | 208.778541 |
| rho_2 | 0.060554 | 185.517818 |
| phi_1 | 0.388166 | 146.387685 |
| phi_2 | 0.460330 | 141.537118 |
| p0 | 1.605857 | 1748.088904 |
| n0 | 4.015641 | 844.656645 |
| rho | 0.456586 | 48.709613 |
| phi_plus | 0.717412 | 87.958656 |
| phi_minus | 0.258686 | 46.146226 |
| sigma_p | 0.013609 | 0.002349 |
| sigma_n | 0.093082 | 3.823173 |

### Rank 17: Seed 21, Draw 24

- LogLik: `-160.459730`; AIC: `344.919460`; BIC: `385.367116`
- Max shape path: `212.717795`; max implied variance: `1.310455`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.691098)}{-0.348222} + \underset{(6.961471)}{0.600701}\,\pi_t + \underset{(12.347757)}{-0.155649}\,\pi_{t-1} + \underset{(52.312419)}{1.426854}\,SPF_t + \underset{(29.305178)}{-0.645501}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.066296)}{0.151343}\,\omega_{p,t} - \underset{(0.098460)}{0.029785}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(7.646418)}{5.037063} + \underset{(0.052628)}{0.879273}\,p_{t-1} + \frac{\underset{(0.029457)}{0.000852}}{2(\underset{(0.066296)}{0.151343})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.388497)}{0.022891}}{2(\underset{(0.066296)}{0.151343})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(38.984446)}{3.076017} + \underset{(0.052628)}{0.879273}\,n_{t-1} + \frac{\underset{(0.029457)}{0.000852}}{2(\underset{(0.098460)}{0.029785})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.388497)}{0.022891}}{2(\underset{(0.098460)}{0.029785})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.348222 | 0.691098 |
| rho_1 | 0.600701 | 6.961471 |
| rho_2 | -0.155649 | 12.347757 |
| phi_1 | 1.426854 | 52.312419 |
| phi_2 | -0.645501 | 29.305178 |
| p0 | 5.037063 | 7.646418 |
| n0 | 3.076017 | 38.984446 |
| rho | 0.879273 | 0.052628 |
| phi_plus | 0.000852 | 0.029457 |
| phi_minus | 0.022891 | 0.388497 |
| sigma_p | 0.151343 | 0.066296 |
| sigma_n | 0.029785 | 0.098460 |

### Rank 18: Seed 25, Draw 9

- LogLik: `-162.078426`; AIC: `348.156853`; BIC: `388.604509`
- Max shape path: `27336.120557`; max implied variance: `4.691223`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000013)}{0.177888} + \underset{(0.026595)}{0.325432}\,\pi_t + \underset{(0.010667)}{0.084989}\,\pi_{t-1} + \underset{(0.000291)}{0.154195}\,SPF_t + \underset{(0.003530)}{0.372911}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000013)}{0.009019}\,\omega_{p,t} - \underset{(0.006301)}{0.168809}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(3.164033)}{5.093219} + \underset{(0.015933)}{0.360895}\,p_{t-1} + \frac{\underset{(0.003786)}{0.496306}}{2(\underset{(0.000013)}{0.009019})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.293904)}{0.238244}}{2(\underset{(0.000013)}{0.009019})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.108069)}{5.494096} + \underset{(0.015933)}{0.360895}\,n_{t-1} + \frac{\underset{(0.003786)}{0.496306}}{2(\underset{(0.006301)}{0.168809})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.293904)}{0.238244}}{2(\underset{(0.006301)}{0.168809})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.177888 | 0.000013 |
| rho_1 | 0.325432 | 0.026595 |
| rho_2 | 0.084989 | 0.010667 |
| phi_1 | 0.154195 | 0.000291 |
| phi_2 | 0.372911 | 0.003530 |
| p0 | 5.093219 | 3.164033 |
| n0 | 5.494096 | 0.108069 |
| rho | 0.360895 | 0.015933 |
| phi_plus | 0.496306 | 0.003786 |
| phi_minus | 0.238244 | 0.293904 |
| sigma_p | 0.009019 | 0.000013 |
| sigma_n | 0.168809 | 0.006301 |

### Rank 19: Seed 10, Draw 26

- LogLik: `-163.378955`; AIC: `350.757910`; BIC: `391.205566`
- Max shape path: `2822.105088`; max implied variance: `3.372188`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000509)}{0.168269} + \underset{(0.001499)}{0.389252}\,\pi_t + \underset{(0.001304)}{0.229046}\,\pi_{t-1} + \underset{(0.001304)}{0.199950}\,SPF_t + \underset{(0.001400)}{-0.096090}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.022941}\,\omega_{p,t} - \underset{(0.000100)}{0.135106}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.001652)}{9.652240} + \underset{(0.000022)}{0.632068}\,p_{t-1} + \frac{\underset{(0.000002)}{0.258077}}{2(\underset{(0.000002)}{0.022941})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000508)}{0.148399}}{2(\underset{(0.000002)}{0.022941})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.001652)}{8.376026} + \underset{(0.000022)}{0.632068}\,n_{t-1} + \frac{\underset{(0.000002)}{0.258077}}{2(\underset{(0.000100)}{0.135106})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.000508)}{0.148399}}{2(\underset{(0.000100)}{0.135106})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.168269 | 0.000509 |
| rho_1 | 0.389252 | 0.001499 |
| rho_2 | 0.229046 | 0.001304 |
| phi_1 | 0.199950 | 0.001304 |
| phi_2 | -0.096090 | 0.001400 |
| p0 | 9.652240 | 0.001652 |
| n0 | 8.376026 | 0.001652 |
| rho | 0.632068 | 0.000022 |
| phi_plus | 0.258077 | 0.000002 |
| phi_minus | 0.148399 | 0.000508 |
| sigma_p | 0.022941 | 0.000002 |
| sigma_n | 0.135106 | 0.000100 |

### Rank 20: Seed 30, Draw 40

- LogLik: `-166.629932`; AIC: `357.259864`; BIC: `397.707520`
- Max shape path: `898.602978`; max implied variance: `1.918708`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.860309)}{0.202206} + \underset{(0.843477)}{0.424403}\,\pi_t + \underset{(0.492270)}{0.308242}\,\pi_{t-1} + \underset{(0.369743)}{0.072583}\,SPF_t + \underset{(0.684205)}{-0.002725}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000012)}{0.032692}\,\omega_{p,t} - \underset{(0.000012)}{0.032879}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.282099)}{0.289173} + \underset{(0.046148)}{0.943574}\,p_{t-1} + \frac{\underset{(0.072913)}{0.014748}}{2(\underset{(0.000012)}{0.032692})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.065381)}{0.076922}}{2(\underset{(0.000012)}{0.032692})^2}\,(u_{t-1}^-)^2,\\
n_t &= \underset{(0.364923)}{0.178557} + \underset{(0.046148)}{0.943574}\,n_{t-1} + \frac{\underset{(0.072913)}{0.014748}}{2(\underset{(0.000012)}{0.032879})^2}\,(u_{t-1}^+)^2 + \frac{\underset{(0.065381)}{0.076922}}{2(\underset{(0.000012)}{0.032879})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.202206 | 0.860309 |
| rho_1 | 0.424403 | 0.843477 |
| rho_2 | 0.308242 | 0.492270 |
| phi_1 | 0.072583 | 0.369743 |
| phi_2 | -0.002725 | 0.684205 |
| p0 | 0.289173 | 0.282099 |
| n0 | 0.178557 | 0.364923 |
| rho | 0.943574 | 0.046148 |
| phi_plus | 0.014748 | 0.072913 |
| phi_minus | 0.076922 | 0.065381 |
| sigma_p | 0.032692 | 0.000012 |
| sigma_n | 0.032879 | 0.000012 |
