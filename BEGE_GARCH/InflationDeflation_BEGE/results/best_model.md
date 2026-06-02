```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-06-02T11:45:13`
Total estimations: `8000`
Converged estimations: `7310`
Eligible estimations for best-model selection: `7273`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `20` admissible estimates by corrected log likelihood. Standard errors are shown below substituted equation coefficients in parentheses.

```{note}
Flagged 106 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## constant

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 22 | 15 | -157.812140 | 331.624281 | 358.589385 | 6312.732437 | 3.103930 | no | `computed` |
| 2 | 9 | 39 | -177.882535 | 371.765069 | 398.730173 | 125.826145 | 12.973910 | no | `computed` |
| 3 | 17 | 10 | -177.882535 | 371.765069 | 398.730174 | 125.832345 | 12.972768 | no | `computed` |
| 4 | 34 | 10 | -177.882535 | 371.765070 | 398.730174 | 125.813979 | 12.972114 | no | `computed` |
| 5 | 1 | 28 | -177.882535 | 371.765070 | 398.730174 | 125.819901 | 12.972392 | no | `computed` |
| 6 | 46 | 10 | -177.882535 | 371.765070 | 398.730174 | 125.838307 | 12.972599 | no | `computed` |
| 7 | 22 | 21 | -177.882535 | 371.765070 | 398.730174 | 125.811413 | 12.970894 | no | `computed` |
| 8 | 40 | 25 | -177.882535 | 371.765070 | 398.730174 | 125.879887 | 12.973129 | no | `computed` |
| 9 | 22 | 40 | -177.882535 | 371.765070 | 398.730174 | 125.818011 | 12.972290 | no | `computed` |
| 10 | 44 | 17 | -177.882535 | 371.765070 | 398.730174 | 125.814765 | 12.975424 | no | `computed` |
| 11 | 29 | 10 | -177.882535 | 371.765070 | 398.730174 | 125.839734 | 12.971690 | no | `computed` |
| 12 | 10 | 25 | -177.882535 | 371.765070 | 398.730174 | 125.894736 | 12.974168 | no | `computed` |
| 13 | 27 | 3 | -177.882535 | 371.765070 | 398.730175 | 125.882848 | 12.972095 | no | `computed` |
| 14 | 26 | 27 | -177.882535 | 371.765070 | 398.730175 | 125.820330 | 12.972146 | no | `computed` |
| 15 | 17 | 39 | -177.882535 | 371.765070 | 398.730175 | 125.891940 | 12.973209 | no | `computed` |
| 16 | 7 | 35 | -177.882535 | 371.765070 | 398.730175 | 125.755146 | 12.973078 | no | `computed` |
| 17 | 29 | 30 | -177.882535 | 371.765070 | 398.730175 | 125.841884 | 12.971348 | no | `computed` |
| 18 | 47 | 25 | -177.882535 | 371.765070 | 398.730175 | 125.795457 | 12.971014 | no | `computed` |
| 19 | 45 | 28 | -177.882535 | 371.765071 | 398.730175 | 125.886074 | 12.973609 | no | `computed` |
| 20 | 16 | 40 | -177.882535 | 371.765071 | 398.730175 | 125.882973 | 12.972418 | no | `computed` |

### Rank 1: Seed 22, Draw 15

- LogLik: `-157.812140`; AIC: `331.624281`; BIC: `358.589385`
- Max shape path: `6312.732437`; max implied variance: `3.103930`
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
u_t &= \underset{(0.156253)}{0.190782}\,\omega_{p,t} - \underset{(0.000087)}{0.018752}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.156115)}{4.756536} + \underset{(0.000274)}{0.775695}\,p_{t-1} + \frac{\underset{(0.000002)}{0.135188}}{2(\underset{(0.156253)}{0.190782})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.000000)}{9.503285} + \underset{(0.000273)}{0.553343}\,n_{t-1} + \frac{\underset{(0.000273)}{0.274879}}{2(\underset{(0.000087)}{0.018752})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 4.756536 | 0.156115 |
| n0 | 9.503285 | 0.000000 |
| rho_p | 0.775695 | 0.000274 |
| rho_n | 0.553343 | 0.000273 |
| phi_p_plus | 0.135188 | 0.000002 |
| phi_n_minus | 0.274879 | 0.000273 |
| sigma_p | 0.190782 | 0.156253 |
| sigma_n | 0.018752 | 0.000087 |

### Rank 2: Seed 9, Draw 39

- LogLik: `-177.882535`; AIC: `371.765069`; BIC: `398.730173`
- Max shape path: `125.826145`; max implied variance: `12.973910`
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
u_t &= \underset{(0.048995)}{0.141989}\,\omega_{p,t} - \underset{(0.669873)}{1.638761}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.578634)}{3.212923} + \underset{(0.100694)}{0.382743}\,p_{t-1} + \frac{\underset{(0.211396)}{0.862959}}{2(\underset{(0.048995)}{0.141989})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025656)}{0.048702} + \underset{(0.054374)}{0.101885}\,n_{t-1} + \frac{\underset{(0.160762)}{1.566022}}{2(\underset{(0.669873)}{1.638761})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.212923 | 1.578634 |
| n0 | 0.048702 | 0.025656 |
| rho_p | 0.382743 | 0.100694 |
| rho_n | 0.101885 | 0.054374 |
| phi_p_plus | 0.862959 | 0.211396 |
| phi_n_minus | 1.566022 | 0.160762 |
| sigma_p | 0.141989 | 0.048995 |
| sigma_n | 1.638761 | 0.669873 |

### Rank 3: Seed 17, Draw 10

- LogLik: `-177.882535`; AIC: `371.765069`; BIC: `398.730174`
- Max shape path: `125.832345`; max implied variance: `12.972768`
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
u_t &= \underset{(0.046144)}{0.141986}\,\omega_{p,t} - \underset{(0.660043)}{1.638802}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.386774)}{3.213698} + \underset{(0.100152)}{0.382658}\,p_{t-1} + \frac{\underset{(0.209102)}{0.863017}}{2(\underset{(0.046144)}{0.141986})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025481)}{0.048710} + \underset{(0.053581)}{0.101899}\,n_{t-1} + \frac{\underset{(0.161579)}{1.565882}}{2(\underset{(0.660043)}{1.638802})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.213698 | 1.386774 |
| n0 | 0.048710 | 0.025481 |
| rho_p | 0.382658 | 0.100152 |
| rho_n | 0.101899 | 0.053581 |
| phi_p_plus | 0.863017 | 0.209102 |
| phi_n_minus | 1.565882 | 0.161579 |
| sigma_p | 0.141986 | 0.046144 |
| sigma_n | 1.638802 | 0.660043 |

### Rank 4: Seed 34, Draw 10

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.813979`; max implied variance: `12.972114`
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
u_t &= \underset{(0.051542)}{0.142002}\,\omega_{p,t} - \underset{(0.679085)}{1.638479}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.649062)}{3.212323} + \underset{(0.100551)}{0.382723}\,p_{t-1} + \frac{\underset{(0.211256)}{0.863054}}{2(\underset{(0.051542)}{0.142002})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025827)}{0.048718} + \underset{(0.055358)}{0.101884}\,n_{t-1} + \frac{\underset{(0.162282)}{1.565800}}{2(\underset{(0.679085)}{1.638479})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.212323 | 1.649062 |
| n0 | 0.048718 | 0.025827 |
| rho_p | 0.382723 | 0.100551 |
| rho_n | 0.101884 | 0.055358 |
| phi_p_plus | 0.863054 | 0.211256 |
| phi_n_minus | 1.565800 | 0.162282 |
| sigma_p | 0.142002 | 0.051542 |
| sigma_n | 1.638479 | 0.679085 |

### Rank 5: Seed 1, Draw 28

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.819901`; max implied variance: `12.972392`
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
u_t &= \underset{(0.049872)}{0.141995}\,\omega_{p,t} - \underset{(0.670207)}{1.638416}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.579923)}{3.212419} + \underset{(0.100945)}{0.382775}\,p_{t-1} + \frac{\underset{(0.211572)}{0.862971}}{2(\underset{(0.049872)}{0.141995})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025660)}{0.048720} + \underset{(0.054820)}{0.101841}\,n_{t-1} + \frac{\underset{(0.161481)}{1.565833}}{2(\underset{(0.670207)}{1.638416})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.212419 | 1.579923 |
| n0 | 0.048720 | 0.025660 |
| rho_p | 0.382775 | 0.100945 |
| rho_n | 0.101841 | 0.054820 |
| phi_p_plus | 0.862971 | 0.211572 |
| phi_n_minus | 1.565833 | 0.161481 |
| sigma_p | 0.141995 | 0.049872 |
| sigma_n | 1.638416 | 0.670207 |

### Rank 6: Seed 46, Draw 10

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.838307`; max implied variance: `12.972599`
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
u_t &= \underset{(0.050454)}{0.141993}\,\omega_{p,t} - \underset{(0.673849)}{1.638243}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.644961)}{3.213098} + \underset{(0.101337)}{0.382681}\,p_{t-1} + \frac{\underset{(0.212415)}{0.863133}}{2(\underset{(0.050454)}{0.141993})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025758)}{0.048720} + \underset{(0.054645)}{0.101842}\,n_{t-1} + \frac{\underset{(0.161399)}{1.565867}}{2(\underset{(0.673849)}{1.638243})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.213098 | 1.644961 |
| n0 | 0.048720 | 0.025758 |
| rho_p | 0.382681 | 0.101337 |
| rho_n | 0.101842 | 0.054645 |
| phi_p_plus | 0.863133 | 0.212415 |
| phi_n_minus | 1.565867 | 0.161399 |
| sigma_p | 0.141993 | 0.050454 |
| sigma_n | 1.638243 | 0.673849 |

### Rank 7: Seed 22, Draw 21

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.811413`; max implied variance: `12.970894`
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
u_t &= \underset{(0.049537)}{0.141990}\,\omega_{p,t} - \underset{(0.664931)}{1.638507}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.565874)}{3.213024} + \underset{(0.100869)}{0.382744}\,p_{t-1} + \frac{\underset{(0.211125)}{0.862864}}{2(\underset{(0.049537)}{0.141990})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025548)}{0.048722} + \underset{(0.054303)}{0.101896}\,n_{t-1} + \frac{\underset{(0.161687)}{1.565647}}{2(\underset{(0.664931)}{1.638507})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.213024 | 1.565874 |
| n0 | 0.048722 | 0.025548 |
| rho_p | 0.382744 | 0.100869 |
| rho_n | 0.101896 | 0.054303 |
| phi_p_plus | 0.862864 | 0.211125 |
| phi_n_minus | 1.565647 | 0.161687 |
| sigma_p | 0.141990 | 0.049537 |
| sigma_n | 1.638507 | 0.664931 |

### Rank 8: Seed 40, Draw 25

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.879887`; max implied variance: `12.973129`
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
u_t &= \underset{(0.049358)}{0.141963}\,\omega_{p,t} - \underset{(0.661935)}{1.638089}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.582452)}{3.213968} + \underset{(0.101161)}{0.382727}\,p_{t-1} + \frac{\underset{(0.209996)}{0.863027}}{2(\underset{(0.049358)}{0.141963})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025483)}{0.048730} + \underset{(0.053877)}{0.101840}\,n_{t-1} + \frac{\underset{(0.161353)}{1.565932}}{2(\underset{(0.661935)}{1.638089})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.213968 | 1.582452 |
| n0 | 0.048730 | 0.025483 |
| rho_p | 0.382727 | 0.101161 |
| rho_n | 0.101840 | 0.053877 |
| phi_p_plus | 0.863027 | 0.209996 |
| phi_n_minus | 1.565932 | 0.161353 |
| sigma_p | 0.141963 | 0.049358 |
| sigma_n | 1.638089 | 0.661935 |

### Rank 9: Seed 22, Draw 40

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.818011`; max implied variance: `12.972290`
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
u_t &= \underset{(0.051254)}{0.142003}\,\omega_{p,t} - \underset{(0.676033)}{1.638920}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.648305)}{3.212207} + \underset{(0.100855)}{0.382746}\,p_{t-1} + \frac{\underset{(0.211497)}{0.863081}}{2(\underset{(0.051254)}{0.142003})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025789)}{0.048706} + \underset{(0.054800)}{0.101900}\,n_{t-1} + \frac{\underset{(0.161503)}{1.565814}}{2(\underset{(0.676033)}{1.638920})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.212207 | 1.648305 |
| n0 | 0.048706 | 0.025789 |
| rho_p | 0.382746 | 0.100855 |
| rho_n | 0.101900 | 0.054800 |
| phi_p_plus | 0.863081 | 0.211497 |
| phi_n_minus | 1.565814 | 0.161503 |
| sigma_p | 0.142003 | 0.051254 |
| sigma_n | 1.638920 | 0.676033 |

### Rank 10: Seed 44, Draw 17

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.814765`; max implied variance: `12.975424`
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
u_t &= \underset{(0.049394)}{0.141989}\,\omega_{p,t} - \underset{(0.671885)}{1.638977}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.551921)}{3.212607} + \underset{(0.100869)}{0.382795}\,p_{t-1} + \frac{\underset{(0.210720)}{0.862855}}{2(\underset{(0.049394)}{0.141989})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025662)}{0.048704} + \underset{(0.054675)}{0.101859}\,n_{t-1} + \frac{\underset{(0.161385)}{1.566203}}{2(\underset{(0.671885)}{1.638977})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.212607 | 1.551921 |
| n0 | 0.048704 | 0.025662 |
| rho_p | 0.382795 | 0.100869 |
| rho_n | 0.101859 | 0.054675 |
| phi_p_plus | 0.862855 | 0.210720 |
| phi_n_minus | 1.566203 | 0.161385 |
| sigma_p | 0.141989 | 0.049394 |
| sigma_n | 1.638977 | 0.671885 |

### Rank 11: Seed 29, Draw 10

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.839734`; max implied variance: `12.971690`
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
u_t &= \underset{(0.045264)}{0.141995}\,\omega_{p,t} - \underset{(0.671480)}{1.638380}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.354349)}{3.213562} + \underset{(0.104156)}{0.382601}\,p_{t-1} + \frac{\underset{(0.214013)}{0.863214}}{2(\underset{(0.045264)}{0.141995})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025831)}{0.048720} + \underset{(0.053251)}{0.101886}\,n_{t-1} + \frac{\underset{(0.162643)}{1.565755}}{2(\underset{(0.671480)}{1.638380})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.213562 | 1.354349 |
| n0 | 0.048720 | 0.025831 |
| rho_p | 0.382601 | 0.104156 |
| rho_n | 0.101886 | 0.053251 |
| phi_p_plus | 0.863214 | 0.214013 |
| phi_n_minus | 1.565755 | 0.162643 |
| sigma_p | 0.141995 | 0.045264 |
| sigma_n | 1.638380 | 0.671480 |

### Rank 12: Seed 10, Draw 25

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.894736`; max implied variance: `12.974168`
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
u_t &= \underset{(0.046238)}{0.141957}\,\omega_{p,t} - \underset{(0.655420)}{1.638633}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.389602)}{3.214314} + \underset{(0.100757)}{0.382715}\,p_{t-1} + \frac{\underset{(0.210091)}{0.863068}}{2(\underset{(0.046238)}{0.141957})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025445)}{0.048720} + \underset{(0.053668)}{0.101823}\,n_{t-1} + \frac{\underset{(0.160913)}{1.566053}}{2(\underset{(0.655420)}{1.638633})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.214314 | 1.389602 |
| n0 | 0.048720 | 0.025445 |
| rho_p | 0.382715 | 0.100757 |
| rho_n | 0.101823 | 0.053668 |
| phi_p_plus | 0.863068 | 0.210091 |
| phi_n_minus | 1.566053 | 0.160913 |
| sigma_p | 0.141957 | 0.046238 |
| sigma_n | 1.638633 | 0.655420 |

### Rank 13: Seed 27, Draw 3

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730175`
- Max shape path: `125.882848`; max implied variance: `12.972095`
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
u_t &= \underset{(0.049810)}{0.141956}\,\omega_{p,t} - \underset{(0.668131)}{1.638693}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.586545)}{3.214762} + \underset{(0.100985)}{0.382682}\,p_{t-1} + \frac{\underset{(0.211214)}{0.862978}}{2(\underset{(0.049810)}{0.141956})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025629)}{0.048715} + \underset{(0.054421)}{0.101895}\,n_{t-1} + \frac{\underset{(0.161102)}{1.565798}}{2(\underset{(0.668131)}{1.638693})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.214762 | 1.586545 |
| n0 | 0.048715 | 0.025629 |
| rho_p | 0.382682 | 0.100985 |
| rho_n | 0.101895 | 0.054421 |
| phi_p_plus | 0.862978 | 0.211214 |
| phi_n_minus | 1.565798 | 0.161102 |
| sigma_p | 0.141956 | 0.049810 |
| sigma_n | 1.638693 | 0.668131 |

### Rank 14: Seed 26, Draw 27

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730175`
- Max shape path: `125.820330`; max implied variance: `12.972146`
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
u_t &= \underset{(0.049005)}{0.141993}\,\omega_{p,t} - \underset{(0.656155)}{1.637972}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.531213)}{3.212403} + \underset{(0.100656)}{0.382785}\,p_{t-1} + \frac{\underset{(0.210545)}{0.862952}}{2(\underset{(0.049005)}{0.141993})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025503)}{0.048734} + \underset{(0.053792)}{0.101861}\,n_{t-1} + \frac{\underset{(0.161361)}{1.565806}}{2(\underset{(0.656155)}{1.637972})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.212403 | 1.531213 |
| n0 | 0.048734 | 0.025503 |
| rho_p | 0.382785 | 0.100656 |
| rho_n | 0.101861 | 0.053792 |
| phi_p_plus | 0.862952 | 0.210545 |
| phi_n_minus | 1.565806 | 0.161361 |
| sigma_p | 0.141993 | 0.049005 |
| sigma_n | 1.637972 | 0.656155 |

### Rank 15: Seed 17, Draw 39

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730175`
- Max shape path: `125.891940`; max implied variance: `12.973209`
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
u_t &= \underset{(0.048350)}{0.141965}\,\omega_{p,t} - \underset{(0.663982)}{1.638986}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.502881)}{3.214351} + \underset{(0.100719)}{0.382659}\,p_{t-1} + \frac{\underset{(0.210543)}{0.863176}}{2(\underset{(0.048350)}{0.141965})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025561)}{0.048704} + \underset{(0.054310)}{0.101893}\,n_{t-1} + \frac{\underset{(0.161147)}{1.565933}}{2(\underset{(0.663982)}{1.638986})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.214351 | 1.502881 |
| n0 | 0.048704 | 0.025561 |
| rho_p | 0.382659 | 0.100719 |
| rho_n | 0.101893 | 0.054310 |
| phi_p_plus | 0.863176 | 0.210543 |
| phi_n_minus | 1.565933 | 0.161147 |
| sigma_p | 0.141965 | 0.048350 |
| sigma_n | 1.638986 | 0.663982 |

### Rank 16: Seed 7, Draw 35

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730175`
- Max shape path: `125.755146`; max implied variance: `12.973078`
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
u_t &= \underset{(0.013443)}{0.142024}\,\omega_{p,t} - \underset{(0.077232)}{1.638557}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.186457)}{3.211421} + \underset{(0.000501)}{0.382762}\,p_{t-1} + \frac{\underset{(0.001887)}{0.862890}}{2(\underset{(0.013443)}{0.142024})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.020448)}{0.048715} + \underset{(0.029183)}{0.101870}\,n_{t-1} + \frac{\underset{(0.050317)}{1.565918}}{2(\underset{(0.077232)}{1.638557})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.211421 | 0.186457 |
| n0 | 0.048715 | 0.020448 |
| rho_p | 0.382762 | 0.000501 |
| rho_n | 0.101870 | 0.029183 |
| phi_p_plus | 0.862890 | 0.001887 |
| phi_n_minus | 1.565918 | 0.050317 |
| sigma_p | 0.142024 | 0.013443 |
| sigma_n | 1.638557 | 0.077232 |

### Rank 17: Seed 29, Draw 30

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730175`
- Max shape path: `125.841884`; max implied variance: `12.971348`
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
u_t &= \underset{(0.041354)}{0.141969}\,\omega_{p,t} - \underset{(0.622528)}{1.638197}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.066656)}{3.213151} + \underset{(0.102063)}{0.382816}\,p_{t-1} + \frac{\underset{(0.211218)}{0.862786}}{2(\underset{(0.041354)}{0.141969})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025065)}{0.048731} + \underset{(0.052414)}{0.101891}\,n_{t-1} + \frac{\underset{(0.159968)}{1.565704}}{2(\underset{(0.622528)}{1.638197})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.213151 | 1.066656 |
| n0 | 0.048731 | 0.025065 |
| rho_p | 0.382816 | 0.102063 |
| rho_n | 0.101891 | 0.052414 |
| phi_p_plus | 0.862786 | 0.211218 |
| phi_n_minus | 1.565704 | 0.159968 |
| sigma_p | 0.141969 | 0.041354 |
| sigma_n | 1.638197 | 0.622528 |

### Rank 18: Seed 47, Draw 25

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730175`
- Max shape path: `125.795457`; max implied variance: `12.971014`
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
u_t &= \underset{(0.048857)}{0.142017}\,\omega_{p,t} - \underset{(0.682077)}{1.638265}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.681622)}{3.212083} + \underset{(0.101978)}{0.382679}\,p_{t-1} + \frac{\underset{(0.217108)}{0.863132}}{2(\underset{(0.048857)}{0.142017})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025832)}{0.048721} + \underset{(0.054024)}{0.101894}\,n_{t-1} + \frac{\underset{(0.163209)}{1.565668}}{2(\underset{(0.682077)}{1.638265})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.212083 | 1.681622 |
| n0 | 0.048721 | 0.025832 |
| rho_p | 0.382679 | 0.101978 |
| rho_n | 0.101894 | 0.054024 |
| phi_p_plus | 0.863132 | 0.217108 |
| phi_n_minus | 1.565668 | 0.163209 |
| sigma_p | 0.142017 | 0.048857 |
| sigma_n | 1.638265 | 0.682077 |

### Rank 19: Seed 45, Draw 28

- LogLik: `-177.882535`; AIC: `371.765071`; BIC: `398.730175`
- Max shape path: `125.886074`; max implied variance: `12.973609`
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
u_t &= \underset{(0.051185)}{0.141963}\,\omega_{p,t} - \underset{(0.678046)}{1.637972}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.652083)}{3.213669} + \underset{(0.101493)}{0.382747}\,p_{t-1} + \frac{\underset{(0.212717)}{0.863063}}{2(\underset{(0.051185)}{0.141963})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025834)}{0.048725} + \underset{(0.054721)}{0.101803}\,n_{t-1} + \frac{\underset{(0.160470)}{1.565995}}{2(\underset{(0.678046)}{1.637972})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.213669 | 1.652083 |
| n0 | 0.048725 | 0.025834 |
| rho_p | 0.382747 | 0.101493 |
| rho_n | 0.101803 | 0.054721 |
| phi_p_plus | 0.863063 | 0.212717 |
| phi_n_minus | 1.565995 | 0.160470 |
| sigma_p | 0.141963 | 0.051185 |
| sigma_n | 1.637972 | 0.678046 |

### Rank 20: Seed 16, Draw 40

- LogLik: `-177.882535`; AIC: `371.765071`; BIC: `398.730175`
- Max shape path: `125.882973`; max implied variance: `12.972418`
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
u_t &= \underset{(0.049329)}{0.141957}\,\omega_{p,t} - \underset{(0.667705)}{1.638415}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.560573)}{3.214421} + \underset{(0.100972)}{0.382737}\,p_{t-1} + \frac{\underset{(0.211171)}{0.862962}}{2(\underset{(0.049329)}{0.141957})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.025601)}{0.048736} + \underset{(0.054455)}{0.101881}\,n_{t-1} + \frac{\underset{(0.160854)}{1.565833}}{2(\underset{(0.667705)}{1.638415})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| p0 | 3.214421 | 1.560573 |
| n0 | 0.048736 | 0.025601 |
| rho_p | 0.382737 | 0.100972 |
| rho_n | 0.101881 | 0.054455 |
| phi_p_plus | 0.862962 | 0.211171 |
| phi_n_minus | 1.565833 | 0.160854 |
| sigma_p | 0.141957 | 0.049329 |
| sigma_n | 1.638415 | 0.667705 |

## ARX(1,1)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 3 | 40 | -157.746763 | 337.493526 | 374.570544 | 509.003819 | 12.048805 | no | `computed` |
| 2 | 6 | 34 | -170.519038 | 363.038075 | 400.115094 | 60.171153 | 9.274512 | no | `computed` |
| 3 | 38 | 24 | -170.519039 | 363.038078 | 400.115096 | 60.208945 | 9.271182 | no | `computed` |
| 4 | 4 | 28 | -170.519039 | 363.038079 | 400.115097 | 60.205258 | 9.275218 | no | `computed` |
| 5 | 37 | 32 | -170.519040 | 363.038079 | 400.115098 | 60.163204 | 9.269005 | no | `computed` |
| 6 | 36 | 18 | -170.519040 | 363.038081 | 400.115099 | 60.203985 | 9.269312 | no | `computed` |
| 7 | 15 | 21 | -170.519041 | 363.038083 | 400.115101 | 60.190827 | 9.279012 | no | `computed` |
| 8 | 20 | 33 | -170.519041 | 363.038083 | 400.115101 | 60.145937 | 9.273568 | no | `computed` |
| 9 | 13 | 20 | -170.519042 | 363.038084 | 400.115102 | 60.075727 | 9.269097 | no | `computed` |
| 10 | 2 | 25 | -170.519042 | 363.038085 | 400.115103 | 60.229962 | 9.282599 | no | `computed` |
| 11 | 30 | 31 | -170.519043 | 363.038085 | 400.115104 | 60.150935 | 9.282664 | no | `computed` |
| 12 | 11 | 17 | -170.519043 | 363.038086 | 400.115104 | 60.166303 | 9.262445 | no | `computed` |
| 13 | 8 | 37 | -170.519043 | 363.038087 | 400.115105 | 60.069354 | 9.271536 | no | `computed` |
| 14 | 18 | 23 | -170.519044 | 363.038087 | 400.115105 | 60.166748 | 9.273158 | no | `computed` |
| 15 | 15 | 12 | -170.519044 | 363.038087 | 400.115106 | 60.233294 | 9.278171 | no | `computed` |
| 16 | 13 | 13 | -170.519044 | 363.038089 | 400.115107 | 60.084779 | 9.273729 | no | `computed` |
| 17 | 4 | 32 | -170.519045 | 363.038089 | 400.115107 | 60.262414 | 9.282285 | no | `computed` |
| 18 | 1 | 24 | -170.519045 | 363.038090 | 400.115108 | 60.249027 | 9.260523 | no | `computed` |
| 19 | 30 | 7 | -170.519045 | 363.038090 | 400.115108 | 60.171451 | 9.292653 | no | `computed` |
| 20 | 41 | 30 | -170.519045 | 363.038090 | 400.115108 | 60.281074 | 9.266821 | no | `computed` |

### Rank 1: Seed 3, Draw 40

- LogLik: `-157.746763`; AIC: `337.493526`; BIC: `374.570544`
- Max shape path: `509.003819`; max implied variance: `12.048805`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.052645} + \underset{(0.000002)}{0.379824}\,\pi_t + \underset{(0.000002)}{0.615393}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.048940}\,\omega_{p,t} - \underset{(0.000002)}{0.160659}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000001)}{9.176419} + \underset{(0.000002)}{0.690579}\,p_{t-1} + \frac{\underset{(0.000002)}{0.482635}}{2(\underset{(0.000002)}{0.048940})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.000000)}{2.397874} + \underset{(0.000001)}{0.000000}\,n_{t-1} + \frac{\underset{(0.000001)}{1.406178}}{2(\underset{(0.000002)}{0.160659})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.052645 | 0.000002 |
| rho_1 | 0.379824 | 0.000002 |
| phi_1 | 0.615393 | 0.000002 |
| p0 | 9.176419 | 0.000001 |
| n0 | 2.397874 | 0.000000 |
| rho_p | 0.690579 | 0.000002 |
| rho_n | 0.000000 | 0.000001 |
| phi_p_plus | 0.482635 | 0.000002 |
| phi_n_minus | 1.406178 | 0.000001 |
| sigma_p | 0.048940 | 0.000002 |
| sigma_n | 0.160659 | 0.000002 |

### Rank 2: Seed 6, Draw 34

- LogLik: `-170.519038`; AIC: `363.038075`; BIC: `400.115094`
- Max shape path: `60.171153`; max implied variance: `9.274512`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.093810)}{0.132322} + \underset{(0.094502)}{0.292944}\,\pi_t + \underset{(0.160645)}{0.624955}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.061373)}{0.158209}\,\omega_{p,t} - \underset{(0.481497)}{0.846063}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.341897)}{1.966086} + \underset{(0.202849)}{0.560865}\,p_{t-1} + \frac{\underset{(0.322698)}{0.633747}}{2(\underset{(0.061373)}{0.158209})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.077126)}{0.077597} + \underset{(0.050332)}{0.025731}\,n_{t-1} + \frac{\underset{(0.842231)}{1.059762}}{2(\underset{(0.481497)}{0.846063})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132322 | 0.093810 |
| rho_1 | 0.292944 | 0.094502 |
| phi_1 | 0.624955 | 0.160645 |
| p0 | 1.966086 | 1.341897 |
| n0 | 0.077597 | 0.077126 |
| rho_p | 0.560865 | 0.202849 |
| rho_n | 0.025731 | 0.050332 |
| phi_p_plus | 0.633747 | 0.322698 |
| phi_n_minus | 1.059762 | 0.842231 |
| sigma_p | 0.158209 | 0.061373 |
| sigma_n | 0.846063 | 0.481497 |

### Rank 3: Seed 38, Draw 24

- LogLik: `-170.519039`; AIC: `363.038078`; BIC: `400.115096`
- Max shape path: `60.208945`; max implied variance: `9.271182`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.098874)}{0.132351} + \underset{(0.094852)}{0.292936}\,\pi_t + \underset{(0.165976)}{0.624945}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.067176)}{0.158192}\,\omega_{p,t} - \underset{(0.544887)}{0.845739}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.032835)}{1.967814} + \underset{(0.183911)}{0.560661}\,p_{t-1} + \frac{\underset{(0.295716)}{0.634042}}{2(\underset{(0.067176)}{0.158192})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.082523)}{0.077643} + \underset{(0.050520)}{0.025697}\,n_{t-1} + \frac{\underset{(0.917469)}{1.059366}}{2(\underset{(0.544887)}{0.845739})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132351 | 0.098874 |
| rho_1 | 0.292936 | 0.094852 |
| phi_1 | 0.624945 | 0.165976 |
| p0 | 1.967814 | 1.032835 |
| n0 | 0.077643 | 0.082523 |
| rho_p | 0.560661 | 0.183911 |
| rho_n | 0.025697 | 0.050520 |
| phi_p_plus | 0.634042 | 0.295716 |
| phi_n_minus | 1.059366 | 0.917469 |
| sigma_p | 0.158192 | 0.067176 |
| sigma_n | 0.845739 | 0.544887 |

### Rank 4: Seed 4, Draw 28

- LogLik: `-170.519039`; AIC: `363.038079`; BIC: `400.115097`
- Max shape path: `60.205258`; max implied variance: `9.275218`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090404)}{0.132252} + \underset{(0.105034)}{0.292866}\,\pi_t + \underset{(0.192805)}{0.625090}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.047242)}{0.158150}\,\omega_{p,t} - \underset{(0.263965)}{0.845887}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.733378)}{1.966445} + \underset{(0.293010)}{0.560990}\,p_{t-1} + \frac{\underset{(0.383547)}{0.633611}}{2(\underset{(0.047242)}{0.158150})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.053287)}{0.077669} + \underset{(0.120934)}{0.025601}\,n_{t-1} + \frac{\underset{(0.383914)}{1.059862}}{2(\underset{(0.263965)}{0.845887})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132252 | 0.090404 |
| rho_1 | 0.292866 | 0.105034 |
| phi_1 | 0.625090 | 0.192805 |
| p0 | 1.966445 | 0.733378 |
| n0 | 0.077669 | 0.053287 |
| rho_p | 0.560990 | 0.293010 |
| rho_n | 0.025601 | 0.120934 |
| phi_p_plus | 0.633611 | 0.383547 |
| phi_n_minus | 1.059862 | 0.383914 |
| sigma_p | 0.158150 | 0.047242 |
| sigma_n | 0.845887 | 0.263965 |

### Rank 5: Seed 37, Draw 32

- LogLik: `-170.519040`; AIC: `363.038079`; BIC: `400.115098`
- Max shape path: `60.163204`; max implied variance: `9.269005`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.096275)}{0.132291} + \underset{(0.093108)}{0.292969}\,\pi_t + \underset{(0.162162)}{0.624927}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.064399)}{0.158217}\,\omega_{p,t} - \underset{(0.496328)}{0.846327}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.148086)}{1.965564} + \underset{(0.179413)}{0.560926}\,p_{t-1} + \frac{\underset{(0.283797)}{0.633698}}{2(\underset{(0.064399)}{0.158217})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.079864)}{0.077619} + \underset{(0.051065)}{0.025908}\,n_{t-1} + \frac{\underset{(0.861611)}{1.059114}}{2(\underset{(0.496328)}{0.846327})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132291 | 0.096275 |
| rho_1 | 0.292969 | 0.093108 |
| phi_1 | 0.624927 | 0.162162 |
| p0 | 1.965564 | 1.148086 |
| n0 | 0.077619 | 0.079864 |
| rho_p | 0.560926 | 0.179413 |
| rho_n | 0.025908 | 0.051065 |
| phi_p_plus | 0.633698 | 0.283797 |
| phi_n_minus | 1.059114 | 0.861611 |
| sigma_p | 0.158217 | 0.064399 |
| sigma_n | 0.846327 | 0.496328 |

### Rank 6: Seed 36, Draw 18

- LogLik: `-170.519040`; AIC: `363.038081`; BIC: `400.115099`
- Max shape path: `60.203985`; max implied variance: `9.269312`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.097340)}{0.132362} + \underset{(0.095545)}{0.292954}\,\pi_t + \underset{(0.166958)}{0.624896}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.073813)}{0.158211}\,\omega_{p,t} - \underset{(0.508609)}{0.846259}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.631114)}{1.967309} + \underset{(0.191653)}{0.560660}\,p_{t-1} + \frac{\underset{(0.306330)}{0.634145}}{2(\underset{(0.073813)}{0.158211})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.083029)}{0.077595} + \underset{(0.051298)}{0.025646}\,n_{t-1} + \frac{\underset{(0.912898)}{1.059142}}{2(\underset{(0.508609)}{0.846259})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132362 | 0.097340 |
| rho_1 | 0.292954 | 0.095545 |
| phi_1 | 0.624896 | 0.166958 |
| p0 | 1.967309 | 1.631114 |
| n0 | 0.077595 | 0.083029 |
| rho_p | 0.560660 | 0.191653 |
| rho_n | 0.025646 | 0.051298 |
| phi_p_plus | 0.634145 | 0.306330 |
| phi_n_minus | 1.059142 | 0.912898 |
| sigma_p | 0.158211 | 0.073813 |
| sigma_n | 0.846259 | 0.508609 |

### Rank 7: Seed 15, Draw 21

- LogLik: `-170.519041`; AIC: `363.038083`; BIC: `400.115101`
- Max shape path: `60.190827`; max implied variance: `9.279012`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.099851)}{0.132290} + \underset{(0.085434)}{0.293055}\,\pi_t + \underset{(0.158959)}{0.624828}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.056850)}{0.158161}\,\omega_{p,t} - \underset{(0.401048)}{0.845454}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.992109)}{1.966500} + \underset{(0.225955)}{0.560917}\,p_{t-1} + \frac{\underset{(0.342289)}{0.633522}}{2(\underset{(0.056850)}{0.158161})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.073197)}{0.077708} + \underset{(0.056360)}{0.025702}\,n_{t-1} + \frac{\underset{(0.864824)}{1.060319}}{2(\underset{(0.401048)}{0.845454})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132290 | 0.099851 |
| rho_1 | 0.293055 | 0.085434 |
| phi_1 | 0.624828 | 0.158959 |
| p0 | 1.966500 | 0.992109 |
| n0 | 0.077708 | 0.073197 |
| rho_p | 0.560917 | 0.225955 |
| rho_n | 0.025702 | 0.056360 |
| phi_p_plus | 0.633522 | 0.342289 |
| phi_n_minus | 1.060319 | 0.864824 |
| sigma_p | 0.158161 | 0.056850 |
| sigma_n | 0.845454 | 0.401048 |

### Rank 8: Seed 20, Draw 33

- LogLik: `-170.519041`; AIC: `363.038083`; BIC: `400.115101`
- Max shape path: `60.145937`; max implied variance: `9.273568`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.097622)}{0.132395} + \underset{(0.095008)}{0.293075}\,\pi_t + \underset{(0.137687)}{0.624729}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.060207)}{0.158258}\,\omega_{p,t} - \underset{(0.495615)}{0.846106}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.122470)}{1.965579} + \underset{(0.167631)}{0.560724}\,p_{t-1} + \frac{\underset{(0.266751)}{0.633884}}{2(\underset{(0.060207)}{0.158258})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.077991)}{0.077627} + \underset{(0.053200)}{0.025946}\,n_{t-1} + \frac{\underset{(0.843510)}{1.059639}}{2(\underset{(0.495615)}{0.846106})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132395 | 0.097622 |
| rho_1 | 0.293075 | 0.095008 |
| phi_1 | 0.624729 | 0.137687 |
| p0 | 1.965579 | 1.122470 |
| n0 | 0.077627 | 0.077991 |
| rho_p | 0.560724 | 0.167631 |
| rho_n | 0.025946 | 0.053200 |
| phi_p_plus | 0.633884 | 0.266751 |
| phi_n_minus | 1.059639 | 0.843510 |
| sigma_p | 0.158258 | 0.060207 |
| sigma_n | 0.846106 | 0.495615 |

### Rank 9: Seed 13, Draw 20

- LogLik: `-170.519042`; AIC: `363.038084`; BIC: `400.115102`
- Max shape path: `60.075727`; max implied variance: `9.269097`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.095952)}{0.132379} + \underset{(0.092217)}{0.293070}\,\pi_t + \underset{(0.160188)}{0.624725}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.065300)}{0.158283}\,\omega_{p,t} - \underset{(0.514747)}{0.845897}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.215259)}{1.963185} + \underset{(0.179794)}{0.561096}\,p_{t-1} + \frac{\underset{(0.282979)}{0.633272}}{2(\underset{(0.065300)}{0.158283})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.082695)}{0.077693} + \underset{(0.050231)}{0.025536}\,n_{t-1} + \frac{\underset{(0.926812)}{1.059116}}{2(\underset{(0.514747)}{0.845897})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132379 | 0.095952 |
| rho_1 | 0.293070 | 0.092217 |
| phi_1 | 0.624725 | 0.160188 |
| p0 | 1.963185 | 1.215259 |
| n0 | 0.077693 | 0.082695 |
| rho_p | 0.561096 | 0.179794 |
| rho_n | 0.025536 | 0.050231 |
| phi_p_plus | 0.633272 | 0.282979 |
| phi_n_minus | 1.059116 | 0.926812 |
| sigma_p | 0.158283 | 0.065300 |
| sigma_n | 0.845897 | 0.514747 |

### Rank 10: Seed 2, Draw 25

- LogLik: `-170.519042`; AIC: `363.038085`; BIC: `400.115103`
- Max shape path: `60.229962`; max implied variance: `9.282599`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.365034)}{0.132260} + \underset{(0.173347)}{0.292896}\,\pi_t + \underset{(0.022167)}{0.625049}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.004760)}{0.158143}\,\omega_{p,t} - \underset{(4.487847)}{0.846870}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(9.390642)}{1.967881} + \underset{(1.695509)}{0.560842}\,p_{t-1} + \frac{\underset{(2.710792)}{0.633824}}{2(\underset{(0.004760)}{0.158143})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.607310)}{0.077494} + \underset{(0.258912)}{0.025678}\,n_{t-1} + \frac{\underset{(8.746005)}{1.060733}}{2(\underset{(4.487847)}{0.846870})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132260 | 0.365034 |
| rho_1 | 0.292896 | 0.173347 |
| phi_1 | 0.625049 | 0.022167 |
| p0 | 1.967881 | 9.390642 |
| n0 | 0.077494 | 0.607310 |
| rho_p | 0.560842 | 1.695509 |
| rho_n | 0.025678 | 0.258912 |
| phi_p_plus | 0.633824 | 2.710792 |
| phi_n_minus | 1.060733 | 8.746005 |
| sigma_p | 0.158143 | 0.004760 |
| sigma_n | 0.846870 | 4.487847 |

### Rank 11: Seed 30, Draw 31

- LogLik: `-170.519043`; AIC: `363.038085`; BIC: `400.115104`
- Max shape path: `60.150935`; max implied variance: `9.282664`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.096365)}{0.132333} + \underset{(0.093097)}{0.292977}\,\pi_t + \underset{(0.160872)}{0.624844}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.065886)}{0.158284}\,\omega_{p,t} - \underset{(0.539709)}{0.846000}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.346389)}{1.965587} + \underset{(0.193735)}{0.560601}\,p_{t-1} + \frac{\underset{(0.305787)}{0.634152}}{2(\underset{(0.065886)}{0.158284})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.085147)}{0.077637} + \underset{(0.051681)}{0.025710}\,n_{t-1} + \frac{\underset{(0.926628)}{1.060748}}{2(\underset{(0.539709)}{0.846000})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132333 | 0.096365 |
| rho_1 | 0.292977 | 0.093097 |
| phi_1 | 0.624844 | 0.160872 |
| p0 | 1.965587 | 1.346389 |
| n0 | 0.077637 | 0.085147 |
| rho_p | 0.560601 | 0.193735 |
| rho_n | 0.025710 | 0.051681 |
| phi_p_plus | 0.634152 | 0.305787 |
| phi_n_minus | 1.060748 | 0.926628 |
| sigma_p | 0.158284 | 0.065886 |
| sigma_n | 0.846000 | 0.539709 |

### Rank 12: Seed 11, Draw 17

- LogLik: `-170.519043`; AIC: `363.038086`; BIC: `400.115104`
- Max shape path: `60.166303`; max implied variance: `9.262445`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.096217)}{0.132429} + \underset{(0.093051)}{0.292919}\,\pi_t + \underset{(0.160060)}{0.624886}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.066543)}{0.158228}\,\omega_{p,t} - \underset{(0.535814)}{0.845516}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.369127)}{1.966298} + \underset{(0.193507)}{0.560834}\,p_{t-1} + \frac{\underset{(0.305597)}{0.633880}}{2(\underset{(0.066543)}{0.158228})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.084396)}{0.077719} + \underset{(0.050910)}{0.025932}\,n_{t-1} + \frac{\underset{(0.945793)}{1.058300}}{2(\underset{(0.535814)}{0.845516})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132429 | 0.096217 |
| rho_1 | 0.292919 | 0.093051 |
| phi_1 | 0.624886 | 0.160060 |
| p0 | 1.966298 | 1.369127 |
| n0 | 0.077719 | 0.084396 |
| rho_p | 0.560834 | 0.193507 |
| rho_n | 0.025932 | 0.050910 |
| phi_p_plus | 0.633880 | 0.305597 |
| phi_n_minus | 1.058300 | 0.945793 |
| sigma_p | 0.158228 | 0.066543 |
| sigma_n | 0.845516 | 0.535814 |

### Rank 13: Seed 8, Draw 37

- LogLik: `-170.519043`; AIC: `363.038087`; BIC: `400.115105`
- Max shape path: `60.069354`; max implied variance: `9.271536`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.077247)}{0.132334} + \underset{(0.086775)}{0.293053}\,\pi_t + \underset{(0.136143)}{0.624805}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001813)}{0.158244}\,\omega_{p,t} - \underset{(0.385496)}{0.845650}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.116219)}{1.962303} + \underset{(0.067993)}{0.561430}\,p_{t-1} + \frac{\underset{(0.152472)}{0.632845}}{2(\underset{(0.001813)}{0.158244})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.064396)}{0.077683} + \underset{(0.051255)}{0.025818}\,n_{t-1} + \frac{\underset{(0.521322)}{1.059398}}{2(\underset{(0.385496)}{0.845650})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132334 | 0.077247 |
| rho_1 | 0.293053 | 0.086775 |
| phi_1 | 0.624805 | 0.136143 |
| p0 | 1.962303 | 0.116219 |
| n0 | 0.077683 | 0.064396 |
| rho_p | 0.561430 | 0.067993 |
| rho_n | 0.025818 | 0.051255 |
| phi_p_plus | 0.632845 | 0.152472 |
| phi_n_minus | 1.059398 | 0.521322 |
| sigma_p | 0.158244 | 0.001813 |
| sigma_n | 0.845650 | 0.385496 |

### Rank 14: Seed 18, Draw 23

- LogLik: `-170.519044`; AIC: `363.038087`; BIC: `400.115105`
- Max shape path: `60.166748`; max implied variance: `9.273158`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.096592)}{0.132450} + \underset{(0.097045)}{0.293181}\,\pi_t + \underset{(0.168164)}{0.624529}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.076475)}{0.158253}\,\omega_{p,t} - \underset{(0.548625)}{0.846016}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.858793)}{1.966760} + \underset{(0.209156)}{0.560610}\,p_{t-1} + \frac{\underset{(0.329897)}{0.634063}}{2(\underset{(0.076475)}{0.158253})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.085637)}{0.077608} + \underset{(0.051054)}{0.025787}\,n_{t-1} + \frac{\underset{(0.966216)}{1.059595}}{2(\underset{(0.548625)}{0.846016})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132450 | 0.096592 |
| rho_1 | 0.293181 | 0.097045 |
| phi_1 | 0.624529 | 0.168164 |
| p0 | 1.966760 | 1.858793 |
| n0 | 0.077608 | 0.085637 |
| rho_p | 0.560610 | 0.209156 |
| rho_n | 0.025787 | 0.051054 |
| phi_p_plus | 0.634063 | 0.329897 |
| phi_n_minus | 1.059595 | 0.966216 |
| sigma_p | 0.158253 | 0.076475 |
| sigma_n | 0.846016 | 0.548625 |

### Rank 15: Seed 15, Draw 12

- LogLik: `-170.519044`; AIC: `363.038087`; BIC: `400.115106`
- Max shape path: `60.233294`; max implied variance: `9.278171`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.097785)}{0.132293} + \underset{(0.091789)}{0.293016}\,\pi_t + \underset{(0.138022)}{0.624870}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.066063)}{0.158154}\,\omega_{p,t} - \underset{(0.463420)}{0.845456}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.968378)}{1.966488} + \underset{(0.247461)}{0.560871}\,p_{t-1} + \frac{\underset{(0.380202)}{0.633970}}{2(\underset{(0.066063)}{0.158154})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.072206)}{0.077633} + \underset{(0.054274)}{0.025879}\,n_{t-1} + \frac{\underset{(0.748856)}{1.060215}}{2(\underset{(0.463420)}{0.845456})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132293 | 0.097785 |
| rho_1 | 0.293016 | 0.091789 |
| phi_1 | 0.624870 | 0.138022 |
| p0 | 1.966488 | 0.968378 |
| n0 | 0.077633 | 0.072206 |
| rho_p | 0.560871 | 0.247461 |
| rho_n | 0.025879 | 0.054274 |
| phi_p_plus | 0.633970 | 0.380202 |
| phi_n_minus | 1.060215 | 0.748856 |
| sigma_p | 0.158154 | 0.066063 |
| sigma_n | 0.845456 | 0.463420 |

### Rank 16: Seed 13, Draw 13

- LogLik: `-170.519044`; AIC: `363.038089`; BIC: `400.115107`
- Max shape path: `60.084779`; max implied variance: `9.273729`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.097230)}{0.132221} + \underset{(0.091826)}{0.293084}\,\pi_t + \underset{(0.159929)}{0.624879}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.064147)}{0.158246}\,\omega_{p,t} - \underset{(0.568784)}{0.845473}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.135674)}{1.962653} + \underset{(0.173301)}{0.561348}\,p_{t-1} + \frac{\underset{(0.276273)}{0.633001}}{2(\underset{(0.064147)}{0.158246})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.089104)}{0.077739} + \underset{(0.052035)}{0.025555}\,n_{t-1} + \frac{\underset{(1.045320)}{1.059685}}{2(\underset{(0.568784)}{0.845473})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132221 | 0.097230 |
| rho_1 | 0.293084 | 0.091826 |
| phi_1 | 0.624879 | 0.159929 |
| p0 | 1.962653 | 1.135674 |
| n0 | 0.077739 | 0.089104 |
| rho_p | 0.561348 | 0.173301 |
| rho_n | 0.025555 | 0.052035 |
| phi_p_plus | 0.633001 | 0.276273 |
| phi_n_minus | 1.059685 | 1.045320 |
| sigma_p | 0.158246 | 0.064147 |
| sigma_n | 0.845473 | 0.568784 |

### Rank 17: Seed 4, Draw 32

- LogLik: `-170.519045`; AIC: `363.038089`; BIC: `400.115107`
- Max shape path: `60.262414`; max implied variance: `9.282285`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.092457)}{0.132306} + \underset{(0.090895)}{0.293152}\,\pi_t + \underset{(0.155069)}{0.624727}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.057282)}{0.158128}\,\omega_{p,t} - \underset{(0.489374)}{0.845977}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.031559)}{1.968522} + \underset{(0.183711)}{0.560705}\,p_{t-1} + \frac{\underset{(0.288262)}{0.634054}}{2(\underset{(0.057282)}{0.158128})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.077250)}{0.077596} + \underset{(0.049898)}{0.025702}\,n_{t-1} + \frac{\underset{(0.832725)}{1.060699}}{2(\underset{(0.489374)}{0.845977})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132306 | 0.092457 |
| rho_1 | 0.293152 | 0.090895 |
| phi_1 | 0.624727 | 0.155069 |
| p0 | 1.968522 | 1.031559 |
| n0 | 0.077596 | 0.077250 |
| rho_p | 0.560705 | 0.183711 |
| rho_n | 0.025702 | 0.049898 |
| phi_p_plus | 0.634054 | 0.288262 |
| phi_n_minus | 1.060699 | 0.832725 |
| sigma_p | 0.158128 | 0.057282 |
| sigma_n | 0.845977 | 0.489374 |

### Rank 18: Seed 1, Draw 24

- LogLik: `-170.519045`; AIC: `363.038090`; BIC: `400.115108`
- Max shape path: `60.249027`; max implied variance: `9.260523`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.130152)}{0.132377} + \underset{(0.090138)}{0.292852}\,\pi_t + \underset{(0.156731)}{0.624997}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.069776)}{0.158171}\,\omega_{p,t} - \underset{(0.952440)}{0.845421}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(4.676365)}{1.968678} + \underset{(0.672114)}{0.560577}\,p_{t-1} + \frac{\underset{(1.086298)}{0.634334}}{2(\underset{(0.069776)}{0.158171})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.107858)}{0.077650} + \underset{(0.054072)}{0.025795}\,n_{t-1} + \frac{\underset{(1.539647)}{1.058104}}{2(\underset{(0.952440)}{0.845421})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132377 | 0.130152 |
| rho_1 | 0.292852 | 0.090138 |
| phi_1 | 0.624997 | 0.156731 |
| p0 | 1.968678 | 4.676365 |
| n0 | 0.077650 | 0.107858 |
| rho_p | 0.560577 | 0.672114 |
| rho_n | 0.025795 | 0.054072 |
| phi_p_plus | 0.634334 | 1.086298 |
| phi_n_minus | 1.058104 | 1.539647 |
| sigma_p | 0.158171 | 0.069776 |
| sigma_n | 0.845421 | 0.952440 |

### Rank 19: Seed 30, Draw 7

- LogLik: `-170.519045`; AIC: `363.038090`; BIC: `400.115108`
- Max shape path: `60.171451`; max implied variance: `9.292653`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.095403)}{0.132300} + \underset{(0.092882)}{0.293106}\,\pi_t + \underset{(0.160690)}{0.624741}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.063274)}{0.158203}\,\omega_{p,t} - \underset{(0.486560)}{0.846681}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.215675)}{1.965500} + \underset{(0.186466)}{0.560964}\,p_{t-1} + \frac{\underset{(0.292050)}{0.633642}}{2(\underset{(0.063274)}{0.158203})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.078254)}{0.077537} + \underset{(0.050331)}{0.025682}\,n_{t-1} + \frac{\underset{(0.821878)}{1.061925}}{2(\underset{(0.486560)}{0.846681})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132300 | 0.095403 |
| rho_1 | 0.293106 | 0.092882 |
| phi_1 | 0.624741 | 0.160690 |
| p0 | 1.965500 | 1.215675 |
| n0 | 0.077537 | 0.078254 |
| rho_p | 0.560964 | 0.186466 |
| rho_n | 0.025682 | 0.050331 |
| phi_p_plus | 0.633642 | 0.292050 |
| phi_n_minus | 1.061925 | 0.821878 |
| sigma_p | 0.158203 | 0.063274 |
| sigma_n | 0.846681 | 0.486560 |

### Rank 20: Seed 41, Draw 30

- LogLik: `-170.519045`; AIC: `363.038090`; BIC: `400.115108`
- Max shape path: `60.281074`; max implied variance: `9.266821`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072119)}{0.132273} + \underset{(0.086278)}{0.292999}\,\pi_t + \underset{(0.130436)}{0.624917}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.017470)}{0.158113}\,\omega_{p,t} - \underset{(0.380236)}{0.845976}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.387067)}{1.968463} + \underset{(0.092484)}{0.560717}\,p_{t-1} + \frac{\underset{(0.168406)}{0.634151}}{2(\underset{(0.017470)}{0.158113})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.060560)}{0.077612} + \underset{(0.056878)}{0.025947}\,n_{t-1} + \frac{\underset{(0.431998)}{1.058871}}{2(\underset{(0.380236)}{0.845976})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.132273 | 0.072119 |
| rho_1 | 0.292999 | 0.086278 |
| phi_1 | 0.624917 | 0.130436 |
| p0 | 1.968463 | 0.387067 |
| n0 | 0.077612 | 0.060560 |
| rho_p | 0.560717 | 0.092484 |
| rho_n | 0.025947 | 0.056878 |
| phi_p_plus | 0.634151 | 0.168406 |
| phi_n_minus | 1.058871 | 0.431998 |
| sigma_p | 0.158113 | 0.017470 |
| sigma_n | 0.845976 | 0.380236 |

## ARX(2,1)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 47 | 31 | -140.463596 | 304.927192 | 345.374849 | 409.057396 | 2.576010 | yes | `computed` |
| 2 | 46 | 7 | -153.423467 | 330.846934 | 371.294590 | 3060.322794 | 1.080083 | no | `computed` |
| 3 | 5 | 16 | -165.929106 | 355.858212 | 396.305868 | 2486.656875 | 17.282857 | no | `computed` |
| 4 | 34 | 29 | -168.076017 | 360.152033 | 400.599690 | 49.082928 | 12.034470 | no | `computed` |
| 5 | 11 | 34 | -168.076017 | 360.152034 | 400.599690 | 49.081808 | 12.036596 | no | `computed` |
| 6 | 24 | 20 | -168.076019 | 360.152039 | 400.599695 | 49.057950 | 12.025682 | no | `computed` |
| 7 | 24 | 19 | -168.076020 | 360.152039 | 400.599696 | 49.059462 | 12.021813 | no | `computed` |
| 8 | 14 | 31 | -168.076020 | 360.152040 | 400.599697 | 49.122247 | 12.035804 | no | `computed` |
| 9 | 10 | 8 | -168.076020 | 360.152040 | 400.599697 | 49.100695 | 12.025774 | no | `computed` |
| 10 | 10 | 31 | -168.076020 | 360.152041 | 400.599697 | 49.094042 | 12.046562 | no | `computed` |
| 11 | 45 | 19 | -168.076021 | 360.152042 | 400.599698 | 49.058240 | 12.020127 | no | `computed` |
| 12 | 12 | 15 | -168.076021 | 360.152042 | 400.599698 | 49.087567 | 12.043173 | no | `computed` |
| 13 | 21 | 22 | -168.076021 | 360.152043 | 400.599699 | 49.059068 | 12.045171 | no | `computed` |
| 14 | 40 | 14 | -168.076022 | 360.152044 | 400.599701 | 49.016566 | 12.028748 | no | `computed` |
| 15 | 3 | 27 | -168.076023 | 360.152046 | 400.599703 | 48.980287 | 12.026439 | no | `computed` |
| 16 | 19 | 19 | -168.076023 | 360.152047 | 400.599703 | 49.105067 | 12.037275 | no | `computed` |
| 17 | 12 | 27 | -168.076023 | 360.152047 | 400.599703 | 49.067856 | 12.047911 | no | `computed` |
| 18 | 36 | 40 | -168.076024 | 360.152049 | 400.599705 | 49.053205 | 12.038503 | no | `computed` |
| 19 | 1 | 28 | -168.076025 | 360.152050 | 400.599706 | 49.149770 | 12.034727 | no | `computed` |
| 20 | 25 | 38 | -168.076025 | 360.152050 | 400.599706 | 49.016143 | 12.025992 | no | `computed` |

### Rank 1: Seed 47, Draw 31

- LogLik: `-140.463596`; AIC: `304.927192`; BIC: `345.374849`
- Max shape path: `409.057396`; max implied variance: `2.576010`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.028273} + \underset{(0.000002)}{0.345298}\,\pi_t + \underset{(0.000002)}{0.053284}\,\pi_{t-1} + \underset{(0.000002)}{0.670520}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.049190}\,\omega_{p,t} - \underset{(0.000007)}{0.190090}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000000)}{4.831677} + \underset{(0.000007)}{0.785878}\,p_{t-1} + \frac{\underset{(0.000002)}{0.392201}}{2(\underset{(0.000002)}{0.049190})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.000001)}{2.032389} + \underset{(0.000002)}{0.216245}\,n_{t-1} + \frac{\underset{(0.000002)}{0.251030}}{2(\underset{(0.000007)}{0.190090})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.028273 | 0.000002 |
| rho_1 | 0.345298 | 0.000002 |
| rho_2 | 0.053284 | 0.000002 |
| phi_1 | 0.670520 | 0.000002 |
| p0 | 4.831677 | 0.000000 |
| n0 | 2.032389 | 0.000001 |
| rho_p | 0.785878 | 0.000007 |
| rho_n | 0.216245 | 0.000002 |
| phi_p_plus | 0.392201 | 0.000002 |
| phi_n_minus | 0.251030 | 0.000002 |
| sigma_p | 0.049190 | 0.000002 |
| sigma_n | 0.190090 | 0.000007 |

### Rank 2: Seed 46, Draw 7

- LogLik: `-153.423467`; AIC: `330.846934`; BIC: `371.294590`
- Max shape path: `3060.322794`; max implied variance: `1.080083`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.077050)}{0.073322} + \underset{(0.017815)}{0.270793}\,\pi_t + \underset{(0.256291)}{0.072230}\,\pi_{t-1} + \underset{(0.045672)}{0.647075}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.001185)}{0.017058}\,\omega_{p,t} - \underset{(0.076576)}{0.080176}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.359270)}{9.799563} + \underset{(0.005755)}{0.795104}\,p_{t-1} + \frac{\underset{(0.041219)}{0.363053}}{2(\underset{(0.001185)}{0.017058})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(2.085485)}{3.077954} + \underset{(0.138436)}{0.895644}\,n_{t-1} + \frac{\underset{(0.108521)}{0.000000}}{2(\underset{(0.076576)}{0.080176})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.073322 | 0.077050 |
| rho_1 | 0.270793 | 0.017815 |
| rho_2 | 0.072230 | 0.256291 |
| phi_1 | 0.647075 | 0.045672 |
| p0 | 9.799563 | 0.359270 |
| n0 | 3.077954 | 2.085485 |
| rho_p | 0.795104 | 0.005755 |
| rho_n | 0.895644 | 0.138436 |
| phi_p_plus | 0.363053 | 0.041219 |
| phi_n_minus | 0.000000 | 0.108521 |
| sigma_p | 0.017058 | 0.001185 |
| sigma_n | 0.080176 | 0.076576 |

### Rank 3: Seed 5, Draw 16

- LogLik: `-165.929106`; AIC: `355.858212`; BIC: `396.305868`
- Max shape path: `2486.656875`; max implied variance: `17.282857`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{-0.011164} + \underset{(0.000002)}{0.527992}\,\pi_t + \underset{(0.000002)}{0.139352}\,\pi_{t-1} + \underset{(0.000002)}{0.344646}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.025389}\,\omega_{p,t} - \underset{(0.000002)}{0.160542}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000002)}{0.079741} + \underset{(0.000002)}{0.483727}\,p_{t-1} + \frac{\underset{(0.000002)}{0.657988}}{2(\underset{(0.000002)}{0.025389})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.000000)}{6.516500} + \underset{(0.000002)}{0.021806}\,n_{t-1} + \frac{\underset{(0.000001)}{1.913988}}{2(\underset{(0.000002)}{0.160542})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.011164 | 0.000002 |
| rho_1 | 0.527992 | 0.000002 |
| rho_2 | 0.139352 | 0.000002 |
| phi_1 | 0.344646 | 0.000002 |
| p0 | 0.079741 | 0.000002 |
| n0 | 6.516500 | 0.000000 |
| rho_p | 0.483727 | 0.000002 |
| rho_n | 0.021806 | 0.000002 |
| phi_p_plus | 0.657988 | 0.000002 |
| phi_n_minus | 1.913988 | 0.000001 |
| sigma_p | 0.025389 | 0.000002 |
| sigma_n | 0.160542 | 0.000002 |

### Rank 4: Seed 34, Draw 29

- LogLik: `-168.076017`; AIC: `360.152033`; BIC: `400.599690`
- Max shape path: `49.082928`; max implied variance: `12.034470`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071761)}{0.126408} + \underset{(0.075857)}{0.256718}\,\pi_t + \underset{(0.074172)}{0.167312}\,\pi_{t-1} + \underset{(0.119497)}{0.480898}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043961)}{0.149209}\,\omega_{p,t} - \underset{(0.562284)}{0.986608}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.820889)}{1.361429} + \underset{(0.121199)}{0.691663}\,p_{t-1} + \frac{\underset{(0.175479)}{0.446940}}{2(\underset{(0.043961)}{0.149209})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046216)}{0.048911} + \underset{(0.057790)}{0.136762}\,n_{t-1} + \frac{\underset{(0.833787)}{1.260904}}{2(\underset{(0.562284)}{0.986608})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126408 | 0.071761 |
| rho_1 | 0.256718 | 0.075857 |
| rho_2 | 0.167312 | 0.074172 |
| phi_1 | 0.480898 | 0.119497 |
| p0 | 1.361429 | 0.820889 |
| n0 | 0.048911 | 0.046216 |
| rho_p | 0.691663 | 0.121199 |
| rho_n | 0.136762 | 0.057790 |
| phi_p_plus | 0.446940 | 0.175479 |
| phi_n_minus | 1.260904 | 0.833787 |
| sigma_p | 0.149209 | 0.043961 |
| sigma_n | 0.986608 | 0.562284 |

### Rank 5: Seed 11, Draw 34

- LogLik: `-168.076017`; AIC: `360.152034`; BIC: `400.599690`
- Max shape path: `49.081808`; max implied variance: `12.036596`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071994)}{0.126442} + \underset{(0.076008)}{0.256680}\,\pi_t + \underset{(0.075335)}{0.167318}\,\pi_{t-1} + \underset{(0.120247)}{0.480903}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.045092)}{0.149219}\,\omega_{p,t} - \underset{(0.552721)}{0.985755}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.893555)}{1.361226} + \underset{(0.126159)}{0.691659}\,p_{t-1} + \frac{\underset{(0.181504)}{0.446957}}{2(\underset{(0.045092)}{0.149219})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.045920)}{0.048951} + \underset{(0.058467)}{0.136650}\,n_{t-1} + \frac{\underset{(0.829451)}{1.261127}}{2(\underset{(0.552721)}{0.985755})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126442 | 0.071994 |
| rho_1 | 0.256680 | 0.076008 |
| rho_2 | 0.167318 | 0.075335 |
| phi_1 | 0.480903 | 0.120247 |
| p0 | 1.361226 | 0.893555 |
| n0 | 0.048951 | 0.045920 |
| rho_p | 0.691659 | 0.126159 |
| rho_n | 0.136650 | 0.058467 |
| phi_p_plus | 0.446957 | 0.181504 |
| phi_n_minus | 1.261127 | 0.829451 |
| sigma_p | 0.149219 | 0.045092 |
| sigma_n | 0.985755 | 0.552721 |

### Rank 6: Seed 24, Draw 20

- LogLik: `-168.076019`; AIC: `360.152039`; BIC: `400.599695`
- Max shape path: `49.057950`; max implied variance: `12.025682`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072111)}{0.126465} + \underset{(0.075572)}{0.256701}\,\pi_t + \underset{(0.074378)}{0.167225}\,\pi_{t-1} + \underset{(0.120036)}{0.480999}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042540)}{0.149280}\,\omega_{p,t} - \underset{(0.512118)}{0.985430}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.785137)}{1.360645} + \underset{(0.121513)}{0.691602}\,p_{t-1} + \frac{\underset{(0.176704)}{0.447152}}{2(\underset{(0.042540)}{0.149280})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.043940)}{0.048972} + \underset{(0.057209)}{0.136672}\,n_{t-1} + \frac{\underset{(0.756883)}{1.259997}}{2(\underset{(0.512118)}{0.985430})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126465 | 0.072111 |
| rho_1 | 0.256701 | 0.075572 |
| rho_2 | 0.167225 | 0.074378 |
| phi_1 | 0.480999 | 0.120036 |
| p0 | 1.360645 | 0.785137 |
| n0 | 0.048972 | 0.043940 |
| rho_p | 0.691602 | 0.121513 |
| rho_n | 0.136672 | 0.057209 |
| phi_p_plus | 0.447152 | 0.176704 |
| phi_n_minus | 1.259997 | 0.756883 |
| sigma_p | 0.149280 | 0.042540 |
| sigma_n | 0.985430 | 0.512118 |

### Rank 7: Seed 24, Draw 19

- LogLik: `-168.076020`; AIC: `360.152039`; BIC: `400.599696`
- Max shape path: `49.059462`; max implied variance: `12.021813`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.073033)}{0.126400} + \underset{(0.075752)}{0.256679}\,\pi_t + \underset{(0.074928)}{0.167313}\,\pi_{t-1} + \underset{(0.119695)}{0.480972}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.044086)}{0.149233}\,\omega_{p,t} - \underset{(0.624686)}{0.985837}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.855488)}{1.360267} + \underset{(0.124742)}{0.691793}\,p_{t-1} + \frac{\underset{(0.180980)}{0.446805}}{2(\underset{(0.044086)}{0.149233})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.049388)}{0.048960} + \underset{(0.057276)}{0.136854}\,n_{t-1} + \frac{\underset{(0.930253)}{1.259534}}{2(\underset{(0.624686)}{0.985837})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126400 | 0.073033 |
| rho_1 | 0.256679 | 0.075752 |
| rho_2 | 0.167313 | 0.074928 |
| phi_1 | 0.480972 | 0.119695 |
| p0 | 1.360267 | 0.855488 |
| n0 | 0.048960 | 0.049388 |
| rho_p | 0.691793 | 0.124742 |
| rho_n | 0.136854 | 0.057276 |
| phi_p_plus | 0.446805 | 0.180980 |
| phi_n_minus | 1.259534 | 0.930253 |
| sigma_p | 0.149233 | 0.044086 |
| sigma_n | 0.985837 | 0.624686 |

### Rank 8: Seed 14, Draw 31

- LogLik: `-168.076020`; AIC: `360.152040`; BIC: `400.599697`
- Max shape path: `49.122247`; max implied variance: `12.035804`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072223)}{0.126448} + \underset{(0.075623)}{0.256729}\,\pi_t + \underset{(0.074868)}{0.167290}\,\pi_{t-1} + \underset{(0.119855)}{0.480836}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043415)}{0.149198}\,\omega_{p,t} - \underset{(0.574710)}{0.985927}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.800486)}{1.362762} + \underset{(0.120104)}{0.691407}\,p_{t-1} + \frac{\underset{(0.174986)}{0.447338}}{2(\underset{(0.043415)}{0.149198})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046929)}{0.048933} + \underset{(0.057795)}{0.136804}\,n_{t-1} + \frac{\underset{(0.833497)}{1.261074}}{2(\underset{(0.574710)}{0.985927})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126448 | 0.072223 |
| rho_1 | 0.256729 | 0.075623 |
| rho_2 | 0.167290 | 0.074868 |
| phi_1 | 0.480836 | 0.119855 |
| p0 | 1.362762 | 0.800486 |
| n0 | 0.048933 | 0.046929 |
| rho_p | 0.691407 | 0.120104 |
| rho_n | 0.136804 | 0.057795 |
| phi_p_plus | 0.447338 | 0.174986 |
| phi_n_minus | 1.261074 | 0.833497 |
| sigma_p | 0.149198 | 0.043415 |
| sigma_n | 0.985927 | 0.574710 |

### Rank 9: Seed 10, Draw 8

- LogLik: `-168.076020`; AIC: `360.152040`; BIC: `400.599697`
- Max shape path: `49.100695`; max implied variance: `12.025774`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072873)}{0.126545} + \underset{(0.076034)}{0.256773}\,\pi_t + \underset{(0.075053)}{0.167354}\,\pi_{t-1} + \underset{(0.120131)}{0.480667}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043543)}{0.149209}\,\omega_{p,t} - \underset{(0.589708)}{0.986005}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.832154)}{1.362276} + \underset{(0.123293)}{0.691523}\,p_{t-1} + \frac{\underset{(0.179441)}{0.447170}}{2(\underset{(0.043543)}{0.149209})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.047680)}{0.048930} + \underset{(0.057196)}{0.136748}\,n_{t-1} + \frac{\underset{(0.886614)}{1.259911}}{2(\underset{(0.589708)}{0.986005})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126545 | 0.072873 |
| rho_1 | 0.256773 | 0.076034 |
| rho_2 | 0.167354 | 0.075053 |
| phi_1 | 0.480667 | 0.120131 |
| p0 | 1.362276 | 0.832154 |
| n0 | 0.048930 | 0.047680 |
| rho_p | 0.691523 | 0.123293 |
| rho_n | 0.136748 | 0.057196 |
| phi_p_plus | 0.447170 | 0.179441 |
| phi_n_minus | 1.259911 | 0.886614 |
| sigma_p | 0.149209 | 0.043543 |
| sigma_n | 0.986005 | 0.589708 |

### Rank 10: Seed 10, Draw 31

- LogLik: `-168.076020`; AIC: `360.152041`; BIC: `400.599697`
- Max shape path: `49.094042`; max implied variance: `12.046562`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.192406)}{0.126337} + \underset{(0.939631)}{0.256690}\,\pi_t + \underset{(0.068120)}{0.167203}\,\pi_{t-1} + \underset{(0.809775)}{0.481131}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.045745)}{0.149199}\,\omega_{p,t} - \underset{(0.832187)}{0.986372}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(8.332116)}{1.360951} + \underset{(1.460044)}{0.691727}\,p_{t-1} + \frac{\underset{(1.909430)}{0.446950}}{2(\underset{(0.045745)}{0.149199})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.060002)}{0.048927} + \underset{(0.123172)}{0.136483}\,n_{t-1} + \frac{\underset{(1.026920)}{1.262296}}{2(\underset{(0.832187)}{0.986372})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126337 | 0.192406 |
| rho_1 | 0.256690 | 0.939631 |
| rho_2 | 0.167203 | 0.068120 |
| phi_1 | 0.481131 | 0.809775 |
| p0 | 1.360951 | 8.332116 |
| n0 | 0.048927 | 0.060002 |
| rho_p | 0.691727 | 1.460044 |
| rho_n | 0.136483 | 0.123172 |
| phi_p_plus | 0.446950 | 1.909430 |
| phi_n_minus | 1.262296 | 1.026920 |
| sigma_p | 0.149199 | 0.045745 |
| sigma_n | 0.986372 | 0.832187 |

### Rank 11: Seed 45, Draw 19

- LogLik: `-168.076021`; AIC: `360.152042`; BIC: `400.599698`
- Max shape path: `49.058240`; max implied variance: `12.020127`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071941)}{0.126514} + \underset{(0.075701)}{0.256724}\,\pi_t + \underset{(0.074468)}{0.167329}\,\pi_{t-1} + \underset{(0.119614)}{0.480795}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043190)}{0.149237}\,\omega_{p,t} - \underset{(0.542511)}{0.985281}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.798531)}{1.360661} + \underset{(0.120474)}{0.691762}\,p_{t-1} + \frac{\underset{(0.175251)}{0.446795}}{2(\underset{(0.043190)}{0.149237})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.045514)}{0.049004} + \underset{(0.057199)}{0.136635}\,n_{t-1} + \frac{\underset{(0.795741)}{1.259321}}{2(\underset{(0.542511)}{0.985281})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126514 | 0.071941 |
| rho_1 | 0.256724 | 0.075701 |
| rho_2 | 0.167329 | 0.074468 |
| phi_1 | 0.480795 | 0.119614 |
| p0 | 1.360661 | 0.798531 |
| n0 | 0.049004 | 0.045514 |
| rho_p | 0.691762 | 0.120474 |
| rho_n | 0.136635 | 0.057199 |
| phi_p_plus | 0.446795 | 0.175251 |
| phi_n_minus | 1.259321 | 0.795741 |
| sigma_p | 0.149237 | 0.043190 |
| sigma_n | 0.985281 | 0.542511 |

### Rank 12: Seed 12, Draw 15

- LogLik: `-168.076021`; AIC: `360.152042`; BIC: `400.599698`
- Max shape path: `49.087567`; max implied variance: `12.043173`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071427)}{0.126366} + \underset{(0.075770)}{0.256597}\,\pi_t + \underset{(0.074149)}{0.167289}\,\pi_{t-1} + \underset{(0.119840)}{0.481091}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042606)}{0.149187}\,\omega_{p,t} - \underset{(0.522427)}{0.986794}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.740462)}{1.361564} + \underset{(0.114469)}{0.691725}\,p_{t-1} + \frac{\underset{(0.167248)}{0.446733}}{2(\underset{(0.042606)}{0.149187})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.044338)}{0.048924} + \underset{(0.057207)}{0.136569}\,n_{t-1} + \frac{\underset{(0.762257)}{1.261870}}{2(\underset{(0.522427)}{0.986794})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126366 | 0.071427 |
| rho_1 | 0.256597 | 0.075770 |
| rho_2 | 0.167289 | 0.074149 |
| phi_1 | 0.481091 | 0.119840 |
| p0 | 1.361564 | 0.740462 |
| n0 | 0.048924 | 0.044338 |
| rho_p | 0.691725 | 0.114469 |
| rho_n | 0.136569 | 0.057207 |
| phi_p_plus | 0.446733 | 0.167248 |
| phi_n_minus | 1.261870 | 0.762257 |
| sigma_p | 0.149187 | 0.042606 |
| sigma_n | 0.986794 | 0.522427 |

### Rank 13: Seed 21, Draw 22

- LogLik: `-168.076021`; AIC: `360.152043`; BIC: `400.599699`
- Max shape path: `49.059068`; max implied variance: `12.045171`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072835)}{0.126357} + \underset{(0.075658)}{0.256719}\,\pi_t + \underset{(0.074666)}{0.167216}\,\pi_{t-1} + \underset{(0.119280)}{0.481066}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042620)}{0.149250}\,\omega_{p,t} - \underset{(0.555004)}{0.986872}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.789183)}{1.360041} + \underset{(0.120073)}{0.691764}\,p_{t-1} + \frac{\underset{(0.175645)}{0.446921}}{2(\underset{(0.042620)}{0.149250})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046068)}{0.048916} + \underset{(0.057427)}{0.136404}\,n_{t-1} + \frac{\underset{(0.821074)}{1.262126}}{2(\underset{(0.555004)}{0.986872})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126357 | 0.072835 |
| rho_1 | 0.256719 | 0.075658 |
| rho_2 | 0.167216 | 0.074666 |
| phi_1 | 0.481066 | 0.119280 |
| p0 | 1.360041 | 0.789183 |
| n0 | 0.048916 | 0.046068 |
| rho_p | 0.691764 | 0.120073 |
| rho_n | 0.136404 | 0.057427 |
| phi_p_plus | 0.446921 | 0.175645 |
| phi_n_minus | 1.262126 | 0.821074 |
| sigma_p | 0.149250 | 0.042620 |
| sigma_n | 0.986872 | 0.555004 |

### Rank 14: Seed 40, Draw 14

- LogLik: `-168.076022`; AIC: `360.152044`; BIC: `400.599701`
- Max shape path: `49.016566`; max implied variance: `12.028748`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.070268)}{0.126421} + \underset{(0.075737)}{0.256733}\,\pi_t + \underset{(0.073970)}{0.167276}\,\pi_{t-1} + \underset{(0.119221)}{0.480909}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.041689)}{0.149308}\,\omega_{p,t} - \underset{(0.585538)}{0.986130}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.580442)}{1.359036} + \underset{(0.100626)}{0.691781}\,p_{t-1} + \frac{\underset{(0.149611)}{0.446845}}{2(\underset{(0.041689)}{0.149308})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.047595)}{0.048953} + \underset{(0.054216)}{0.136570}\,n_{t-1} + \frac{\underset{(0.824997)}{1.260310}}{2(\underset{(0.585538)}{0.986130})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126421 | 0.070268 |
| rho_1 | 0.256733 | 0.075737 |
| rho_2 | 0.167276 | 0.073970 |
| phi_1 | 0.480909 | 0.119221 |
| p0 | 1.359036 | 0.580442 |
| n0 | 0.048953 | 0.047595 |
| rho_p | 0.691781 | 0.100626 |
| rho_n | 0.136570 | 0.054216 |
| phi_p_plus | 0.446845 | 0.149611 |
| phi_n_minus | 1.260310 | 0.824997 |
| sigma_p | 0.149308 | 0.041689 |
| sigma_n | 0.986130 | 0.585538 |

### Rank 15: Seed 3, Draw 27

- LogLik: `-168.076023`; AIC: `360.152046`; BIC: `400.599703`
- Max shape path: `48.980287`; max implied variance: `12.026439`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.196783)}{0.126429} + \underset{(0.084145)}{0.256757}\,\pi_t + \underset{(0.163873)}{0.167319}\,\pi_{t-1} + \underset{(0.154865)}{0.480880}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039723)}{0.149336}\,\omega_{p,t} - \underset{(2.288454)}{0.985887}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(4.321034)}{1.358990} + \underset{(0.618135)}{0.691788}\,p_{t-1} + \frac{\underset{(0.852788)}{0.446745}}{2(\underset{(0.039723)}{0.149336})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.142883)}{0.048939} + \underset{(0.118455)}{0.136750}\,n_{t-1} + \frac{\underset{(3.462482)}{1.260014}}{2(\underset{(2.288454)}{0.985887})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126429 | 0.196783 |
| rho_1 | 0.256757 | 0.084145 |
| rho_2 | 0.167319 | 0.163873 |
| phi_1 | 0.480880 | 0.154865 |
| p0 | 1.358990 | 4.321034 |
| n0 | 0.048939 | 0.142883 |
| rho_p | 0.691788 | 0.618135 |
| rho_n | 0.136750 | 0.118455 |
| phi_p_plus | 0.446745 | 0.852788 |
| phi_n_minus | 1.260014 | 3.462482 |
| sigma_p | 0.149336 | 0.039723 |
| sigma_n | 0.985887 | 2.288454 |

### Rank 16: Seed 19, Draw 19

- LogLik: `-168.076023`; AIC: `360.152047`; BIC: `400.599703`
- Max shape path: `49.105067`; max implied variance: `12.037275`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072375)}{0.126556} + \underset{(0.075852)}{0.256819}\,\pi_t + \underset{(0.074735)}{0.167327}\,\pi_{t-1} + \underset{(0.119770)}{0.480605}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043414)}{0.149217}\,\omega_{p,t} - \underset{(0.610210)}{0.985342}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.797498)}{1.362355} + \underset{(0.119663)}{0.691442}\,p_{t-1} + \frac{\underset{(0.174505)}{0.447293}}{2(\underset{(0.043414)}{0.149217})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.048812)}{0.048957} + \underset{(0.056691)}{0.136634}\,n_{t-1} + \frac{\underset{(0.893467)}{1.261182}}{2(\underset{(0.610210)}{0.985342})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126556 | 0.072375 |
| rho_1 | 0.256819 | 0.075852 |
| rho_2 | 0.167327 | 0.074735 |
| phi_1 | 0.480605 | 0.119770 |
| p0 | 1.362355 | 0.797498 |
| n0 | 0.048957 | 0.048812 |
| rho_p | 0.691442 | 0.119663 |
| rho_n | 0.136634 | 0.056691 |
| phi_p_plus | 0.447293 | 0.174505 |
| phi_n_minus | 1.261182 | 0.893467 |
| sigma_p | 0.149217 | 0.043414 |
| sigma_n | 0.985342 | 0.610210 |

### Rank 17: Seed 12, Draw 27

- LogLik: `-168.076023`; AIC: `360.152047`; BIC: `400.599703`
- Max shape path: `49.067856`; max implied variance: `12.047911`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.073360)}{0.126416} + \underset{(0.075844)}{0.256691}\,\pi_t + \underset{(0.075097)}{0.167257}\,\pi_{t-1} + \underset{(0.120160)}{0.480968}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043282)}{0.149251}\,\omega_{p,t} - \underset{(0.615527)}{0.987877}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.817533)}{1.360602} + \underset{(0.122852)}{0.691690}\,p_{t-1} + \frac{\underset{(0.180288)}{0.446979}}{2(\underset{(0.043282)}{0.149251})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.048813)}{0.048830} + \underset{(0.058023)}{0.136660}\,n_{t-1} + \frac{\underset{(0.911079)}{1.262382}}{2(\underset{(0.615527)}{0.987877})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126416 | 0.073360 |
| rho_1 | 0.256691 | 0.075844 |
| rho_2 | 0.167257 | 0.075097 |
| phi_1 | 0.480968 | 0.120160 |
| p0 | 1.360602 | 0.817533 |
| n0 | 0.048830 | 0.048813 |
| rho_p | 0.691690 | 0.122852 |
| rho_n | 0.136660 | 0.058023 |
| phi_p_plus | 0.446979 | 0.180288 |
| phi_n_minus | 1.262382 | 0.911079 |
| sigma_p | 0.149251 | 0.043282 |
| sigma_n | 0.987877 | 0.615527 |

### Rank 18: Seed 36, Draw 40

- LogLik: `-168.076024`; AIC: `360.152049`; BIC: `400.599705`
- Max shape path: `49.053205`; max implied variance: `12.038503`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071980)}{0.126347} + \underset{(0.075791)}{0.256599}\,\pi_t + \underset{(0.074830)}{0.167344}\,\pi_{t-1} + \underset{(0.120619)}{0.481064}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043084)}{0.149236}\,\omega_{p,t} - \underset{(0.574509)}{0.987566}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.777574)}{1.360184} + \underset{(0.117468)}{0.691820}\,p_{t-1} + \frac{\underset{(0.171601)}{0.446722}}{2(\underset{(0.043084)}{0.149236})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046926)}{0.048853} + \underset{(0.056932)}{0.136860}\,n_{t-1} + \frac{\underset{(0.848191)}{1.261316}}{2(\underset{(0.574509)}{0.987566})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126347 | 0.071980 |
| rho_1 | 0.256599 | 0.075791 |
| rho_2 | 0.167344 | 0.074830 |
| phi_1 | 0.481064 | 0.120619 |
| p0 | 1.360184 | 0.777574 |
| n0 | 0.048853 | 0.046926 |
| rho_p | 0.691820 | 0.117468 |
| rho_n | 0.136860 | 0.056932 |
| phi_p_plus | 0.446722 | 0.171601 |
| phi_n_minus | 1.261316 | 0.848191 |
| sigma_p | 0.149236 | 0.043084 |
| sigma_n | 0.987566 | 0.574509 |

### Rank 19: Seed 1, Draw 28

- LogLik: `-168.076025`; AIC: `360.152050`; BIC: `400.599706`
- Max shape path: `49.149770`; max implied variance: `12.034727`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072821)}{0.126466} + \underset{(0.075798)}{0.256753}\,\pi_t + \underset{(0.075664)}{0.167239}\,\pi_{t-1} + \underset{(0.120509)}{0.480908}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.044090)}{0.149134}\,\omega_{p,t} - \underset{(0.614640)}{0.985763}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.845376)}{1.363417} + \underset{(0.123360)}{0.691532}\,p_{t-1} + \frac{\underset{(0.178896)}{0.447175}}{2(\underset{(0.044090)}{0.149134})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.048902)}{0.048972} + \underset{(0.058661)}{0.136303}\,n_{t-1} + \frac{\underset{(0.944961)}{1.260973}}{2(\underset{(0.614640)}{0.985763})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126466 | 0.072821 |
| rho_1 | 0.256753 | 0.075798 |
| rho_2 | 0.167239 | 0.075664 |
| phi_1 | 0.480908 | 0.120509 |
| p0 | 1.363417 | 0.845376 |
| n0 | 0.048972 | 0.048902 |
| rho_p | 0.691532 | 0.123360 |
| rho_n | 0.136303 | 0.058661 |
| phi_p_plus | 0.447175 | 0.178896 |
| phi_n_minus | 1.260973 | 0.944961 |
| sigma_p | 0.149134 | 0.044090 |
| sigma_n | 0.985763 | 0.614640 |

### Rank 20: Seed 25, Draw 38

- LogLik: `-168.076025`; AIC: `360.152050`; BIC: `400.599706`
- Max shape path: `49.016143`; max implied variance: `12.025992`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.069328)}{0.126373} + \underset{(0.076051)}{0.256796}\,\pi_t + \underset{(0.072105)}{0.167197}\,\pi_{t-1} + \underset{(0.120038)}{0.481030}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.046637)}{0.149301}\,\omega_{p,t} - \underset{(0.494129)}{0.986048}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.809129)}{1.359962} + \underset{(0.093703)}{0.691749}\,p_{t-1} + \frac{\underset{(0.138692)}{0.446896}}{2(\underset{(0.046637)}{0.149301})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.043637)}{0.048941} + \underset{(0.055526)}{0.136520}\,n_{t-1} + \frac{\underset{(0.684741)}{1.260061}}{2(\underset{(0.494129)}{0.986048})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.126373 | 0.069328 |
| rho_1 | 0.256796 | 0.076051 |
| rho_2 | 0.167197 | 0.072105 |
| phi_1 | 0.481030 | 0.120038 |
| p0 | 1.359962 | 0.809129 |
| n0 | 0.048941 | 0.043637 |
| rho_p | 0.691749 | 0.093703 |
| rho_n | 0.136520 | 0.055526 |
| phi_p_plus | 0.446896 | 0.138692 |
| phi_n_minus | 1.260061 | 0.684741 |
| sigma_p | 0.149301 | 0.046637 |
| sigma_n | 0.986048 | 0.494129 |

## ARX(2,2)

Top 20 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 4 | 20 | -37.452490 | 100.904979 | 144.723274 | 380.815763 | 97.559272 | yes | `computed` |
| 2 | 45 | 6 | -82.714014 | 191.428028 | 235.246323 | 515.371106 | 1.220388 | yes | `computed` |
| 3 | 13 | 37 | -159.918150 | 345.836301 | 389.654595 | 6782.964492 | 2.472044 | no | `computed` |
| 4 | 8 | 33 | -167.901751 | 361.803502 | 405.621796 | 47.754787 | 12.087754 | no | `computed` |
| 5 | 32 | 32 | -167.901752 | 361.803504 | 405.621798 | 47.736459 | 12.081901 | no | `computed` |
| 6 | 7 | 25 | -167.901755 | 361.803511 | 405.621805 | 47.759695 | 12.100717 | no | `computed` |
| 7 | 28 | 27 | -167.901756 | 361.803511 | 405.621806 | 47.763705 | 12.069869 | no | `computed` |
| 8 | 23 | 25 | -167.901758 | 361.803515 | 405.621810 | 47.813652 | 12.095096 | no | `computed` |
| 9 | 49 | 5 | -167.901758 | 361.803516 | 405.621810 | 47.724522 | 12.095358 | no | `computed` |
| 10 | 35 | 29 | -167.901758 | 361.803517 | 405.621811 | 47.695699 | 12.096702 | no | `computed` |
| 11 | 12 | 22 | -167.901759 | 361.803518 | 405.621812 | 47.686671 | 12.064548 | no | `computed` |
| 12 | 34 | 33 | -167.901759 | 361.803518 | 405.621812 | 47.653514 | 12.101978 | no | `computed` |
| 13 | 3 | 17 | -167.901759 | 361.803519 | 405.621813 | 47.693082 | 12.060110 | no | `computed` |
| 14 | 7 | 32 | -167.901760 | 361.803519 | 405.621813 | 47.798676 | 12.077718 | no | `computed` |
| 15 | 6 | 14 | -167.901760 | 361.803519 | 405.621814 | 47.733328 | 12.110893 | no | `computed` |
| 16 | 37 | 15 | -167.901760 | 361.803521 | 405.621815 | 47.704362 | 12.081168 | no | `computed` |
| 17 | 4 | 21 | -167.901761 | 361.803521 | 405.621816 | 47.761229 | 12.086573 | no | `computed` |
| 18 | 3 | 8 | -167.901761 | 361.803522 | 405.621816 | 47.735259 | 12.101691 | no | `computed` |
| 19 | 47 | 23 | -167.901762 | 361.803524 | 405.621818 | 47.803776 | 12.101864 | no | `computed` |
| 20 | 41 | 26 | -167.901762 | 361.803524 | 405.621819 | 47.771920 | 12.089646 | no | `computed` |

### Rank 1: Seed 4, Draw 20

- LogLik: `-37.452490`; AIC: `100.904979`; BIC: `144.723274`
- Max shape path: `380.815763`; max implied variance: `97.559272`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.001828} + \underset{(0.000003)}{0.239259}\,\pi_t + \underset{(0.000002)}{0.310740}\,\pi_{t-1} + \underset{(0.005442)}{-0.463672}\,SPF_t + \underset{(0.005442)}{0.053363}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000006)}{0.180775}\,\omega_{p,t} - \underset{(0.000001)}{1.899770}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000002)}{6.451490} + \underset{(0.000005)}{0.955084}\,p_{t-1} + \frac{\underset{(0.000004)}{0.055950}}{2(\underset{(0.000006)}{0.180775})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.000006)}{0.134824} + \underset{(0.000003)}{0.994276}\,n_{t-1} + \frac{\underset{(0.000002)}{0.000015}}{2(\underset{(0.000001)}{1.899770})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.001828 | 0.000002 |
| rho_1 | 0.239259 | 0.000003 |
| rho_2 | 0.310740 | 0.000002 |
| phi_1 | -0.463672 | 0.005442 |
| phi_2 | 0.053363 | 0.005442 |
| p0 | 6.451490 | 0.000002 |
| n0 | 0.134824 | 0.000006 |
| rho_p | 0.955084 | 0.000005 |
| rho_n | 0.994276 | 0.000003 |
| phi_p_plus | 0.055950 | 0.000004 |
| phi_n_minus | 0.000015 | 0.000002 |
| sigma_p | 0.180775 | 0.000006 |
| sigma_n | 1.899770 | 0.000001 |

### Rank 2: Seed 45, Draw 6

- LogLik: `-82.714014`; AIC: `191.428028`; BIC: `235.246323`
- Max shape path: `515.371106`; max implied variance: `1.220388`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000001)}{-0.007064} + \underset{(0.000001)}{0.073481}\,\pi_t + \underset{(0.000003)}{0.254380}\,\pi_{t-1} + \underset{(0.000001)}{0.783877}\,SPF_t + \underset{(0.000001)}{-0.501450}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000001)}{0.023533}\,\omega_{p,t} - \underset{(0.000002)}{0.140828}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000000)}{6.002783} + \underset{(0.000002)}{0.810085}\,p_{t-1} + \frac{\underset{(0.000001)}{0.034336}}{2(\underset{(0.000001)}{0.023533})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.000000)}{6.370434} + \underset{(0.000001)}{0.864843}\,n_{t-1} + \frac{\underset{(0.000000)}{0.017451}}{2(\underset{(0.000002)}{0.140828})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | -0.007064 | 0.000001 |
| rho_1 | 0.073481 | 0.000001 |
| rho_2 | 0.254380 | 0.000003 |
| phi_1 | 0.783877 | 0.000001 |
| phi_2 | -0.501450 | 0.000001 |
| p0 | 6.002783 | 0.000000 |
| n0 | 6.370434 | 0.000000 |
| rho_p | 0.810085 | 0.000002 |
| rho_n | 0.864843 | 0.000001 |
| phi_p_plus | 0.034336 | 0.000001 |
| phi_n_minus | 0.017451 | 0.000000 |
| sigma_p | 0.023533 | 0.000001 |
| sigma_n | 0.140828 | 0.000002 |

### Rank 3: Seed 13, Draw 37

- LogLik: `-159.918150`; AIC: `345.836301`; BIC: `389.654595`
- Max shape path: `6782.964492`; max implied variance: `2.472044`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.000002)}{0.094374} + \underset{(0.000002)}{0.267877}\,\pi_t + \underset{(0.000002)}{-0.104270}\,\pi_{t-1} + \underset{(0.000002)}{0.880123}\,SPF_t + \underset{(0.000002)}{-0.105546}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.000002)}{0.010884}\,\omega_{p,t} - \underset{(0.000002)}{0.160732}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000001)}{1.736754} + \underset{(0.000002)}{0.090900}\,p_{t-1} + \frac{\underset{(0.000002)}{0.355108}}{2(\underset{(0.000002)}{0.010884})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.000001)}{6.467638} + \underset{(0.000002)}{0.633481}\,n_{t-1} + \frac{\underset{(0.000001)}{0.256760}}{2(\underset{(0.000002)}{0.160732})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.094374 | 0.000002 |
| rho_1 | 0.267877 | 0.000002 |
| rho_2 | -0.104270 | 0.000002 |
| phi_1 | 0.880123 | 0.000002 |
| phi_2 | -0.105546 | 0.000002 |
| p0 | 1.736754 | 0.000001 |
| n0 | 6.467638 | 0.000001 |
| rho_p | 0.090900 | 0.000002 |
| rho_n | 0.633481 | 0.000002 |
| phi_p_plus | 0.355108 | 0.000002 |
| phi_n_minus | 0.256760 | 0.000001 |
| sigma_p | 0.010884 | 0.000002 |
| sigma_n | 0.160732 | 0.000002 |

### Rank 4: Seed 8, Draw 33

- LogLik: `-167.901751`; AIC: `361.803502`; BIC: `405.621796`
- Max shape path: `47.754787`; max implied variance: `12.087754`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.016995)}{0.124550} + \underset{(0.008781)}{0.261655}\,\pi_t + \underset{(0.060726)}{0.174155}\,\pi_{t-1} + \underset{(0.050780)}{0.292225}\,SPF_t + \underset{(0.038629)}{0.174432}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.019692)}{0.148550}\,\omega_{p,t} - \underset{(0.159126)}{0.989584}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.141959)}{1.267720} + \underset{(0.000649)}{0.707487}\,p_{t-1} + \frac{\underset{(0.013969)}{0.425057}}{2(\underset{(0.019692)}{0.148550})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.006026)}{0.050768} + \underset{(0.060366)}{0.131322}\,n_{t-1} + \frac{\underset{(0.128550)}{1.264011}}{2(\underset{(0.159126)}{0.989584})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124550 | 0.016995 |
| rho_1 | 0.261655 | 0.008781 |
| rho_2 | 0.174155 | 0.060726 |
| phi_1 | 0.292225 | 0.050780 |
| phi_2 | 0.174432 | 0.038629 |
| p0 | 1.267720 | 0.141959 |
| n0 | 0.050768 | 0.006026 |
| rho_p | 0.707487 | 0.000649 |
| rho_n | 0.131322 | 0.060366 |
| phi_p_plus | 0.425057 | 0.013969 |
| phi_n_minus | 1.264011 | 0.128550 |
| sigma_p | 0.148550 | 0.019692 |
| sigma_n | 0.989584 | 0.159126 |

### Rank 5: Seed 32, Draw 32

- LogLik: `-167.901752`; AIC: `361.803504`; BIC: `405.621798`
- Max shape path: `47.736459`; max implied variance: `12.081901`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072577)}{0.124533} + \underset{(0.076981)}{0.261732}\,\pi_t + \underset{(0.076327)}{0.174066}\,\pi_{t-1} + \underset{(0.377686)}{0.292003}\,SPF_t + \underset{(0.318274)}{0.174748}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042736)}{0.148577}\,\omega_{p,t} - \underset{(0.555036)}{0.989715}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.810365)}{1.267306} + \underset{(0.121578)}{0.707515}\,p_{t-1} + \frac{\underset{(0.175091)}{0.425126}}{2(\underset{(0.042736)}{0.148577})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.047201)}{0.050780} + \underset{(0.058250)}{0.131377}\,n_{t-1} + \frac{\underset{(0.835230)}{1.263420}}{2(\underset{(0.555036)}{0.989715})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124533 | 0.072577 |
| rho_1 | 0.261732 | 0.076981 |
| rho_2 | 0.174066 | 0.076327 |
| phi_1 | 0.292003 | 0.377686 |
| phi_2 | 0.174748 | 0.318274 |
| p0 | 1.267306 | 0.810365 |
| n0 | 0.050780 | 0.047201 |
| rho_p | 0.707515 | 0.121578 |
| rho_n | 0.131377 | 0.058250 |
| phi_p_plus | 0.425126 | 0.175091 |
| phi_n_minus | 1.263420 | 0.835230 |
| sigma_p | 0.148577 | 0.042736 |
| sigma_n | 0.989715 | 0.555036 |

### Rank 6: Seed 7, Draw 25

- LogLik: `-167.901755`; AIC: `361.803511`; BIC: `405.621805`
- Max shape path: `47.759695`; max implied variance: `12.100717`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.002267)}{0.124573} + \underset{(0.084638)}{0.261633}\,\pi_t + \underset{(0.000276)}{0.174174}\,\pi_{t-1} + \underset{(0.050495)}{0.292275}\,SPF_t + \underset{(0.067665)}{0.174353}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.024226)}{0.148562}\,\omega_{p,t} - \underset{(0.121635)}{0.990249}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.000301)}{1.267135} + \underset{(0.032981)}{0.707523}\,p_{t-1} + \frac{\underset{(0.007320)}{0.425139}}{2(\underset{(0.024226)}{0.148562})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.028318)}{0.050646} + \underset{(0.054256)}{0.131534}\,n_{t-1} + \frac{\underset{(0.544482)}{1.265385}}{2(\underset{(0.121635)}{0.990249})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124573 | 0.002267 |
| rho_1 | 0.261633 | 0.084638 |
| rho_2 | 0.174174 | 0.000276 |
| phi_1 | 0.292275 | 0.050495 |
| phi_2 | 0.174353 | 0.067665 |
| p0 | 1.267135 | 0.000301 |
| n0 | 0.050646 | 0.028318 |
| rho_p | 0.707523 | 0.032981 |
| rho_n | 0.131534 | 0.054256 |
| phi_p_plus | 0.425139 | 0.007320 |
| phi_n_minus | 1.265385 | 0.544482 |
| sigma_p | 0.148562 | 0.024226 |
| sigma_n | 0.990249 | 0.121635 |

### Rank 7: Seed 28, Draw 27

- LogLik: `-167.901756`; AIC: `361.803511`; BIC: `405.621806`
- Max shape path: `47.763705`; max implied variance: `12.069869`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.099061)}{0.124573} + \underset{(0.004487)}{0.261789}\,\pi_t + \underset{(0.067415)}{0.174054}\,\pi_{t-1} + \underset{(0.213797)}{0.292584}\,SPF_t + \underset{(0.216877)}{0.174061}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.072735)}{0.148540}\,\omega_{p,t} - \underset{(0.466171)}{0.988623}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.759641)}{1.268064} + \underset{(0.122268)}{0.707434}\,p_{t-1} + \frac{\underset{(0.166455)}{0.425200}}{2(\underset{(0.072735)}{0.148540})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.042916)}{0.050829} + \underset{(0.052576)}{0.131423}\,n_{t-1} + \frac{\underset{(0.726216)}{1.262143}}{2(\underset{(0.466171)}{0.988623})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124573 | 0.099061 |
| rho_1 | 0.261789 | 0.004487 |
| rho_2 | 0.174054 | 0.067415 |
| phi_1 | 0.292584 | 0.213797 |
| phi_2 | 0.174061 | 0.216877 |
| p0 | 1.268064 | 0.759641 |
| n0 | 0.050829 | 0.042916 |
| rho_p | 0.707434 | 0.122268 |
| rho_n | 0.131423 | 0.052576 |
| phi_p_plus | 0.425200 | 0.166455 |
| phi_n_minus | 1.262143 | 0.726216 |
| sigma_p | 0.148540 | 0.072735 |
| sigma_n | 0.988623 | 0.466171 |

### Rank 8: Seed 23, Draw 25

- LogLik: `-167.901758`; AIC: `361.803515`; BIC: `405.621810`
- Max shape path: `47.813652`; max implied variance: `12.095096`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072158)}{0.124621} + \underset{(0.075988)}{0.261672}\,\pi_t + \underset{(0.075544)}{0.174131}\,\pi_{t-1} + \underset{(0.367201)}{0.291994}\,SPF_t + \underset{(0.310997)}{0.174594}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042487)}{0.148500}\,\omega_{p,t} - \underset{(0.545668)}{0.990070}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.737588)}{1.269335} + \underset{(0.113201)}{0.707293}\,p_{t-1} + \frac{\underset{(0.163265)}{0.425391}}{2(\underset{(0.042487)}{0.148500})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046395)}{0.050713} + \underset{(0.055739)}{0.131399}\,n_{t-1} + \frac{\underset{(0.763879)}{1.264805}}{2(\underset{(0.545668)}{0.990070})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124621 | 0.072158 |
| rho_1 | 0.261672 | 0.075988 |
| rho_2 | 0.174131 | 0.075544 |
| phi_1 | 0.291994 | 0.367201 |
| phi_2 | 0.174594 | 0.310997 |
| p0 | 1.269335 | 0.737588 |
| n0 | 0.050713 | 0.046395 |
| rho_p | 0.707293 | 0.113201 |
| rho_n | 0.131399 | 0.055739 |
| phi_p_plus | 0.425391 | 0.163265 |
| phi_n_minus | 1.264805 | 0.763879 |
| sigma_p | 0.148500 | 0.042487 |
| sigma_n | 0.990070 | 0.545668 |

### Rank 9: Seed 49, Draw 5

- LogLik: `-167.901758`; AIC: `361.803516`; BIC: `405.621810`
- Max shape path: `47.724522`; max implied variance: `12.095358`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072668)}{0.124502} + \underset{(0.076438)}{0.261705}\,\pi_t + \underset{(0.075986)}{0.174119}\,\pi_{t-1} + \underset{(0.372328)}{0.292451}\,SPF_t + \underset{(0.313272)}{0.174217}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043379)}{0.148586}\,\omega_{p,t} - \underset{(0.535649)}{0.991262}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.829115)}{1.266589} + \underset{(0.122689)}{0.707573}\,p_{t-1} + \frac{\underset{(0.175418)}{0.424959}}{2(\underset{(0.043379)}{0.148586})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.045975)}{0.050668} + \underset{(0.055705)}{0.131352}\,n_{t-1} + \frac{\underset{(0.760760)}{1.264858}}{2(\underset{(0.535649)}{0.991262})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124502 | 0.072668 |
| rho_1 | 0.261705 | 0.076438 |
| rho_2 | 0.174119 | 0.075986 |
| phi_1 | 0.292451 | 0.372328 |
| phi_2 | 0.174217 | 0.313272 |
| p0 | 1.266589 | 0.829115 |
| n0 | 0.050668 | 0.045975 |
| rho_p | 0.707573 | 0.122689 |
| rho_n | 0.131352 | 0.055705 |
| phi_p_plus | 0.424959 | 0.175418 |
| phi_n_minus | 1.264858 | 0.760760 |
| sigma_p | 0.148586 | 0.043379 |
| sigma_n | 0.991262 | 0.535649 |

### Rank 10: Seed 35, Draw 29

- LogLik: `-167.901758`; AIC: `361.803517`; BIC: `405.621811`
- Max shape path: `47.695699`; max implied variance: `12.096702`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071217)}{0.124530} + \underset{(0.076038)}{0.261627}\,\pi_t + \underset{(0.075926)}{0.174267}\,\pi_{t-1} + \underset{(0.373253)}{0.292474}\,SPF_t + \underset{(0.317074)}{0.174085}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.041718)}{0.148632}\,\omega_{p,t} - \underset{(0.558624)}{0.989402}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.689672)}{1.265339} + \underset{(0.111012)}{0.707643}\,p_{t-1} + \frac{\underset{(0.162015)}{0.424926}}{2(\underset{(0.041718)}{0.148632})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046871)}{0.050753} + \underset{(0.059095)}{0.131605}\,n_{t-1} + \frac{\underset{(0.810748)}{1.264897}}{2(\underset{(0.558624)}{0.989402})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124530 | 0.071217 |
| rho_1 | 0.261627 | 0.076038 |
| rho_2 | 0.174267 | 0.075926 |
| phi_1 | 0.292474 | 0.373253 |
| phi_2 | 0.174085 | 0.317074 |
| p0 | 1.265339 | 0.689672 |
| n0 | 0.050753 | 0.046871 |
| rho_p | 0.707643 | 0.111012 |
| rho_n | 0.131605 | 0.059095 |
| phi_p_plus | 0.424926 | 0.162015 |
| phi_n_minus | 1.264897 | 0.810748 |
| sigma_p | 0.148632 | 0.041718 |
| sigma_n | 0.989402 | 0.558624 |

### Rank 11: Seed 12, Draw 22

- LogLik: `-167.901759`; AIC: `361.803518`; BIC: `405.621812`
- Max shape path: `47.686671`; max implied variance: `12.064548`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071824)}{0.124598} + \underset{(0.075930)}{0.261740}\,\pi_t + \underset{(0.075297)}{0.174221}\,\pi_{t-1} + \underset{(0.356205)}{0.292401}\,SPF_t + \underset{(0.300020)}{0.174092}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.041781)}{0.148617}\,\omega_{p,t} - \underset{(0.536494)}{0.988588}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.677038)}{1.265204} + \underset{(0.106992)}{0.707776}\,p_{t-1} + \frac{\underset{(0.155830)}{0.424758}}{2(\underset{(0.041781)}{0.148617})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046341)}{0.050802} + \underset{(0.055684)}{0.131642}\,n_{t-1} + \frac{\underset{(0.745967)}{1.261439}}{2(\underset{(0.536494)}{0.988588})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124598 | 0.071824 |
| rho_1 | 0.261740 | 0.075930 |
| rho_2 | 0.174221 | 0.075297 |
| phi_1 | 0.292401 | 0.356205 |
| phi_2 | 0.174092 | 0.300020 |
| p0 | 1.265204 | 0.677038 |
| n0 | 0.050802 | 0.046341 |
| rho_p | 0.707776 | 0.106992 |
| rho_n | 0.131642 | 0.055684 |
| phi_p_plus | 0.424758 | 0.155830 |
| phi_n_minus | 1.261439 | 0.745967 |
| sigma_p | 0.148617 | 0.041781 |
| sigma_n | 0.988588 | 0.536494 |

### Rank 12: Seed 34, Draw 33

- LogLik: `-167.901759`; AIC: `361.803518`; BIC: `405.621812`
- Max shape path: `47.653514`; max implied variance: `12.101978`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.075376)}{0.124491} + \underset{(0.071455)}{0.261829}\,\pi_t + \underset{(0.078806)}{0.174198}\,\pi_{t-1} + \underset{(0.395545)}{0.292107}\,SPF_t + \underset{(0.341994)}{0.174329}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.047410)}{0.148683}\,\omega_{p,t} - \underset{(0.635483)}{0.990303}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.899761)}{1.264482} + \underset{(0.126002)}{0.707693}\,p_{t-1} + \frac{\underset{(0.178557)}{0.424893}}{2(\underset{(0.047410)}{0.148683})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.051476)}{0.050668} + \underset{(0.059559)}{0.131432}\,n_{t-1} + \frac{\underset{(0.971266)}{1.265525}}{2(\underset{(0.635483)}{0.990303})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124491 | 0.075376 |
| rho_1 | 0.261829 | 0.071455 |
| rho_2 | 0.174198 | 0.078806 |
| phi_1 | 0.292107 | 0.395545 |
| phi_2 | 0.174329 | 0.341994 |
| p0 | 1.264482 | 0.899761 |
| n0 | 0.050668 | 0.051476 |
| rho_p | 0.707693 | 0.126002 |
| rho_n | 0.131432 | 0.059559 |
| phi_p_plus | 0.424893 | 0.178557 |
| phi_n_minus | 1.265525 | 0.971266 |
| sigma_p | 0.148683 | 0.047410 |
| sigma_n | 0.990303 | 0.635483 |

### Rank 13: Seed 3, Draw 17

- LogLik: `-167.901759`; AIC: `361.803519`; BIC: `405.621813`
- Max shape path: `47.693082`; max implied variance: `12.060110`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071474)}{0.124605} + \underset{(0.075799)}{0.261768}\,\pi_t + \underset{(0.075333)}{0.174142}\,\pi_{t-1} + \underset{(0.362221)}{0.291788}\,SPF_t + \underset{(0.305146)}{0.174746}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.041539)}{0.148657}\,\omega_{p,t} - \underset{(0.495549)}{0.987911}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.653154)}{1.265878} + \underset{(0.105843)}{0.707508}\,p_{t-1} + \frac{\underset{(0.154332)}{0.425195}}{2(\underset{(0.041539)}{0.148657})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.044171)}{0.050872} + \underset{(0.056358)}{0.131552}\,n_{t-1} + \frac{\underset{(0.693521)}{1.261027}}{2(\underset{(0.495549)}{0.987911})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124605 | 0.071474 |
| rho_1 | 0.261768 | 0.075799 |
| rho_2 | 0.174142 | 0.075333 |
| phi_1 | 0.291788 | 0.362221 |
| phi_2 | 0.174746 | 0.305146 |
| p0 | 1.265878 | 0.653154 |
| n0 | 0.050872 | 0.044171 |
| rho_p | 0.707508 | 0.105843 |
| rho_n | 0.131552 | 0.056358 |
| phi_p_plus | 0.425195 | 0.154332 |
| phi_n_minus | 1.261027 | 0.693521 |
| sigma_p | 0.148657 | 0.041539 |
| sigma_n | 0.987911 | 0.495549 |

### Rank 14: Seed 7, Draw 32

- LogLik: `-167.901760`; AIC: `361.803519`; BIC: `405.621813`
- Max shape path: `47.798676`; max implied variance: `12.077718`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.074192)}{0.124556} + \underset{(0.076688)}{0.261667}\,\pi_t + \underset{(0.077632)}{0.174183}\,\pi_{t-1} + \underset{(0.394813)}{0.291970}\,SPF_t + \underset{(0.333291)}{0.174643}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.044240)}{0.148549}\,\omega_{p,t} - \underset{(0.621084)}{0.989247}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.894395)}{1.269054} + \underset{(0.128061)}{0.707164}\,p_{t-1} + \frac{\underset{(0.181548)}{0.425708}}{2(\underset{(0.044240)}{0.148549})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.050436)}{0.050780} + \underset{(0.058427)}{0.131559}\,n_{t-1} + \frac{\underset{(0.912397)}{1.262913}}{2(\underset{(0.621084)}{0.989247})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124556 | 0.074192 |
| rho_1 | 0.261667 | 0.076688 |
| rho_2 | 0.174183 | 0.077632 |
| phi_1 | 0.291970 | 0.394813 |
| phi_2 | 0.174643 | 0.333291 |
| p0 | 1.269054 | 0.894395 |
| n0 | 0.050780 | 0.050436 |
| rho_p | 0.707164 | 0.128061 |
| rho_n | 0.131559 | 0.058427 |
| phi_p_plus | 0.425708 | 0.181548 |
| phi_n_minus | 1.262913 | 0.912397 |
| sigma_p | 0.148549 | 0.044240 |
| sigma_n | 0.989247 | 0.621084 |

### Rank 15: Seed 6, Draw 14

- LogLik: `-167.901760`; AIC: `361.803519`; BIC: `405.621814`
- Max shape path: `47.733328`; max implied variance: `12.110893`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.073134)}{0.124568} + \underset{(0.076459)}{0.261722}\,\pi_t + \underset{(0.075929)}{0.174267}\,\pi_{t-1} + \underset{(0.362214)}{0.291532}\,SPF_t + \underset{(0.306092)}{0.174856}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042045)}{0.148579}\,\omega_{p,t} - \underset{(0.599375)}{0.991062}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.762042)}{1.267014} + \underset{(0.115865)}{0.707498}\,p_{t-1} + \frac{\underset{(0.166492)}{0.425046}}{2(\underset{(0.042045)}{0.148579})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.049334)}{0.050663} + \underset{(0.057819)}{0.131434}\,n_{t-1} + \frac{\underset{(0.889234)}{1.266423}}{2(\underset{(0.599375)}{0.991062})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124568 | 0.073134 |
| rho_1 | 0.261722 | 0.076459 |
| rho_2 | 0.174267 | 0.075929 |
| phi_1 | 0.291532 | 0.362214 |
| phi_2 | 0.174856 | 0.306092 |
| p0 | 1.267014 | 0.762042 |
| n0 | 0.050663 | 0.049334 |
| rho_p | 0.707498 | 0.115865 |
| rho_n | 0.131434 | 0.057819 |
| phi_p_plus | 0.425046 | 0.166492 |
| phi_n_minus | 1.266423 | 0.889234 |
| sigma_p | 0.148579 | 0.042045 |
| sigma_n | 0.991062 | 0.599375 |

### Rank 16: Seed 37, Draw 15

- LogLik: `-167.901760`; AIC: `361.803521`; BIC: `405.621815`
- Max shape path: `47.704362`; max implied variance: `12.081168`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.077660)}{0.124597} + \underset{(0.076799)}{0.261610}\,\pi_t + \underset{(0.079870)}{0.174079}\,\pi_{t-1} + \underset{(0.413743)}{0.293309}\,SPF_t + \underset{(0.353284)}{0.173530}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039238)}{0.148665}\,\omega_{p,t} - \underset{(0.584681)}{0.989783}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.660563)}{1.266763} + \underset{(0.131745)}{0.707366}\,p_{t-1} + \frac{\underset{(0.185929)}{0.425382}}{2(\underset{(0.039238)}{0.148665})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.048113)}{0.050671} + \underset{(0.057787)}{0.131601}\,n_{t-1} + \frac{\underset{(0.864969)}{1.263307}}{2(\underset{(0.584681)}{0.989783})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124597 | 0.077660 |
| rho_1 | 0.261610 | 0.076799 |
| rho_2 | 0.174079 | 0.079870 |
| phi_1 | 0.293309 | 0.413743 |
| phi_2 | 0.173530 | 0.353284 |
| p0 | 1.266763 | 0.660563 |
| n0 | 0.050671 | 0.048113 |
| rho_p | 0.707366 | 0.131745 |
| rho_n | 0.131601 | 0.057787 |
| phi_p_plus | 0.425382 | 0.185929 |
| phi_n_minus | 1.263307 | 0.864969 |
| sigma_p | 0.148665 | 0.039238 |
| sigma_n | 0.989783 | 0.584681 |

### Rank 17: Seed 4, Draw 21

- LogLik: `-167.901761`; AIC: `361.803521`; BIC: `405.621816`
- Max shape path: `47.761229`; max implied variance: `12.086573`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.097239)}{0.124576} + \underset{(0.236975)}{0.261763}\,\pi_t + \underset{(0.116767)}{0.174093}\,\pi_{t-1} + \underset{(0.004007)}{0.292339}\,SPF_t + \underset{(0.271069)}{0.174200}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.086188)}{0.148570}\,\omega_{p,t} - \underset{(0.995330)}{0.989670}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(1.186212)}{1.268004} + \underset{(0.006278)}{0.707358}\,p_{t-1} + \frac{\underset{(0.000037)}{0.425289}}{2(\underset{(0.086188)}{0.148570})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.069200)}{0.050708} + \underset{(0.139878)}{0.131788}\,n_{t-1} + \frac{\underset{(2.406895)}{1.263934}}{2(\underset{(0.995330)}{0.989670})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124576 | 0.097239 |
| rho_1 | 0.261763 | 0.236975 |
| rho_2 | 0.174093 | 0.116767 |
| phi_1 | 0.292339 | 0.004007 |
| phi_2 | 0.174200 | 0.271069 |
| p0 | 1.268004 | 1.186212 |
| n0 | 0.050708 | 0.069200 |
| rho_p | 0.707358 | 0.006278 |
| rho_n | 0.131788 | 0.139878 |
| phi_p_plus | 0.425289 | 0.000037 |
| phi_n_minus | 1.263934 | 2.406895 |
| sigma_p | 0.148570 | 0.086188 |
| sigma_n | 0.989670 | 0.995330 |

### Rank 18: Seed 3, Draw 8

- LogLik: `-167.901761`; AIC: `361.803522`; BIC: `405.621816`
- Max shape path: `47.735259`; max implied variance: `12.101691`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.073382)}{0.124514} + \underset{(0.076166)}{0.261642}\,\pi_t + \underset{(0.075363)}{0.174193}\,\pi_{t-1} + \underset{(0.375159)}{0.292462}\,SPF_t + \underset{(0.317338)}{0.174210}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043463)}{0.148582}\,\omega_{p,t} - \underset{(0.558411)}{0.990425}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.812464)}{1.265839} + \underset{(0.120031)}{0.707713}\,p_{t-1} + \frac{\underset{(0.172047)}{0.424984}}{2(\underset{(0.043463)}{0.148582})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046849)}{0.050706} + \underset{(0.059705)}{0.131059}\,n_{t-1} + \frac{\underset{(0.841699)}{1.265473}}{2(\underset{(0.558411)}{0.990425})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124514 | 0.073382 |
| rho_1 | 0.261642 | 0.076166 |
| rho_2 | 0.174193 | 0.075363 |
| phi_1 | 0.292462 | 0.375159 |
| phi_2 | 0.174210 | 0.317338 |
| p0 | 1.265839 | 0.812464 |
| n0 | 0.050706 | 0.046849 |
| rho_p | 0.707713 | 0.120031 |
| rho_n | 0.131059 | 0.059705 |
| phi_p_plus | 0.424984 | 0.172047 |
| phi_n_minus | 1.265473 | 0.841699 |
| sigma_p | 0.148582 | 0.043463 |
| sigma_n | 0.990425 | 0.558411 |

### Rank 19: Seed 47, Draw 23

- LogLik: `-167.901762`; AIC: `361.803524`; BIC: `405.621818`
- Max shape path: `47.803776`; max implied variance: `12.101864`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.073332)}{0.124608} + \underset{(0.076458)}{0.261640}\,\pi_t + \underset{(0.077541)}{0.174136}\,\pi_{t-1} + \underset{(0.399448)}{0.292686}\,SPF_t + \underset{(0.336854)}{0.173927}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043116)}{0.148548}\,\omega_{p,t} - \underset{(0.561509)}{0.990170}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.850109)}{1.269338} + \underset{(0.125356)}{0.707145}\,p_{t-1} + \frac{\underset{(0.180099)}{0.425645}}{2(\underset{(0.043116)}{0.148548})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.046926)}{0.050703} + \underset{(0.057381)}{0.131362}\,n_{t-1} + \frac{\underset{(0.822263)}{1.265533}}{2(\underset{(0.561509)}{0.990170})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124608 | 0.073332 |
| rho_1 | 0.261640 | 0.076458 |
| rho_2 | 0.174136 | 0.077541 |
| phi_1 | 0.292686 | 0.399448 |
| phi_2 | 0.173927 | 0.336854 |
| p0 | 1.269338 | 0.850109 |
| n0 | 0.050703 | 0.046926 |
| rho_p | 0.707145 | 0.125356 |
| rho_n | 0.131362 | 0.057381 |
| phi_p_plus | 0.425645 | 0.180099 |
| phi_n_minus | 1.265533 | 0.822263 |
| sigma_p | 0.148548 | 0.043116 |
| sigma_n | 0.990170 | 0.561509 |

### Rank 20: Seed 41, Draw 26

- LogLik: `-167.901762`; AIC: `361.803524`; BIC: `405.621819`
- Max shape path: `47.771920`; max implied variance: `12.089646`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071782)}{0.124601} + \underset{(0.075994)}{0.261758}\,\pi_t + \underset{(0.075425)}{0.174113}\,\pi_{t-1} + \underset{(0.346128)}{0.292189}\,SPF_t + \underset{(0.291820)}{0.174373}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042410)}{0.148558}\,\omega_{p,t} - \underset{(0.507479)}{0.988287}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= \underset{(0.734806)}{1.268410} + \underset{(0.112367)}{0.707317}\,p_{t-1} + \frac{\underset{(0.162791)}{0.425418}}{2(\underset{(0.042410)}{0.148558})^2}\,(u_{t-1}^+)^2,\\
n_t &= \underset{(0.045220)}{0.050836} + \underset{(0.055578)}{0.131213}\,n_{t-1} + \frac{\underset{(0.711223)}{1.264234}}{2(\underset{(0.507479)}{0.988287})^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.124601 | 0.071782 |
| rho_1 | 0.261758 | 0.075994 |
| rho_2 | 0.174113 | 0.075425 |
| phi_1 | 0.292189 | 0.346128 |
| phi_2 | 0.174373 | 0.291820 |
| p0 | 1.268410 | 0.734806 |
| n0 | 0.050836 | 0.045220 |
| rho_p | 0.707317 | 0.112367 |
| rho_n | 0.131213 | 0.055578 |
| phi_p_plus | 0.425418 | 0.162791 |
| phi_n_minus | 1.264234 | 0.711223 |
| sigma_p | 0.148558 | 0.042410 |
| sigma_n | 0.988287 | 0.507479 |
