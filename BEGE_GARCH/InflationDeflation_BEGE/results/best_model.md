```{raw:typst}
#set page(margin: auto)
```

# Inflation/Deflation BEGE-GJR Best Model Summary

Generated: `2026-06-03T14:05:19`
Total estimations: `8000`
Converged estimations: `7310`
Eligible estimations for best-model selection: `7273`

Saved likelihoods are recomputed from the stored parameter paths before ranking. Large recursive shape states are evaluated by the BEGE saddlepoint density backend; `max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.

Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, finite positive shape paths, positive conditional variance paths, EWMA implied-variance bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance constraints. Corrected log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `5` admissible estimates by corrected log likelihood.

```{note}
Flagged 106 estimate(s) with corrected log likelihood above `-150` for manual review. These rows remain eligible if they pass the admissibility checks.
```

## constant

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 22 | 15 | -157.812140 | 331.624281 | 358.589385 | 6312.732437 | 3.103930 | no |
| 2 | 9 | 39 | -177.882535 | 371.765069 | 398.730173 | 125.826145 | 12.973910 | no |
| 3 | 17 | 10 | -177.882535 | 371.765069 | 398.730174 | 125.832345 | 12.972768 | no |
| 4 | 34 | 10 | -177.882535 | 371.765070 | 398.730174 | 125.813979 | 12.972114 | no |
| 5 | 1 | 28 | -177.882535 | 371.765070 | 398.730174 | 125.819901 | 12.972392 | no |

### Rank 1: Seed 22, Draw 15

- LogLik: `-157.812140`; AIC: `331.624281`; BIC: `358.589385`
- Max shape path: `6312.732437`; max implied variance: `3.103930`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.190782\,\omega_{p,t} - 0.018752\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 4.756536 + 0.775695\,p_{t-1} + \frac{0.135188}{2(0.190782)^2}\,(u_{t-1}^+)^2,\\
n_t &= 9.503285 + 0.553343\,n_{t-1} + \frac{0.274879}{2(0.018752)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 4.756536 |
| n0 | 9.503285 |
| rho_p | 0.775695 |
| rho_n | 0.553343 |
| phi_p_plus | 0.135188 |
| phi_n_minus | 0.274879 |
| sigma_p | 0.190782 |
| sigma_n | 0.018752 |

### Rank 2: Seed 9, Draw 39

- LogLik: `-177.882535`; AIC: `371.765069`; BIC: `398.730173`
- Max shape path: `125.826145`; max implied variance: `12.973910`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.141989\,\omega_{p,t} - 1.638761\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.212923 + 0.382743\,p_{t-1} + \frac{0.862959}{2(0.141989)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.048702 + 0.101885\,n_{t-1} + \frac{1.566022}{2(1.638761)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 3.212923 |
| n0 | 0.048702 |
| rho_p | 0.382743 |
| rho_n | 0.101885 |
| phi_p_plus | 0.862959 |
| phi_n_minus | 1.566022 |
| sigma_p | 0.141989 |
| sigma_n | 1.638761 |

### Rank 3: Seed 17, Draw 10

- LogLik: `-177.882535`; AIC: `371.765069`; BIC: `398.730174`
- Max shape path: `125.832345`; max implied variance: `12.972768`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.141986\,\omega_{p,t} - 1.638802\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.213698 + 0.382658\,p_{t-1} + \frac{0.863017}{2(0.141986)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.048710 + 0.101899\,n_{t-1} + \frac{1.565882}{2(1.638802)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 3.213698 |
| n0 | 0.048710 |
| rho_p | 0.382658 |
| rho_n | 0.101899 |
| phi_p_plus | 0.863017 |
| phi_n_minus | 1.565882 |
| sigma_p | 0.141986 |
| sigma_n | 1.638802 |

### Rank 4: Seed 34, Draw 10

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.813979`; max implied variance: `12.972114`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.142002\,\omega_{p,t} - 1.638479\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.212323 + 0.382723\,p_{t-1} + \frac{0.863054}{2(0.142002)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.048718 + 0.101884\,n_{t-1} + \frac{1.565800}{2(1.638479)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 3.212323 |
| n0 | 0.048718 |
| rho_p | 0.382723 |
| rho_n | 0.101884 |
| phi_p_plus | 0.863054 |
| phi_n_minus | 1.565800 |
| sigma_p | 0.142002 |
| sigma_n | 1.638479 |

### Rank 5: Seed 1, Draw 28

- LogLik: `-177.882535`; AIC: `371.765070`; BIC: `398.730174`
- Max shape path: `125.819901`; max implied variance: `12.972392`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.141995\,\omega_{p,t} - 1.638416\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 3.212419 + 0.382775\,p_{t-1} + \frac{0.862971}{2(0.141995)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.048720 + 0.101841\,n_{t-1} + \frac{1.565833}{2(1.638416)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| p0 | 3.212419 |
| n0 | 0.048720 |
| rho_p | 0.382775 |
| rho_n | 0.101841 |
| phi_p_plus | 0.862971 |
| phi_n_minus | 1.565833 |
| sigma_p | 0.141995 |
| sigma_n | 1.638416 |

## ARX(1,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 3 | 40 | -157.746763 | 337.493526 | 374.570544 | 509.003819 | 12.048805 | no |
| 2 | 6 | 34 | -170.519038 | 363.038075 | 400.115094 | 60.171153 | 9.274512 | no |
| 3 | 38 | 24 | -170.519039 | 363.038078 | 400.115096 | 60.208945 | 9.271182 | no |
| 4 | 4 | 28 | -170.519039 | 363.038079 | 400.115097 | 60.205258 | 9.275218 | no |
| 5 | 37 | 32 | -170.519040 | 363.038079 | 400.115098 | 60.163204 | 9.269005 | no |

### Rank 1: Seed 3, Draw 40

- LogLik: `-157.746763`; AIC: `337.493526`; BIC: `374.570544`
- Max shape path: `509.003819`; max implied variance: `12.048805`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.052645 + 0.379824\,\pi_t + 0.615393\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.048940\,\omega_{p,t} - 0.160659\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.176419 + 0.690579\,p_{t-1} + \frac{0.482635}{2(0.048940)^2}\,(u_{t-1}^+)^2,\\
n_t &= 2.397874 + 0.000000\,n_{t-1} + \frac{1.406178}{2(0.160659)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.052645 |
| rho_1 | 0.379824 |
| phi_1 | 0.615393 |
| p0 | 9.176419 |
| n0 | 2.397874 |
| rho_p | 0.690579 |
| rho_n | 0.000000 |
| phi_p_plus | 0.482635 |
| phi_n_minus | 1.406178 |
| sigma_p | 0.048940 |
| sigma_n | 0.160659 |

### Rank 2: Seed 6, Draw 34

- LogLik: `-170.519038`; AIC: `363.038075`; BIC: `400.115094`
- Max shape path: `60.171153`; max implied variance: `9.274512`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.132322 + 0.292944\,\pi_t + 0.624955\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.158209\,\omega_{p,t} - 0.846063\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.966086 + 0.560865\,p_{t-1} + \frac{0.633747}{2(0.158209)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.077597 + 0.025731\,n_{t-1} + \frac{1.059762}{2(0.846063)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.132322 |
| rho_1 | 0.292944 |
| phi_1 | 0.624955 |
| p0 | 1.966086 |
| n0 | 0.077597 |
| rho_p | 0.560865 |
| rho_n | 0.025731 |
| phi_p_plus | 0.633747 |
| phi_n_minus | 1.059762 |
| sigma_p | 0.158209 |
| sigma_n | 0.846063 |

### Rank 3: Seed 38, Draw 24

- LogLik: `-170.519039`; AIC: `363.038078`; BIC: `400.115096`
- Max shape path: `60.208945`; max implied variance: `9.271182`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.132351 + 0.292936\,\pi_t + 0.624945\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.158192\,\omega_{p,t} - 0.845739\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.967814 + 0.560661\,p_{t-1} + \frac{0.634042}{2(0.158192)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.077643 + 0.025697\,n_{t-1} + \frac{1.059366}{2(0.845739)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.132351 |
| rho_1 | 0.292936 |
| phi_1 | 0.624945 |
| p0 | 1.967814 |
| n0 | 0.077643 |
| rho_p | 0.560661 |
| rho_n | 0.025697 |
| phi_p_plus | 0.634042 |
| phi_n_minus | 1.059366 |
| sigma_p | 0.158192 |
| sigma_n | 0.845739 |

### Rank 4: Seed 4, Draw 28

- LogLik: `-170.519039`; AIC: `363.038079`; BIC: `400.115097`
- Max shape path: `60.205258`; max implied variance: `9.275218`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.132252 + 0.292866\,\pi_t + 0.625090\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.158150\,\omega_{p,t} - 0.845887\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.966445 + 0.560990\,p_{t-1} + \frac{0.633611}{2(0.158150)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.077669 + 0.025601\,n_{t-1} + \frac{1.059862}{2(0.845887)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.132252 |
| rho_1 | 0.292866 |
| phi_1 | 0.625090 |
| p0 | 1.966445 |
| n0 | 0.077669 |
| rho_p | 0.560990 |
| rho_n | 0.025601 |
| phi_p_plus | 0.633611 |
| phi_n_minus | 1.059862 |
| sigma_p | 0.158150 |
| sigma_n | 0.845887 |

### Rank 5: Seed 37, Draw 32

- LogLik: `-170.519040`; AIC: `363.038079`; BIC: `400.115098`
- Max shape path: `60.163204`; max implied variance: `9.269005`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.132291 + 0.292969\,\pi_t + 0.624927\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.158217\,\omega_{p,t} - 0.846327\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.965564 + 0.560926\,p_{t-1} + \frac{0.633698}{2(0.158217)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.077619 + 0.025908\,n_{t-1} + \frac{1.059114}{2(0.846327)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.132291 |
| rho_1 | 0.292969 |
| phi_1 | 0.624927 |
| p0 | 1.965564 |
| n0 | 0.077619 |
| rho_p | 0.560926 |
| rho_n | 0.025908 |
| phi_p_plus | 0.633698 |
| phi_n_minus | 1.059114 |
| sigma_p | 0.158217 |
| sigma_n | 0.846327 |

## ARX(2,1)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 47 | 31 | -140.463596 | 304.927192 | 345.374849 | 409.057396 | 2.576010 | yes |
| 2 | 46 | 7 | -153.423467 | 330.846934 | 371.294590 | 3060.322794 | 1.080083 | no |
| 3 | 5 | 16 | -165.929106 | 355.858212 | 396.305868 | 2486.656875 | 17.282857 | no |
| 4 | 34 | 29 | -168.076017 | 360.152033 | 400.599690 | 49.082928 | 12.034470 | no |
| 5 | 11 | 34 | -168.076017 | 360.152034 | 400.599690 | 49.081808 | 12.036596 | no |

### Rank 1: Seed 47, Draw 31

- LogLik: `-140.463596`; AIC: `304.927192`; BIC: `345.374849`
- Max shape path: `409.057396`; max implied variance: `2.576010`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.028273 + 0.345298\,\pi_t + 0.053284\,\pi_{t-1} + 0.670520\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.049190\,\omega_{p,t} - 0.190090\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 4.831677 + 0.785878\,p_{t-1} + \frac{0.392201}{2(0.049190)^2}\,(u_{t-1}^+)^2,\\
n_t &= 2.032389 + 0.216245\,n_{t-1} + \frac{0.251030}{2(0.190090)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.028273 |
| rho_1 | 0.345298 |
| rho_2 | 0.053284 |
| phi_1 | 0.670520 |
| p0 | 4.831677 |
| n0 | 2.032389 |
| rho_p | 0.785878 |
| rho_n | 0.216245 |
| phi_p_plus | 0.392201 |
| phi_n_minus | 0.251030 |
| sigma_p | 0.049190 |
| sigma_n | 0.190090 |

### Rank 2: Seed 46, Draw 7

- LogLik: `-153.423467`; AIC: `330.846934`; BIC: `371.294590`
- Max shape path: `3060.322794`; max implied variance: `1.080083`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.073322 + 0.270793\,\pi_t + 0.072230\,\pi_{t-1} + 0.647075\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.017058\,\omega_{p,t} - 0.080176\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 9.799563 + 0.795104\,p_{t-1} + \frac{0.363053}{2(0.017058)^2}\,(u_{t-1}^+)^2,\\
n_t &= 3.077954 + 0.895644\,n_{t-1} + \frac{0.000000}{2(0.080176)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.073322 |
| rho_1 | 0.270793 |
| rho_2 | 0.072230 |
| phi_1 | 0.647075 |
| p0 | 9.799563 |
| n0 | 3.077954 |
| rho_p | 0.795104 |
| rho_n | 0.895644 |
| phi_p_plus | 0.363053 |
| phi_n_minus | 0.000000 |
| sigma_p | 0.017058 |
| sigma_n | 0.080176 |

### Rank 3: Seed 5, Draw 16

- LogLik: `-165.929106`; AIC: `355.858212`; BIC: `396.305868`
- Max shape path: `2486.656875`; max implied variance: `17.282857`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.011164 + 0.527992\,\pi_t + 0.139352\,\pi_{t-1} + 0.344646\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.025389\,\omega_{p,t} - 0.160542\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 0.079741 + 0.483727\,p_{t-1} + \frac{0.657988}{2(0.025389)^2}\,(u_{t-1}^+)^2,\\
n_t &= 6.516500 + 0.021806\,n_{t-1} + \frac{1.913988}{2(0.160542)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.011164 |
| rho_1 | 0.527992 |
| rho_2 | 0.139352 |
| phi_1 | 0.344646 |
| p0 | 0.079741 |
| n0 | 6.516500 |
| rho_p | 0.483727 |
| rho_n | 0.021806 |
| phi_p_plus | 0.657988 |
| phi_n_minus | 1.913988 |
| sigma_p | 0.025389 |
| sigma_n | 0.160542 |

### Rank 4: Seed 34, Draw 29

- LogLik: `-168.076017`; AIC: `360.152033`; BIC: `400.599690`
- Max shape path: `49.082928`; max implied variance: `12.034470`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.126408 + 0.256718\,\pi_t + 0.167312\,\pi_{t-1} + 0.480898\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.149209\,\omega_{p,t} - 0.986608\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.361429 + 0.691663\,p_{t-1} + \frac{0.446940}{2(0.149209)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.048911 + 0.136762\,n_{t-1} + \frac{1.260904}{2(0.986608)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.126408 |
| rho_1 | 0.256718 |
| rho_2 | 0.167312 |
| phi_1 | 0.480898 |
| p0 | 1.361429 |
| n0 | 0.048911 |
| rho_p | 0.691663 |
| rho_n | 0.136762 |
| phi_p_plus | 0.446940 |
| phi_n_minus | 1.260904 |
| sigma_p | 0.149209 |
| sigma_n | 0.986608 |

### Rank 5: Seed 11, Draw 34

- LogLik: `-168.076017`; AIC: `360.152034`; BIC: `400.599690`
- Max shape path: `49.081808`; max implied variance: `12.036596`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.126442 + 0.256680\,\pi_t + 0.167318\,\pi_{t-1} + 0.480903\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.149219\,\omega_{p,t} - 0.985755\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.361226 + 0.691659\,p_{t-1} + \frac{0.446957}{2(0.149219)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.048951 + 0.136650\,n_{t-1} + \frac{1.261127}{2(0.985755)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.126442 |
| rho_1 | 0.256680 |
| rho_2 | 0.167318 |
| phi_1 | 0.480903 |
| p0 | 1.361226 |
| n0 | 0.048951 |
| rho_p | 0.691659 |
| rho_n | 0.136650 |
| phi_p_plus | 0.446957 |
| phi_n_minus | 1.261127 |
| sigma_p | 0.149219 |
| sigma_n | 0.985755 |

## ARX(2,2)

Top 5 admissible estimates ranked by corrected log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 4 | 20 | -37.452490 | 100.904979 | 144.723274 | 380.815763 | 97.559272 | yes |
| 2 | 45 | 6 | -82.714014 | 191.428028 | 235.246323 | 515.371106 | 1.220388 | yes |
| 3 | 13 | 37 | -159.918150 | 345.836301 | 389.654595 | 6782.964492 | 2.472044 | no |
| 4 | 8 | 33 | -167.901751 | 361.803502 | 405.621796 | 47.754787 | 12.087754 | no |
| 5 | 32 | 32 | -167.901752 | 361.803504 | 405.621798 | 47.736459 | 12.081901 | no |

### Rank 1: Seed 4, Draw 20

- LogLik: `-37.452490`; AIC: `100.904979`; BIC: `144.723274`
- Max shape path: `380.815763`; max implied variance: `97.559272`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.001828 + 0.239259\,\pi_t + 0.310740\,\pi_{t-1} + -0.463672\,SPF_t + 0.053363\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.180775\,\omega_{p,t} - 1.899770\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.451490 + 0.955084\,p_{t-1} + \frac{0.055950}{2(0.180775)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.134824 + 0.994276\,n_{t-1} + \frac{0.000015}{2(1.899770)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.001828 |
| rho_1 | 0.239259 |
| rho_2 | 0.310740 |
| phi_1 | -0.463672 |
| phi_2 | 0.053363 |
| p0 | 6.451490 |
| n0 | 0.134824 |
| rho_p | 0.955084 |
| rho_n | 0.994276 |
| phi_p_plus | 0.055950 |
| phi_n_minus | 0.000015 |
| sigma_p | 0.180775 |
| sigma_n | 1.899770 |

### Rank 2: Seed 45, Draw 6

- LogLik: `-82.714014`; AIC: `191.428028`; BIC: `235.246323`
- Max shape path: `515.371106`; max implied variance: `1.220388`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = -0.007064 + 0.073481\,\pi_t + 0.254380\,\pi_{t-1} + 0.783877\,SPF_t + -0.501450\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.023533\,\omega_{p,t} - 0.140828\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 6.002783 + 0.810085\,p_{t-1} + \frac{0.034336}{2(0.023533)^2}\,(u_{t-1}^+)^2,\\
n_t &= 6.370434 + 0.864843\,n_{t-1} + \frac{0.017451}{2(0.140828)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | -0.007064 |
| rho_1 | 0.073481 |
| rho_2 | 0.254380 |
| phi_1 | 0.783877 |
| phi_2 | -0.501450 |
| p0 | 6.002783 |
| n0 | 6.370434 |
| rho_p | 0.810085 |
| rho_n | 0.864843 |
| phi_p_plus | 0.034336 |
| phi_n_minus | 0.017451 |
| sigma_p | 0.023533 |
| sigma_n | 0.140828 |

### Rank 3: Seed 13, Draw 37

- LogLik: `-159.918150`; AIC: `345.836301`; BIC: `389.654595`
- Max shape path: `6782.964492`; max implied variance: `2.472044`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.094374 + 0.267877\,\pi_t + -0.104270\,\pi_{t-1} + 0.880123\,SPF_t + -0.105546\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.010884\,\omega_{p,t} - 0.160732\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.736754 + 0.090900\,p_{t-1} + \frac{0.355108}{2(0.010884)^2}\,(u_{t-1}^+)^2,\\
n_t &= 6.467638 + 0.633481\,n_{t-1} + \frac{0.256760}{2(0.160732)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.094374 |
| rho_1 | 0.267877 |
| rho_2 | -0.104270 |
| phi_1 | 0.880123 |
| phi_2 | -0.105546 |
| p0 | 1.736754 |
| n0 | 6.467638 |
| rho_p | 0.090900 |
| rho_n | 0.633481 |
| phi_p_plus | 0.355108 |
| phi_n_minus | 0.256760 |
| sigma_p | 0.010884 |
| sigma_n | 0.160732 |

### Rank 4: Seed 8, Draw 33

- LogLik: `-167.901751`; AIC: `361.803502`; BIC: `405.621796`
- Max shape path: `47.754787`; max implied variance: `12.087754`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.124550 + 0.261655\,\pi_t + 0.174155\,\pi_{t-1} + 0.292225\,SPF_t + 0.174432\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.148550\,\omega_{p,t} - 0.989584\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.267720 + 0.707487\,p_{t-1} + \frac{0.425057}{2(0.148550)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.050768 + 0.131322\,n_{t-1} + \frac{1.264011}{2(0.989584)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.124550 |
| rho_1 | 0.261655 |
| rho_2 | 0.174155 |
| phi_1 | 0.292225 |
| phi_2 | 0.174432 |
| p0 | 1.267720 |
| n0 | 0.050768 |
| rho_p | 0.707487 |
| rho_n | 0.131322 |
| phi_p_plus | 0.425057 |
| phi_n_minus | 1.264011 |
| sigma_p | 0.148550 |
| sigma_n | 0.989584 |

### Rank 5: Seed 32, Draw 32

- LogLik: `-167.901752`; AIC: `361.803504`; BIC: `405.621798`
- Max shape path: `47.736459`; max implied variance: `12.081901`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.124533 + 0.261732\,\pi_t + 0.174066\,\pi_{t-1} + 0.292003\,SPF_t + 0.174748\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.148577\,\omega_{p,t} - 0.989715\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).
\end{aligned}
$$

$$
\begin{aligned}
p_t &= 1.267306 + 0.707515\,p_{t-1} + \frac{0.425126}{2(0.148577)^2}\,(u_{t-1}^+)^2,\\
n_t &= 0.050780 + 0.131377\,n_{t-1} + \frac{1.263420}{2(0.989715)^2}\,(u_{t-1}^-)^2
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.124533 |
| rho_1 | 0.261732 |
| rho_2 | 0.174066 |
| phi_1 | 0.292003 |
| phi_2 | 0.174748 |
| p0 | 1.267306 |
| n0 | 0.050780 |
| rho_p | 0.707515 |
| rho_n | 0.131377 |
| phi_p_plus | 0.425126 |
| phi_n_minus | 1.263420 |
| sigma_p | 0.148577 |
| sigma_n | 0.989715 |
