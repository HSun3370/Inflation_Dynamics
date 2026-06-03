```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-06-03T14:02:40`
Total estimations: `8000`
Successful estimations: `7997`
Eligible estimations for best-model selection: `7997`

Selection screen: successful optimizer status, finite positive BEGE parameters, documented parameter bounds, EWMA implied-variance bounds, positive conditional variance, and mean-process stationarity. Log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `5` admissible estimates by log likelihood.

## constant

Top 5 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 46 | 11 | -199.843879 | 407.687759 | 421.170311 | 0.697933 | no |
| 2 | 8 | 9 | -199.843879 | 407.687759 | 421.170311 | 0.697928 | no |
| 3 | 36 | 27 | -199.843879 | 407.687759 | 421.170311 | 0.697937 | no |
| 4 | 18 | 27 | -199.843879 | 407.687759 | 421.170311 | 0.697937 | no |
| 5 | 1 | 8 | -199.843879 | 407.687759 | 421.170311 | 0.697934 | no |

### Rank 1: Seed 46, Draw 11

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697933`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.298261\,\omega_{p,t} - 1.572532\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.675902,\qquad \bar{n} = 0.185974,\\
\operatorname{Var}_t(u_t) &= (0.298261)^2\,2.675902 + (1.572532)^2\,0.185974.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| shape_p | 2.675902 |
| shape_n | 0.185974 |
| sigma_p | 0.298261 |
| sigma_n | 1.572532 |

### Rank 2: Seed 8, Draw 9

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697928`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.298260\,\omega_{p,t} - 1.572519\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.675907,\qquad \bar{n} = 0.185975,\\
\operatorname{Var}_t(u_t) &= (0.298260)^2\,2.675907 + (1.572519)^2\,0.185975.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| shape_p | 2.675907 |
| shape_n | 0.185975 |
| sigma_p | 0.298260 |
| sigma_n | 1.572519 |

### Rank 3: Seed 36, Draw 27

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697937`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.298263\,\omega_{p,t} - 1.572554\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.675859,\qquad \bar{n} = 0.185970,\\
\operatorname{Var}_t(u_t) &= (0.298263)^2\,2.675859 + (1.572554)^2\,0.185970.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| shape_p | 2.675859 |
| shape_n | 0.185970 |
| sigma_p | 0.298263 |
| sigma_n | 1.572554 |

### Rank 4: Seed 18, Draw 27

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697937`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.298257\,\omega_{p,t} - 1.572553\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.675954,\qquad \bar{n} = 0.185971,\\
\operatorname{Var}_t(u_t) &= (0.298257)^2\,2.675954 + (1.572553)^2\,0.185971.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| shape_p | 2.675954 |
| shape_n | 0.185971 |
| sigma_p | 0.298257 |
| sigma_n | 1.572553 |

### Rank 5: Seed 1, Draw 8

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697934`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = SPF_t + u_{t+1}
$$

No estimated mean-process coefficients.

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.298257\,\omega_{p,t} - 1.572549\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.675939,\qquad \bar{n} = 0.185971,\\
\operatorname{Var}_t(u_t) &= (0.298257)^2\,2.675939 + (1.572549)^2\,0.185971.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| shape_p | 2.675939 |
| shape_n | 0.185971 |
| sigma_p | 0.298257 |
| sigma_n | 1.572549 |

## ARX(1,1)

Top 5 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 35 | 31 | -184.396177 | 382.792353 | 406.386819 | 0.394478 | no |
| 2 | 19 | 23 | -184.396177 | 382.792353 | 406.386819 | 0.394457 | no |
| 3 | 47 | 25 | -184.396177 | 382.792353 | 406.386819 | 0.394466 | no |
| 4 | 41 | 11 | -184.396177 | 382.792353 | 406.386819 | 0.394476 | no |
| 5 | 47 | 12 | -184.396177 | 382.792353 | 406.386819 | 0.394500 | no |

### Rank 1: Seed 35, Draw 31

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394478`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.056000 + 0.323710\,\pi_t + 0.737795\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.285677\,\omega_{p,t} - 0.800287\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.627705,\qquad \bar{n} = 0.281090,\\
\operatorname{Var}_t(u_t) &= (0.285677)^2\,2.627705 + (0.800287)^2\,0.281090.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.056000 |
| rho_1 | 0.323710 |
| phi_1 | 0.737795 |
| shape_p | 2.627705 |
| shape_n | 0.281090 |
| sigma_p | 0.285677 |
| sigma_n | 0.800287 |

### Rank 2: Seed 19, Draw 23

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394457`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.056029 + 0.323706\,\pi_t + 0.737776\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.285660\,\omega_{p,t} - 0.800195\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.627966,\qquad \bar{n} = 0.281128,\\
\operatorname{Var}_t(u_t) &= (0.285660)^2\,2.627966 + (0.800195)^2\,0.281128.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.056029 |
| rho_1 | 0.323706 |
| phi_1 | 0.737776 |
| shape_p | 2.627966 |
| shape_n | 0.281128 |
| sigma_p | 0.285660 |
| sigma_n | 0.800195 |

### Rank 3: Seed 47, Draw 25

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394466`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.056023 + 0.323696\,\pi_t + 0.737796\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.285666\,\omega_{p,t} - 0.800203\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.627917,\qquad \bar{n} = 0.281131,\\
\operatorname{Var}_t(u_t) &= (0.285666)^2\,2.627917 + (0.800203)^2\,0.281131.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.056023 |
| rho_1 | 0.323696 |
| phi_1 | 0.737796 |
| shape_p | 2.627917 |
| shape_n | 0.281131 |
| sigma_p | 0.285666 |
| sigma_n | 0.800203 |

### Rank 4: Seed 41, Draw 11

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394476`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.056017 + 0.323714\,\pi_t + 0.737766\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.285679\,\omega_{p,t} - 0.800304\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.627694,\qquad \bar{n} = 0.281072,\\
\operatorname{Var}_t(u_t) &= (0.285679)^2\,2.627694 + (0.800304)^2\,0.281072.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.056017 |
| rho_1 | 0.323714 |
| phi_1 | 0.737766 |
| shape_p | 2.627694 |
| shape_n | 0.281072 |
| sigma_p | 0.285679 |
| sigma_n | 0.800304 |

### Rank 5: Seed 47, Draw 12

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394500`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.056023 + 0.323710\,\pi_t + 0.737769\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.285684\,\omega_{p,t} - 0.800308\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.627702,\qquad \bar{n} = 0.281096,\\
\operatorname{Var}_t(u_t) &= (0.285684)^2\,2.627702 + (0.800308)^2\,0.281096.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.056023 |
| rho_1 | 0.323710 |
| phi_1 | 0.737769 |
| shape_p | 2.627702 |
| shape_n | 0.281096 |
| sigma_p | 0.285684 |
| sigma_n | 0.800308 |

## ARX(2,1)

Top 5 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 27 | 15 | -181.884824 | 379.769649 | 406.734753 | 0.398211 | no |
| 2 | 18 | 26 | -181.884824 | 379.769649 | 406.734753 | 0.398207 | no |
| 3 | 26 | 6 | -181.884824 | 379.769649 | 406.734753 | 0.398205 | no |
| 4 | 26 | 16 | -181.884824 | 379.769649 | 406.734753 | 0.398202 | no |
| 5 | 31 | 34 | -181.884824 | 379.769649 | 406.734753 | 0.398207 | no |

### Rank 1: Seed 27, Draw 15

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398211`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.039240 + 0.316830\,\pi_t + 0.189143\,\pi_{t-1} + 0.539230\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.246533\,\omega_{p,t} - 0.928387\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 3.317099,\qquad \bar{n} = 0.228103,\\
\operatorname{Var}_t(u_t) &= (0.246533)^2\,3.317099 + (0.928387)^2\,0.228103.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.039240 |
| rho_1 | 0.316830 |
| rho_2 | 0.189143 |
| phi_1 | 0.539230 |
| shape_p | 3.317099 |
| shape_n | 0.228103 |
| sigma_p | 0.246533 |
| sigma_n | 0.928387 |

### Rank 2: Seed 18, Draw 26

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398207`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.039251 + 0.316833\,\pi_t + 0.189140\,\pi_{t-1} + 0.539222\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.246534\,\omega_{p,t} - 0.928414\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 3.317080,\qquad \bar{n} = 0.228085,\\
\operatorname{Var}_t(u_t) &= (0.246534)^2\,3.317080 + (0.928414)^2\,0.228085.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.039251 |
| rho_1 | 0.316833 |
| rho_2 | 0.189140 |
| phi_1 | 0.539222 |
| shape_p | 3.317080 |
| shape_n | 0.228085 |
| sigma_p | 0.246534 |
| sigma_n | 0.928414 |

### Rank 3: Seed 26, Draw 6

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398205`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.039235 + 0.316822\,\pi_t + 0.189147\,\pi_{t-1} + 0.539234\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.246529\,\omega_{p,t} - 0.928451\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 3.317117,\qquad \bar{n} = 0.228070,\\
\operatorname{Var}_t(u_t) &= (0.246529)^2\,3.317117 + (0.928451)^2\,0.228070.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.039235 |
| rho_1 | 0.316822 |
| rho_2 | 0.189147 |
| phi_1 | 0.539234 |
| shape_p | 3.317117 |
| shape_n | 0.228070 |
| sigma_p | 0.246529 |
| sigma_n | 0.928451 |

### Rank 4: Seed 26, Draw 16

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398202`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.039250 + 0.316807\,\pi_t + 0.189146\,\pi_{t-1} + 0.539245\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.246526\,\omega_{p,t} - 0.928377\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 3.317293,\qquad \bar{n} = 0.228098,\\
\operatorname{Var}_t(u_t) &= (0.246526)^2\,3.317293 + (0.928377)^2\,0.228098.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.039250 |
| rho_1 | 0.316807 |
| rho_2 | 0.189146 |
| phi_1 | 0.539245 |
| shape_p | 3.317293 |
| shape_n | 0.228098 |
| sigma_p | 0.246526 |
| sigma_n | 0.928377 |

### Rank 5: Seed 31, Draw 34

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398207`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.039227 + 0.316823\,\pi_t + 0.189154\,\pi_{t-1} + 0.539235\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.246524\,\omega_{p,t} - 0.928396\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 3.317327,\qquad \bar{n} = 0.228096,\\
\operatorname{Var}_t(u_t) &= (0.246524)^2\,3.317327 + (0.928396)^2\,0.228096.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.039227 |
| rho_1 | 0.316823 |
| rho_2 | 0.189154 |
| phi_1 | 0.539235 |
| shape_p | 3.317327 |
| shape_n | 0.228096 |
| sigma_p | 0.246524 |
| sigma_n | 0.928396 |

## ARX(2,2)

Top 5 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 18 | 1 | -181.688415 | 381.376830 | 411.712572 | 0.395222 | no |
| 2 | 1 | 20 | -181.688415 | 381.376830 | 411.712572 | 0.395120 | no |
| 3 | 13 | 30 | -181.688415 | 381.376830 | 411.712572 | 0.395224 | no |
| 4 | 31 | 28 | -181.688415 | 381.376831 | 411.712573 | 0.395088 | no |
| 5 | 10 | 38 | -181.688416 | 381.376832 | 411.712575 | 0.395182 | no |

### Rank 1: Seed 18, Draw 1

- LogLik: `-181.688415`; AIC: `381.376830`; BIC: `411.712572`
- Implied variance: `0.395222`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.004171 + 0.323173\,\pi_t + 0.163315\,\pi_{t-1} + 0.410717\,SPF_t + 0.194172\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.271204\,\omega_{p,t} - 0.840368\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.664771,\qquad \bar{n} = 0.282099,\\
\operatorname{Var}_t(u_t) &= (0.271204)^2\,2.664771 + (0.840368)^2\,0.282099.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.004171 |
| rho_1 | 0.323173 |
| rho_2 | 0.163315 |
| phi_1 | 0.410717 |
| phi_2 | 0.194172 |
| shape_p | 2.664771 |
| shape_n | 0.282099 |
| sigma_p | 0.271204 |
| sigma_n | 0.840368 |

### Rank 2: Seed 1, Draw 20

- LogLik: `-181.688415`; AIC: `381.376830`; BIC: `411.712572`
- Implied variance: `0.395120`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.004263 + 0.323278\,\pi_t + 0.163371\,\pi_{t-1} + 0.410449\,SPF_t + 0.194160\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.271157\,\omega_{p,t} - 0.839817\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.665408,\qquad \bar{n} = 0.282355,\\
\operatorname{Var}_t(u_t) &= (0.271157)^2\,2.665408 + (0.839817)^2\,0.282355.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.004263 |
| rho_1 | 0.323278 |
| rho_2 | 0.163371 |
| phi_1 | 0.410449 |
| phi_2 | 0.194160 |
| shape_p | 2.665408 |
| shape_n | 0.282355 |
| sigma_p | 0.271157 |
| sigma_n | 0.839817 |

### Rank 3: Seed 13, Draw 30

- LogLik: `-181.688415`; AIC: `381.376830`; BIC: `411.712572`
- Implied variance: `0.395224`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.004179 + 0.323231\,\pi_t + 0.163357\,\pi_{t-1} + 0.410464\,SPF_t + 0.194290\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.271228\,\omega_{p,t} - 0.840331\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.664400,\qquad \bar{n} = 0.282117,\\
\operatorname{Var}_t(u_t) &= (0.271228)^2\,2.664400 + (0.840331)^2\,0.282117.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.004179 |
| rho_1 | 0.323231 |
| rho_2 | 0.163357 |
| phi_1 | 0.410464 |
| phi_2 | 0.194290 |
| shape_p | 2.664400 |
| shape_n | 0.282117 |
| sigma_p | 0.271228 |
| sigma_n | 0.840331 |

### Rank 4: Seed 31, Draw 28

- LogLik: `-181.688415`; AIC: `381.376831`; BIC: `411.712573`
- Implied variance: `0.395088`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.004208 + 0.323297\,\pi_t + 0.163370\,\pi_{t-1} + 0.410312\,SPF_t + 0.194314\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.271150\,\omega_{p,t} - 0.839907\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.665276,\qquad \bar{n} = 0.282277,\\
\operatorname{Var}_t(u_t) &= (0.271150)^2\,2.665276 + (0.839907)^2\,0.282277.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.004208 |
| rho_1 | 0.323297 |
| rho_2 | 0.163370 |
| phi_1 | 0.410312 |
| phi_2 | 0.194314 |
| shape_p | 2.665276 |
| shape_n | 0.282277 |
| sigma_p | 0.271150 |
| sigma_n | 0.839907 |

### Rank 5: Seed 10, Draw 38

- LogLik: `-181.688416`; AIC: `381.376832`; BIC: `411.712575`
- Implied variance: `0.395182`
- Selection diagnostics: `eligible`

Mean process:

$$
\pi_{t+1} = 0.004169 + 0.323162\,\pi_t + 0.163312\,\pi_{t-1} + 0.410592\,SPF_t + 0.194342\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= 0.271107\,\omega_{p,t} - 0.840465\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= 2.666229,\qquad \bar{n} = 0.282024,\\
\operatorname{Var}_t(u_t) &= (0.271107)^2\,2.666229 + (0.840465)^2\,0.282024.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate |
|---|---:|
| c | 0.004169 |
| rho_1 | 0.323162 |
| rho_2 | 0.163312 |
| phi_1 | 0.410592 |
| phi_2 | 0.194342 |
| shape_p | 2.666229 |
| shape_n | 0.282024 |
| sigma_p | 0.271107 |
| sigma_n | 0.840465 |
