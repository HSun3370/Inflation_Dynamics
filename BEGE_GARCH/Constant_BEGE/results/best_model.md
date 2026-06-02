```{raw:typst}
#set page(margin: auto)
```

# Constant BEGE Best Model Summary

Generated: `2026-06-02T11:22:02`
Total estimations: `8000`
Successful estimations: `7997`
Eligible estimations for best-model selection: `7997`

Selection screen: successful optimizer status, finite positive BEGE parameters, documented parameter bounds, EWMA implied-variance bounds, positive conditional variance, and mean-process stationarity. Log likelihoods above `-150` are flagged for review but are not excluded by this threshold.
Each mean-process section reports the top `20` admissible estimates by log likelihood. Standard errors are shown below substituted equation coefficients in parentheses.

## constant

Top 20 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 46 | 11 | -199.843879 | 407.687759 | 421.170311 | 0.697933 | no | `computed` |
| 2 | 8 | 9 | -199.843879 | 407.687759 | 421.170311 | 0.697928 | no | `computed` |
| 3 | 36 | 27 | -199.843879 | 407.687759 | 421.170311 | 0.697937 | no | `computed` |
| 4 | 18 | 27 | -199.843879 | 407.687759 | 421.170311 | 0.697937 | no | `computed` |
| 5 | 1 | 8 | -199.843879 | 407.687759 | 421.170311 | 0.697934 | no | `computed` |
| 6 | 11 | 9 | -199.843879 | 407.687759 | 421.170311 | 0.697949 | no | `computed` |
| 7 | 27 | 34 | -199.843879 | 407.687759 | 421.170311 | 0.697955 | no | `computed` |
| 8 | 4 | 14 | -199.843879 | 407.687759 | 421.170311 | 0.697937 | no | `computed` |
| 9 | 23 | 36 | -199.843879 | 407.687759 | 421.170311 | 0.697933 | no | `computed` |
| 10 | 15 | 39 | -199.843879 | 407.687759 | 421.170311 | 0.697941 | no | `computed` |
| 11 | 12 | 17 | -199.843879 | 407.687759 | 421.170311 | 0.697949 | no | `computed` |
| 12 | 46 | 38 | -199.843879 | 407.687759 | 421.170311 | 0.697947 | no | `computed` |
| 13 | 31 | 27 | -199.843879 | 407.687759 | 421.170311 | 0.697948 | no | `computed` |
| 14 | 3 | 1 | -199.843879 | 407.687759 | 421.170311 | 0.697946 | no | `computed` |
| 15 | 13 | 26 | -199.843879 | 407.687759 | 421.170311 | 0.697929 | no | `computed` |
| 16 | 28 | 32 | -199.843879 | 407.687759 | 421.170311 | 0.697937 | no | `computed` |
| 17 | 14 | 29 | -199.843879 | 407.687759 | 421.170311 | 0.697932 | no | `computed` |
| 18 | 33 | 27 | -199.843879 | 407.687759 | 421.170311 | 0.697941 | no | `computed` |
| 19 | 26 | 33 | -199.843879 | 407.687759 | 421.170311 | 0.697928 | no | `computed` |
| 20 | 20 | 7 | -199.843879 | 407.687759 | 421.170311 | 0.697925 | no | `computed` |

### Rank 1: Seed 46, Draw 11

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697933`
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
u_t &= \underset{(0.049917)}{0.298261}\,\omega_{p,t} - \underset{(0.932097)}{1.572532}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.614931)}{2.675902},\qquad \bar{n} = \underset{(0.119081)}{0.185974},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.049917)}{0.298261})^2\,\underset{(0.614931)}{2.675902} + (\underset{(0.932097)}{1.572532})^2\,\underset{(0.119081)}{0.185974}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675902 | 0.614931 |
| shape_n | 0.185974 | 0.119081 |
| sigma_p | 0.298261 | 0.049917 |
| sigma_n | 1.572532 | 0.932097 |

### Rank 2: Seed 8, Draw 9

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697928`
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
u_t &= \underset{(0.050692)}{0.298260}\,\omega_{p,t} - \underset{(0.928528)}{1.572519}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.626216)}{2.675907},\qquad \bar{n} = \underset{(0.118700)}{0.185975},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.050692)}{0.298260})^2\,\underset{(0.626216)}{2.675907} + (\underset{(0.928528)}{1.572519})^2\,\underset{(0.118700)}{0.185975}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675907 | 0.626216 |
| shape_n | 0.185975 | 0.118700 |
| sigma_p | 0.298260 | 0.050692 |
| sigma_n | 1.572519 | 0.928528 |

### Rank 3: Seed 36, Draw 27

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697937`
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
u_t &= \underset{(0.051235)}{0.298263}\,\omega_{p,t} - \underset{(0.932799)}{1.572554}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.634409)}{2.675859},\qquad \bar{n} = \underset{(0.119230)}{0.185970},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051235)}{0.298263})^2\,\underset{(0.634409)}{2.675859} + (\underset{(0.932799)}{1.572554})^2\,\underset{(0.119230)}{0.185970}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675859 | 0.634409 |
| shape_n | 0.185970 | 0.119230 |
| sigma_p | 0.298263 | 0.051235 |
| sigma_n | 1.572554 | 0.932799 |

### Rank 4: Seed 18, Draw 27

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697937`
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
u_t &= \underset{(0.050392)}{0.298257}\,\omega_{p,t} - \underset{(0.924197)}{1.572553}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.623668)}{2.675954},\qquad \bar{n} = \underset{(0.118046)}{0.185971},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.050392)}{0.298257})^2\,\underset{(0.623668)}{2.675954} + (\underset{(0.924197)}{1.572553})^2\,\underset{(0.118046)}{0.185971}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675954 | 0.623668 |
| shape_n | 0.185971 | 0.118046 |
| sigma_p | 0.298257 | 0.050392 |
| sigma_n | 1.572553 | 0.924197 |

### Rank 5: Seed 1, Draw 8

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697934`
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
u_t &= \underset{(0.051416)}{0.298257}\,\omega_{p,t} - \underset{(0.928523)}{1.572549}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.637648)}{2.675939},\qquad \bar{n} = \underset{(0.118734)}{0.185971},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051416)}{0.298257})^2\,\underset{(0.637648)}{2.675939} + (\underset{(0.928523)}{1.572549})^2\,\underset{(0.118734)}{0.185971}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675939 | 0.637648 |
| shape_n | 0.185971 | 0.118734 |
| sigma_p | 0.298257 | 0.051416 |
| sigma_n | 1.572549 | 0.928523 |

### Rank 6: Seed 11, Draw 9

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697949`
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
u_t &= \underset{(0.051225)}{0.298264}\,\omega_{p,t} - \underset{(0.930891)}{1.572580}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.631142)}{2.675861},\qquad \bar{n} = \underset{(0.119125)}{0.185968},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051225)}{0.298264})^2\,\underset{(0.631142)}{2.675861} + (\underset{(0.930891)}{1.572580})^2\,\underset{(0.119125)}{0.185968}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675861 | 0.631142 |
| shape_n | 0.185968 | 0.119125 |
| sigma_p | 0.298264 | 0.051225 |
| sigma_n | 1.572580 | 0.930891 |

### Rank 7: Seed 27, Draw 34

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697955`
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
u_t &= \underset{(0.053163)}{0.298262}\,\omega_{p,t} - \underset{(0.933807)}{1.572609}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.667177)}{2.675892},\qquad \bar{n} = \underset{(0.119336)}{0.185964},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.053163)}{0.298262})^2\,\underset{(0.667177)}{2.675892} + (\underset{(0.933807)}{1.572609})^2\,\underset{(0.119336)}{0.185964}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675892 | 0.667177 |
| shape_n | 0.185964 | 0.119336 |
| sigma_p | 0.298262 | 0.053163 |
| sigma_n | 1.572609 | 0.933807 |

### Rank 8: Seed 4, Draw 14

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697937`
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
u_t &= \underset{(0.051563)}{0.298260}\,\omega_{p,t} - \underset{(0.924957)}{1.572539}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.639861)}{2.675910},\qquad \bar{n} = \underset{(0.118286)}{0.185974},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051563)}{0.298260})^2\,\underset{(0.639861)}{2.675910} + (\underset{(0.924957)}{1.572539})^2\,\underset{(0.118286)}{0.185974}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675910 | 0.639861 |
| shape_n | 0.185974 | 0.118286 |
| sigma_p | 0.298260 | 0.051563 |
| sigma_n | 1.572539 | 0.924957 |

### Rank 9: Seed 23, Draw 36

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697933`
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
u_t &= \underset{(0.051699)}{0.298257}\,\omega_{p,t} - \underset{(0.921386)}{1.572532}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.643291)}{2.675951},\qquad \bar{n} = \underset{(0.117821)}{0.185974},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051699)}{0.298257})^2\,\underset{(0.643291)}{2.675951} + (\underset{(0.921386)}{1.572532})^2\,\underset{(0.117821)}{0.185974}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675951 | 0.643291 |
| shape_n | 0.185974 | 0.117821 |
| sigma_p | 0.298257 | 0.051699 |
| sigma_n | 1.572532 | 0.921386 |

### Rank 10: Seed 15, Draw 39

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697941`
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
u_t &= \underset{(0.051825)}{0.298257}\,\omega_{p,t} - \underset{(0.924363)}{1.572552}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.644474)}{2.675956},\qquad \bar{n} = \underset{(0.118175)}{0.185972},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051825)}{0.298257})^2\,\underset{(0.644474)}{2.675956} + (\underset{(0.924363)}{1.572552})^2\,\underset{(0.118175)}{0.185972}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675956 | 0.644474 |
| shape_n | 0.185972 | 0.118175 |
| sigma_p | 0.298257 | 0.051825 |
| sigma_n | 1.572552 | 0.924363 |

### Rank 11: Seed 12, Draw 17

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697949`
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
u_t &= \underset{(0.051267)}{0.298257}\,\omega_{p,t} - \underset{(0.930618)}{1.572589}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.635970)}{2.675955},\qquad \bar{n} = \underset{(0.118941)}{0.185967},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051267)}{0.298257})^2\,\underset{(0.635970)}{2.675955} + (\underset{(0.930618)}{1.572589})^2\,\underset{(0.118941)}{0.185967}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675955 | 0.635970 |
| shape_n | 0.185967 | 0.118941 |
| sigma_p | 0.298257 | 0.051267 |
| sigma_n | 1.572589 | 0.930618 |

### Rank 12: Seed 46, Draw 38

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697947`
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
u_t &= \underset{(0.051000)}{0.298260}\,\omega_{p,t} - \underset{(0.930145)}{1.572592}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.631768)}{2.675909},\qquad \bar{n} = \underset{(0.118867)}{0.185966},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051000)}{0.298260})^2\,\underset{(0.631768)}{2.675909} + (\underset{(0.930145)}{1.572592})^2\,\underset{(0.118867)}{0.185966}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675909 | 0.631768 |
| shape_n | 0.185966 | 0.118867 |
| sigma_p | 0.298260 | 0.051000 |
| sigma_n | 1.572592 | 0.930145 |

### Rank 13: Seed 31, Draw 27

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697948`
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
u_t &= \underset{(0.049987)}{0.298262}\,\omega_{p,t} - \underset{(0.922687)}{1.572582}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.616126)}{2.675880},\qquad \bar{n} = \underset{(0.117933)}{0.185968},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.049987)}{0.298262})^2\,\underset{(0.616126)}{2.675880} + (\underset{(0.922687)}{1.572582})^2\,\underset{(0.117933)}{0.185968}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675880 | 0.616126 |
| shape_n | 0.185968 | 0.117933 |
| sigma_p | 0.298262 | 0.049987 |
| sigma_n | 1.572582 | 0.922687 |

### Rank 14: Seed 3, Draw 1

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697946`
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
u_t &= \underset{(0.053636)}{0.298262}\,\omega_{p,t} - \underset{(0.923207)}{1.572596}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.673837)}{2.675886},\qquad \bar{n} = \underset{(0.118091)}{0.185964},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.053636)}{0.298262})^2\,\underset{(0.673837)}{2.675886} + (\underset{(0.923207)}{1.572596})^2\,\underset{(0.118091)}{0.185964}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675886 | 0.673837 |
| shape_n | 0.185964 | 0.118091 |
| sigma_p | 0.298262 | 0.053636 |
| sigma_n | 1.572596 | 0.923207 |

### Rank 15: Seed 13, Draw 26

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697929`
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
u_t &= \underset{(0.049982)}{0.298260}\,\omega_{p,t} - \underset{(0.917337)}{1.572508}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.617056)}{2.675918},\qquad \bar{n} = \underset{(0.117222)}{0.185978},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.049982)}{0.298260})^2\,\underset{(0.617056)}{2.675918} + (\underset{(0.917337)}{1.572508})^2\,\underset{(0.117222)}{0.185978}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675918 | 0.617056 |
| shape_n | 0.185978 | 0.117222 |
| sigma_p | 0.298260 | 0.049982 |
| sigma_n | 1.572508 | 0.917337 |

### Rank 16: Seed 28, Draw 32

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697937`
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
u_t &= \underset{(0.054155)}{0.298264}\,\omega_{p,t} - \underset{(0.929896)}{1.572532}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.678795)}{2.675878},\qquad \bar{n} = \underset{(0.119095)}{0.185974},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.054155)}{0.298264})^2\,\underset{(0.678795)}{2.675878} + (\underset{(0.929896)}{1.572532})^2\,\underset{(0.119095)}{0.185974}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675878 | 0.678795 |
| shape_n | 0.185974 | 0.119095 |
| sigma_p | 0.298264 | 0.054155 |
| sigma_n | 1.572532 | 0.929896 |

### Rank 17: Seed 14, Draw 29

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697932`
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
u_t &= \underset{(0.051438)}{0.298256}\,\omega_{p,t} - \underset{(0.926219)}{1.572538}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.638546)}{2.675971},\qquad \bar{n} = \underset{(0.118406)}{0.185972},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.051438)}{0.298256})^2\,\underset{(0.638546)}{2.675971} + (\underset{(0.926219)}{1.572538})^2\,\underset{(0.118406)}{0.185972}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675971 | 0.638546 |
| shape_n | 0.185972 | 0.118406 |
| sigma_p | 0.298256 | 0.051438 |
| sigma_n | 1.572538 | 0.926219 |

### Rank 18: Seed 33, Draw 27

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697941`
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
u_t &= \underset{(0.053781)}{0.298256}\,\omega_{p,t} - \underset{(0.933032)}{1.572561}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.675485)}{2.675982},\qquad \bar{n} = \underset{(0.119313)}{0.185970},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.053781)}{0.298256})^2\,\underset{(0.675485)}{2.675982} + (\underset{(0.933032)}{1.572561})^2\,\underset{(0.119313)}{0.185970}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675982 | 0.675485 |
| shape_n | 0.185970 | 0.119313 |
| sigma_p | 0.298256 | 0.053781 |
| sigma_n | 1.572561 | 0.933032 |

### Rank 19: Seed 26, Draw 33

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697928`
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
u_t &= \underset{(0.053901)}{0.298256}\,\omega_{p,t} - \underset{(0.936701)}{1.572528}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.674197)}{2.675945},\qquad \bar{n} = \underset{(0.119918)}{0.185974},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.053901)}{0.298256})^2\,\underset{(0.674197)}{2.675945} + (\underset{(0.936701)}{1.572528})^2\,\underset{(0.119918)}{0.185974}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675945 | 0.674197 |
| shape_n | 0.185974 | 0.119918 |
| sigma_p | 0.298256 | 0.053901 |
| sigma_n | 1.572528 | 0.936701 |

### Rank 20: Seed 20, Draw 7

- LogLik: `-199.843879`; AIC: `407.687759`; BIC: `421.170311`
- Implied variance: `0.697925`
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
u_t &= \underset{(0.053207)}{0.298258}\,\omega_{p,t} - \underset{(0.923934)}{1.572505}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.665556)}{2.675923},\qquad \bar{n} = \underset{(0.118246)}{0.185978},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.053207)}{0.298258})^2\,\underset{(0.665556)}{2.675923} + (\underset{(0.923934)}{1.572505})^2\,\underset{(0.118246)}{0.185978}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| shape_p | 2.675923 | 0.665556 |
| shape_n | 0.185978 | 0.118246 |
| sigma_p | 0.298258 | 0.053207 |
| sigma_n | 1.572505 | 0.923934 |

## ARX(1,1)

Top 20 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 35 | 31 | -184.396177 | 382.792353 | 406.386819 | 0.394478 | no | `computed` |
| 2 | 19 | 23 | -184.396177 | 382.792353 | 406.386819 | 0.394457 | no | `computed` |
| 3 | 47 | 25 | -184.396177 | 382.792353 | 406.386819 | 0.394466 | no | `computed` |
| 4 | 41 | 11 | -184.396177 | 382.792353 | 406.386819 | 0.394476 | no | `computed` |
| 5 | 47 | 12 | -184.396177 | 382.792353 | 406.386819 | 0.394500 | no | `computed` |
| 6 | 43 | 23 | -184.396177 | 382.792353 | 406.386819 | 0.394471 | no | `computed` |
| 7 | 36 | 23 | -184.396177 | 382.792353 | 406.386819 | 0.394457 | no | `computed` |
| 8 | 39 | 9 | -184.396177 | 382.792353 | 406.386819 | 0.394485 | no | `computed` |
| 9 | 31 | 29 | -184.396177 | 382.792353 | 406.386820 | 0.394497 | no | `computed` |
| 10 | 11 | 28 | -184.396177 | 382.792353 | 406.386820 | 0.394465 | no | `computed` |
| 11 | 17 | 22 | -184.396177 | 382.792353 | 406.386820 | 0.394465 | no | `computed` |
| 12 | 31 | 10 | -184.396177 | 382.792353 | 406.386820 | 0.394482 | no | `computed` |
| 13 | 29 | 2 | -184.396177 | 382.792353 | 406.386820 | 0.394491 | no | `computed` |
| 14 | 37 | 37 | -184.396177 | 382.792353 | 406.386820 | 0.394444 | no | `computed` |
| 15 | 24 | 16 | -184.396177 | 382.792353 | 406.386820 | 0.394489 | no | `computed` |
| 16 | 31 | 33 | -184.396177 | 382.792353 | 406.386820 | 0.394484 | no | `computed` |
| 17 | 18 | 16 | -184.396177 | 382.792353 | 406.386820 | 0.394485 | no | `computed` |
| 18 | 1 | 36 | -184.396177 | 382.792353 | 406.386820 | 0.394448 | no | `computed` |
| 19 | 9 | 38 | -184.396177 | 382.792353 | 406.386820 | 0.394495 | no | `computed` |
| 20 | 23 | 8 | -184.396177 | 382.792353 | 406.386820 | 0.394466 | no | `computed` |

### Rank 1: Seed 35, Draw 31

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394478`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071173)}{0.056000} + \underset{(0.081125)}{0.323710}\,\pi_t + \underset{(0.112922)}{0.737795}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.056348)}{0.285677}\,\omega_{p,t} - \underset{(0.323040)}{0.800287}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.689177)}{2.627705},\qquad \bar{n} = \underset{(0.153351)}{0.281090},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.056348)}{0.285677})^2\,\underset{(0.689177)}{2.627705} + (\underset{(0.323040)}{0.800287})^2\,\underset{(0.153351)}{0.281090}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056000 | 0.071173 |
| rho_1 | 0.323710 | 0.081125 |
| phi_1 | 0.737795 | 0.112922 |
| shape_p | 2.627705 | 0.689177 |
| shape_n | 0.281090 | 0.153351 |
| sigma_p | 0.285677 | 0.056348 |
| sigma_n | 0.800287 | 0.323040 |

### Rank 2: Seed 19, Draw 23

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394457`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072103)}{0.056029} + \underset{(0.081590)}{0.323706}\,\pi_t + \underset{(0.114323)}{0.737776}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.055284)}{0.285660}\,\omega_{p,t} - \underset{(0.340482)}{0.800195}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.676880)}{2.627966},\qquad \bar{n} = \underset{(0.162613)}{0.281128},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.055284)}{0.285660})^2\,\underset{(0.676880)}{2.627966} + (\underset{(0.340482)}{0.800195})^2\,\underset{(0.162613)}{0.281128}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056029 | 0.072103 |
| rho_1 | 0.323706 | 0.081590 |
| phi_1 | 0.737776 | 0.114323 |
| shape_p | 2.627966 | 0.676880 |
| shape_n | 0.281128 | 0.162613 |
| sigma_p | 0.285660 | 0.055284 |
| sigma_n | 0.800195 | 0.340482 |

### Rank 3: Seed 47, Draw 25

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394466`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071731)}{0.056023} + \underset{(0.081078)}{0.323696}\,\pi_t + \underset{(0.113295)}{0.737796}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.036615)}{0.285666}\,\omega_{p,t} - \underset{(0.318403)}{0.800203}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.395787)}{2.627917},\qquad \bar{n} = \underset{(0.148028)}{0.281131},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.036615)}{0.285666})^2\,\underset{(0.395787)}{2.627917} + (\underset{(0.318403)}{0.800203})^2\,\underset{(0.148028)}{0.281131}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056023 | 0.071731 |
| rho_1 | 0.323696 | 0.081078 |
| phi_1 | 0.737796 | 0.113295 |
| shape_p | 2.627917 | 0.395787 |
| shape_n | 0.281131 | 0.148028 |
| sigma_p | 0.285666 | 0.036615 |
| sigma_n | 0.800203 | 0.318403 |

### Rank 4: Seed 41, Draw 11

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394476`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071645)}{0.056017} + \underset{(0.081224)}{0.323714}\,\pi_t + \underset{(0.114159)}{0.737766}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.054583)}{0.285679}\,\omega_{p,t} - \underset{(0.331881)}{0.800304}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.670546)}{2.627694},\qquad \bar{n} = \underset{(0.159826)}{0.281072},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.054583)}{0.285679})^2\,\underset{(0.670546)}{2.627694} + (\underset{(0.331881)}{0.800304})^2\,\underset{(0.159826)}{0.281072}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056017 | 0.071645 |
| rho_1 | 0.323714 | 0.081224 |
| phi_1 | 0.737766 | 0.114159 |
| shape_p | 2.627694 | 0.670546 |
| shape_n | 0.281072 | 0.159826 |
| sigma_p | 0.285679 | 0.054583 |
| sigma_n | 0.800304 | 0.331881 |

### Rank 5: Seed 47, Draw 12

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394500`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.073310)}{0.056023} + \underset{(0.082077)}{0.323710}\,\pi_t + \underset{(0.117459)}{0.737769}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.113545)}{0.285684}\,\omega_{p,t} - \underset{(0.414283)}{0.800308}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.509339)}{2.627702},\qquad \bar{n} = \underset{(0.217070)}{0.281096},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.113545)}{0.285684})^2\,\underset{(1.509339)}{2.627702} + (\underset{(0.414283)}{0.800308})^2\,\underset{(0.217070)}{0.281096}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056023 | 0.073310 |
| rho_1 | 0.323710 | 0.082077 |
| phi_1 | 0.737769 | 0.117459 |
| shape_p | 2.627702 | 1.509339 |
| shape_n | 0.281096 | 0.217070 |
| sigma_p | 0.285684 | 0.113545 |
| sigma_n | 0.800308 | 0.414283 |

### Rank 6: Seed 43, Draw 23

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394471`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.070861)}{0.056019} + \underset{(0.081055)}{0.323700}\,\pi_t + \underset{(0.112476)}{0.737785}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039168)}{0.285667}\,\omega_{p,t} - \underset{(0.315377)}{0.800210}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.435774)}{2.627931},\qquad \bar{n} = \underset{(0.147697)}{0.281128},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.039168)}{0.285667})^2\,\underset{(0.435774)}{2.627931} + (\underset{(0.315377)}{0.800210})^2\,\underset{(0.147697)}{0.281128}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056019 | 0.070861 |
| rho_1 | 0.323700 | 0.081055 |
| phi_1 | 0.737785 | 0.112476 |
| shape_p | 2.627931 | 0.435774 |
| shape_n | 0.281128 | 0.147697 |
| sigma_p | 0.285667 | 0.039168 |
| sigma_n | 0.800210 | 0.315377 |

### Rank 7: Seed 36, Draw 23

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394457`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071775)}{0.056020} + \underset{(0.081529)}{0.323709}\,\pi_t + \underset{(0.114040)}{0.737780}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.062792)}{0.285676}\,\omega_{p,t} - \underset{(0.349294)}{0.800266}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.786956)}{2.627691},\qquad \bar{n} = \underset{(0.170126)}{0.281077},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.062792)}{0.285676})^2\,\underset{(0.786956)}{2.627691} + (\underset{(0.349294)}{0.800266})^2\,\underset{(0.170126)}{0.281077}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056020 | 0.071775 |
| rho_1 | 0.323709 | 0.081529 |
| phi_1 | 0.737780 | 0.114040 |
| shape_p | 2.627691 | 0.786956 |
| shape_n | 0.281077 | 0.170126 |
| sigma_p | 0.285676 | 0.062792 |
| sigma_n | 0.800266 | 0.349294 |

### Rank 8: Seed 39, Draw 9

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386819`
- Implied variance: `0.394485`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071317)}{0.056016} + \underset{(0.080937)}{0.323706}\,\pi_t + \underset{(0.113069)}{0.737776}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.045693)}{0.285683}\,\omega_{p,t} - \underset{(0.322114)}{0.800288}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.535701)}{2.627708},\qquad \bar{n} = \underset{(0.152119)}{0.281087},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.045693)}{0.285683})^2\,\underset{(0.535701)}{2.627708} + (\underset{(0.322114)}{0.800288})^2\,\underset{(0.152119)}{0.281087}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056016 | 0.071317 |
| rho_1 | 0.323706 | 0.080937 |
| phi_1 | 0.737776 | 0.113069 |
| shape_p | 2.627708 | 0.535701 |
| shape_n | 0.281087 | 0.152119 |
| sigma_p | 0.285683 | 0.045693 |
| sigma_n | 0.800288 | 0.322114 |

### Rank 9: Seed 31, Draw 29

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394497`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071697)}{0.056018} + \underset{(0.081133)}{0.323717}\,\pi_t + \underset{(0.113301)}{0.737775}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.045792)}{0.285686}\,\omega_{p,t} - \underset{(0.334254)}{0.800292}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.537803)}{2.627646},\qquad \bar{n} = \underset{(0.157618)}{0.281103},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.045792)}{0.285686})^2\,\underset{(0.537803)}{2.627646} + (\underset{(0.334254)}{0.800292})^2\,\underset{(0.157618)}{0.281103}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056018 | 0.071697 |
| rho_1 | 0.323717 | 0.081133 |
| phi_1 | 0.737775 | 0.113301 |
| shape_p | 2.627646 | 0.537803 |
| shape_n | 0.281103 | 0.157618 |
| sigma_p | 0.285686 | 0.045792 |
| sigma_n | 0.800292 | 0.334254 |

### Rank 10: Seed 11, Draw 28

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394465`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071920)}{0.056004} + \underset{(0.082015)}{0.323711}\,\pi_t + \underset{(0.115627)}{0.737781}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.085215)}{0.285667}\,\omega_{p,t} - \underset{(0.360698)}{0.800150}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.108749)}{2.627879},\qquad \bar{n} = \underset{(0.182439)}{0.281169},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.085215)}{0.285667})^2\,\underset{(1.108749)}{2.627879} + (\underset{(0.360698)}{0.800150})^2\,\underset{(0.182439)}{0.281169}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056004 | 0.071920 |
| rho_1 | 0.323711 | 0.082015 |
| phi_1 | 0.737781 | 0.115627 |
| shape_p | 2.627879 | 1.108749 |
| shape_n | 0.281169 | 0.182439 |
| sigma_p | 0.285667 | 0.085215 |
| sigma_n | 0.800150 | 0.360698 |

### Rank 11: Seed 17, Draw 22

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394465`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071739)}{0.056025} + \underset{(0.080979)}{0.323699}\,\pi_t + \underset{(0.112908)}{0.737774}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.030609)}{0.285682}\,\omega_{p,t} - \underset{(0.318490)}{0.800229}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.296305)}{2.627592},\qquad \bar{n} = \underset{(0.145184)}{0.281113},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.030609)}{0.285682})^2\,\underset{(0.296305)}{2.627592} + (\underset{(0.318490)}{0.800229})^2\,\underset{(0.145184)}{0.281113}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056025 | 0.071739 |
| rho_1 | 0.323699 | 0.080979 |
| phi_1 | 0.737774 | 0.112908 |
| shape_p | 2.627592 | 0.296305 |
| shape_n | 0.281113 | 0.145184 |
| sigma_p | 0.285682 | 0.030609 |
| sigma_n | 0.800229 | 0.318490 |

### Rank 12: Seed 31, Draw 10

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394482`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.072323)}{0.056004} + \underset{(0.081603)}{0.323712}\,\pi_t + \underset{(0.115159)}{0.737762}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.075673)}{0.285665}\,\omega_{p,t} - \underset{(0.368901)}{0.800265}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.972591)}{2.627873},\qquad \bar{n} = \underset{(0.183280)}{0.281120},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.075673)}{0.285665})^2\,\underset{(0.972591)}{2.627873} + (\underset{(0.368901)}{0.800265})^2\,\underset{(0.183280)}{0.281120}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056004 | 0.072323 |
| rho_1 | 0.323712 | 0.081603 |
| phi_1 | 0.737762 | 0.115159 |
| shape_p | 2.627873 | 0.972591 |
| shape_n | 0.281120 | 0.183280 |
| sigma_p | 0.285665 | 0.075673 |
| sigma_n | 0.800265 | 0.368901 |

### Rank 13: Seed 29, Draw 2

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394491`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071237)}{0.056020} + \underset{(0.081040)}{0.323712}\,\pi_t + \underset{(0.113070)}{0.737777}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.035429)}{0.285662}\,\omega_{p,t} - \underset{(0.310214)}{0.800359}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.377350)}{2.627976},\qquad \bar{n} = \underset{(0.143646)}{0.281061},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.035429)}{0.285662})^2\,\underset{(0.377350)}{2.627976} + (\underset{(0.310214)}{0.800359})^2\,\underset{(0.143646)}{0.281061}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056020 | 0.071237 |
| rho_1 | 0.323712 | 0.081040 |
| phi_1 | 0.737777 | 0.113070 |
| shape_p | 2.627976 | 0.377350 |
| shape_n | 0.281061 | 0.143646 |
| sigma_p | 0.285662 | 0.035429 |
| sigma_n | 0.800359 | 0.310214 |

### Rank 14: Seed 37, Draw 37

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394444`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071723)}{0.056013} + \underset{(0.081477)}{0.323721}\,\pi_t + \underset{(0.113873)}{0.737783}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.044166)}{0.285665}\,\omega_{p,t} - \underset{(0.321627)}{0.800157}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.508571)}{2.627873},\qquad \bar{n} = \underset{(0.148903)}{0.281137},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.044166)}{0.285665})^2\,\underset{(0.508571)}{2.627873} + (\underset{(0.321627)}{0.800157})^2\,\underset{(0.148903)}{0.281137}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056013 | 0.071723 |
| rho_1 | 0.323721 | 0.081477 |
| phi_1 | 0.737783 | 0.113873 |
| shape_p | 2.627873 | 0.508571 |
| shape_n | 0.281137 | 0.148903 |
| sigma_p | 0.285665 | 0.044166 |
| sigma_n | 0.800157 | 0.321627 |

### Rank 15: Seed 24, Draw 16

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394489`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071713)}{0.055999} + \underset{(0.081297)}{0.323706}\,\pi_t + \underset{(0.113850)}{0.737792}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.052037)}{0.285654}\,\omega_{p,t} - \underset{(0.328382)}{0.800250}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.629575)}{2.628094},\qquad \bar{n} = \underset{(0.156082)}{0.281137},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.052037)}{0.285654})^2\,\underset{(0.629575)}{2.628094} + (\underset{(0.328382)}{0.800250})^2\,\underset{(0.156082)}{0.281137}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.055999 | 0.071713 |
| rho_1 | 0.323706 | 0.081297 |
| phi_1 | 0.737792 | 0.113850 |
| shape_p | 2.628094 | 0.629575 |
| shape_n | 0.281137 | 0.156082 |
| sigma_p | 0.285654 | 0.052037 |
| sigma_n | 0.800250 | 0.328382 |

### Rank 16: Seed 31, Draw 33

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394484`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071867)}{0.056007} + \underset{(0.081189)}{0.323716}\,\pi_t + \underset{(0.113193)}{0.737788}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.035583)}{0.285674}\,\omega_{p,t} - \underset{(0.322743)}{0.800199}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.379523)}{2.627827},\qquad \bar{n} = \underset{(0.149071)}{0.281152},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.035583)}{0.285674})^2\,\underset{(0.379523)}{2.627827} + (\underset{(0.322743)}{0.800199})^2\,\underset{(0.149071)}{0.281152}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056007 | 0.071867 |
| rho_1 | 0.323716 | 0.081189 |
| phi_1 | 0.737788 | 0.113193 |
| shape_p | 2.627827 | 0.379523 |
| shape_n | 0.281152 | 0.149071 |
| sigma_p | 0.285674 | 0.035583 |
| sigma_n | 0.800199 | 0.322743 |

### Rank 17: Seed 18, Draw 16

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394485`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071761)}{0.056006} + \underset{(0.080787)}{0.323741}\,\pi_t + \underset{(0.112984)}{0.737747}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039730)}{0.285679}\,\omega_{p,t} - \underset{(0.327968)}{0.800229}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.451101)}{2.627728},\qquad \bar{n} = \underset{(0.154361)}{0.281136},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.039730)}{0.285679})^2\,\underset{(0.451101)}{2.627728} + (\underset{(0.327968)}{0.800229})^2\,\underset{(0.154361)}{0.281136}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056006 | 0.071761 |
| rho_1 | 0.323741 | 0.080787 |
| phi_1 | 0.737747 | 0.112984 |
| shape_p | 2.627728 | 0.451101 |
| shape_n | 0.281136 | 0.154361 |
| sigma_p | 0.285679 | 0.039730 |
| sigma_n | 0.800229 | 0.327968 |

### Rank 18: Seed 1, Draw 36

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394448`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071522)}{0.056014} + \underset{(0.080886)}{0.323692}\,\pi_t + \underset{(0.112733)}{0.737799}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.030020)}{0.285651}\,\omega_{p,t} - \underset{(0.319557)}{0.800248}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.288885)}{2.627994},\qquad \bar{n} = \underset{(0.146421)}{0.281097},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.030020)}{0.285651})^2\,\underset{(0.288885)}{2.627994} + (\underset{(0.319557)}{0.800248})^2\,\underset{(0.146421)}{0.281097}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056014 | 0.071522 |
| rho_1 | 0.323692 | 0.080886 |
| phi_1 | 0.737799 | 0.112733 |
| shape_p | 2.627994 | 0.288885 |
| shape_n | 0.281097 | 0.146421 |
| sigma_p | 0.285651 | 0.030020 |
| sigma_n | 0.800248 | 0.319557 |

### Rank 19: Seed 9, Draw 38

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394495`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071644)}{0.055989} + \underset{(0.080963)}{0.323711}\,\pi_t + \underset{(0.113174)}{0.737787}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039518)}{0.285657}\,\omega_{p,t} - \underset{(0.321672)}{0.800309}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.440866)}{2.627945},\qquad \bar{n} = \underset{(0.149725)}{0.281118},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.039518)}{0.285657})^2\,\underset{(0.440866)}{2.627945} + (\underset{(0.321672)}{0.800309})^2\,\underset{(0.149725)}{0.281118}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.055989 | 0.071644 |
| rho_1 | 0.323711 | 0.080963 |
| phi_1 | 0.737787 | 0.113174 |
| shape_p | 2.627945 | 0.440866 |
| shape_n | 0.281118 | 0.149725 |
| sigma_p | 0.285657 | 0.039518 |
| sigma_n | 0.800309 | 0.321672 |

### Rank 20: Seed 23, Draw 8

- LogLik: `-184.396177`; AIC: `382.792353`; BIC: `406.386820`
- Implied variance: `0.394466`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.071777)}{0.056043} + \underset{(0.081281)}{0.323712}\,\pi_t + \underset{(0.113609)}{0.737765}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.048246)}{0.285673}\,\omega_{p,t} - \underset{(0.332495)}{0.800162}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.575068)}{2.627915},\qquad \bar{n} = \underset{(0.158250)}{0.281144},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.048246)}{0.285673})^2\,\underset{(0.575068)}{2.627915} + (\underset{(0.332495)}{0.800162})^2\,\underset{(0.158250)}{0.281144}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.056043 | 0.071777 |
| rho_1 | 0.323712 | 0.081281 |
| phi_1 | 0.737765 | 0.113609 |
| shape_p | 2.627915 | 0.575068 |
| shape_n | 0.281144 | 0.158250 |
| sigma_p | 0.285673 | 0.048246 |
| sigma_n | 0.800162 | 0.332495 |

## ARX(2,1)

Top 20 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 27 | 15 | -181.884824 | 379.769649 | 406.734753 | 0.398211 | no | `computed` |
| 2 | 18 | 26 | -181.884824 | 379.769649 | 406.734753 | 0.398207 | no | `computed` |
| 3 | 26 | 6 | -181.884824 | 379.769649 | 406.734753 | 0.398205 | no | `computed` |
| 4 | 26 | 16 | -181.884824 | 379.769649 | 406.734753 | 0.398202 | no | `computed` |
| 5 | 31 | 34 | -181.884824 | 379.769649 | 406.734753 | 0.398207 | no | `computed` |
| 6 | 22 | 28 | -181.884824 | 379.769649 | 406.734753 | 0.398200 | no | `computed` |
| 7 | 15 | 12 | -181.884824 | 379.769649 | 406.734753 | 0.398206 | no | `computed` |
| 8 | 49 | 2 | -181.884824 | 379.769649 | 406.734753 | 0.398211 | no | `computed` |
| 9 | 47 | 38 | -181.884824 | 379.769649 | 406.734753 | 0.398209 | no | `computed` |
| 10 | 6 | 29 | -181.884824 | 379.769649 | 406.734753 | 0.398213 | no | `computed` |
| 11 | 9 | 22 | -181.884824 | 379.769649 | 406.734753 | 0.398218 | no | `computed` |
| 12 | 14 | 26 | -181.884824 | 379.769649 | 406.734753 | 0.398194 | no | `computed` |
| 13 | 21 | 20 | -181.884824 | 379.769649 | 406.734753 | 0.398186 | no | `computed` |
| 14 | 29 | 31 | -181.884824 | 379.769649 | 406.734753 | 0.398213 | no | `computed` |
| 15 | 24 | 1 | -181.884824 | 379.769649 | 406.734753 | 0.398214 | no | `computed` |
| 16 | 23 | 10 | -181.884824 | 379.769649 | 406.734753 | 0.398194 | no | `computed` |
| 17 | 49 | 37 | -181.884824 | 379.769649 | 406.734753 | 0.398225 | no | `computed` |
| 18 | 31 | 28 | -181.884825 | 379.769649 | 406.734753 | 0.398198 | no | `computed` |
| 19 | 38 | 18 | -181.884825 | 379.769649 | 406.734753 | 0.398191 | no | `computed` |
| 20 | 35 | 3 | -181.884825 | 379.769649 | 406.734753 | 0.398187 | no | `computed` |

### Rank 1: Seed 27, Draw 15

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398211`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.096946)}{0.039240} + \underset{(0.078018)}{0.316830}\,\pi_t + \underset{(0.098281)}{0.189143}\,\pi_{t-1} + \underset{(0.162078)}{0.539230}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.125710)}{0.246533}\,\omega_{p,t} - \underset{(0.420712)}{0.928387}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(2.613270)}{3.317099},\qquad \bar{n} = \underset{(0.142525)}{0.228103},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.125710)}{0.246533})^2\,\underset{(2.613270)}{3.317099} + (\underset{(0.420712)}{0.928387})^2\,\underset{(0.142525)}{0.228103}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039240 | 0.096946 |
| rho_1 | 0.316830 | 0.078018 |
| rho_2 | 0.189143 | 0.098281 |
| phi_1 | 0.539230 | 0.162078 |
| shape_p | 3.317099 | 2.613270 |
| shape_n | 0.228103 | 0.142525 |
| sigma_p | 0.246533 | 0.125710 |
| sigma_n | 0.928387 | 0.420712 |

### Rank 2: Seed 18, Draw 26

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398207`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090057)}{0.039251} + \underset{(0.076115)}{0.316833}\,\pi_t + \underset{(0.098466)}{0.189140}\,\pi_{t-1} + \underset{(0.147602)}{0.539222}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.037036)}{0.246534}\,\omega_{p,t} - \underset{(0.421291)}{0.928414}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.634640)}{3.317080},\qquad \bar{n} = \underset{(0.142043)}{0.228085},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.037036)}{0.246534})^2\,\underset{(0.634640)}{3.317080} + (\underset{(0.421291)}{0.928414})^2\,\underset{(0.142043)}{0.228085}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039251 | 0.090057 |
| rho_1 | 0.316833 | 0.076115 |
| rho_2 | 0.189140 | 0.098466 |
| phi_1 | 0.539222 | 0.147602 |
| shape_p | 3.317080 | 0.634640 |
| shape_n | 0.228085 | 0.142043 |
| sigma_p | 0.246534 | 0.037036 |
| sigma_n | 0.928414 | 0.421291 |

### Rank 3: Seed 26, Draw 6

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398205`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.091690)}{0.039235} + \underset{(0.076619)}{0.316822}\,\pi_t + \underset{(0.098304)}{0.189147}\,\pi_{t-1} + \underset{(0.151729)}{0.539234}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.076088)}{0.246529}\,\omega_{p,t} - \underset{(0.417922)}{0.928451}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.517885)}{3.317117},\qquad \bar{n} = \underset{(0.142013)}{0.228070},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.076088)}{0.246529})^2\,\underset{(1.517885)}{3.317117} + (\underset{(0.417922)}{0.928451})^2\,\underset{(0.142013)}{0.228070}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039235 | 0.091690 |
| rho_1 | 0.316822 | 0.076619 |
| rho_2 | 0.189147 | 0.098304 |
| phi_1 | 0.539234 | 0.151729 |
| shape_p | 3.317117 | 1.517885 |
| shape_n | 0.228070 | 0.142013 |
| sigma_p | 0.246529 | 0.076088 |
| sigma_n | 0.928451 | 0.417922 |

### Rank 4: Seed 26, Draw 16

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398202`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089536)}{0.039250} + \underset{(0.076063)}{0.316807}\,\pi_t + \underset{(0.098462)}{0.189146}\,\pi_{t-1} + \underset{(0.147215)}{0.539245}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.031097)}{0.246526}\,\omega_{p,t} - \underset{(0.416092)}{0.928377}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.490850)}{3.317293},\qquad \bar{n} = \underset{(0.140349)}{0.228098},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.031097)}{0.246526})^2\,\underset{(0.490850)}{3.317293} + (\underset{(0.416092)}{0.928377})^2\,\underset{(0.140349)}{0.228098}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039250 | 0.089536 |
| rho_1 | 0.316807 | 0.076063 |
| rho_2 | 0.189146 | 0.098462 |
| phi_1 | 0.539245 | 0.147215 |
| shape_p | 3.317293 | 0.490850 |
| shape_n | 0.228098 | 0.140349 |
| sigma_p | 0.246526 | 0.031097 |
| sigma_n | 0.928377 | 0.416092 |

### Rank 5: Seed 31, Draw 34

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398207`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090486)}{0.039227} + \underset{(0.076121)}{0.316823}\,\pi_t + \underset{(0.098485)}{0.189154}\,\pi_{t-1} + \underset{(0.147985)}{0.539235}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.040642)}{0.246524}\,\omega_{p,t} - \underset{(0.420224)}{0.928396}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.721190)}{3.317327},\qquad \bar{n} = \underset{(0.141604)}{0.228096},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.040642)}{0.246524})^2\,\underset{(0.721190)}{3.317327} + (\underset{(0.420224)}{0.928396})^2\,\underset{(0.141604)}{0.228096}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039227 | 0.090486 |
| rho_1 | 0.316823 | 0.076121 |
| rho_2 | 0.189154 | 0.098485 |
| phi_1 | 0.539235 | 0.147985 |
| shape_p | 3.317327 | 0.721190 |
| shape_n | 0.228096 | 0.141604 |
| sigma_p | 0.246524 | 0.040642 |
| sigma_n | 0.928396 | 0.420224 |

### Rank 6: Seed 22, Draw 28

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398200`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089944)}{0.039234} + \underset{(0.076140)}{0.316818}\,\pi_t + \underset{(0.098297)}{0.189145}\,\pi_{t-1} + \underset{(0.148014)}{0.539251}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.041654)}{0.246544}\,\omega_{p,t} - \underset{(0.416000)}{0.928390}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.743787)}{3.316845},\qquad \bar{n} = \underset{(0.140867)}{0.228085},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.041654)}{0.246544})^2\,\underset{(0.743787)}{3.316845} + (\underset{(0.416000)}{0.928390})^2\,\underset{(0.140867)}{0.228085}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039234 | 0.089944 |
| rho_1 | 0.316818 | 0.076140 |
| rho_2 | 0.189145 | 0.098297 |
| phi_1 | 0.539251 | 0.148014 |
| shape_p | 3.316845 | 0.743787 |
| shape_n | 0.228085 | 0.140867 |
| sigma_p | 0.246544 | 0.041654 |
| sigma_n | 0.928390 | 0.416000 |

### Rank 7: Seed 15, Draw 12

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398206`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090256)}{0.039220} + \underset{(0.076158)}{0.316817}\,\pi_t + \underset{(0.098301)}{0.189150}\,\pi_{t-1} + \underset{(0.148778)}{0.539253}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.050476)}{0.246533}\,\omega_{p,t} - \underset{(0.419993)}{0.928370}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.940267)}{3.317065},\qquad \bar{n} = \underset{(0.143007)}{0.228108},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.050476)}{0.246533})^2\,\underset{(0.940267)}{3.317065} + (\underset{(0.419993)}{0.928370})^2\,\underset{(0.143007)}{0.228108}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039220 | 0.090256 |
| rho_1 | 0.316817 | 0.076158 |
| rho_2 | 0.189150 | 0.098301 |
| phi_1 | 0.539253 | 0.148778 |
| shape_p | 3.317065 | 0.940267 |
| shape_n | 0.228108 | 0.143007 |
| sigma_p | 0.246533 | 0.050476 |
| sigma_n | 0.928370 | 0.419993 |

### Rank 8: Seed 49, Draw 2

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398211`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090048)}{0.039233} + \underset{(0.076179)}{0.316838}\,\pi_t + \underset{(0.098263)}{0.189139}\,\pi_{t-1} + \underset{(0.148449)}{0.539232}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.048789)}{0.246542}\,\omega_{p,t} - \underset{(0.420989)}{0.928406}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.900869)}{3.316879},\qquad \bar{n} = \underset{(0.143385)}{0.228092},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.048789)}{0.246542})^2\,\underset{(0.900869)}{3.316879} + (\underset{(0.420989)}{0.928406})^2\,\underset{(0.143385)}{0.228092}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039233 | 0.090048 |
| rho_1 | 0.316838 | 0.076179 |
| rho_2 | 0.189139 | 0.098263 |
| phi_1 | 0.539232 | 0.148449 |
| shape_p | 3.316879 | 0.900869 |
| shape_n | 0.228092 | 0.143385 |
| sigma_p | 0.246542 | 0.048789 |
| sigma_n | 0.928406 | 0.420989 |

### Rank 9: Seed 47, Draw 38

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398209`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089629)}{0.039248} + \underset{(0.075912)}{0.316828}\,\pi_t + \underset{(0.098331)}{0.189132}\,\pi_{t-1} + \underset{(0.146728)}{0.539240}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.029822)}{0.246538}\,\omega_{p,t} - \underset{(0.416507)}{0.928462}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.459685)}{3.317002},\qquad \bar{n} = \underset{(0.140601)}{0.228062},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.029822)}{0.246538})^2\,\underset{(0.459685)}{3.317002} + (\underset{(0.416507)}{0.928462})^2\,\underset{(0.140601)}{0.228062}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039248 | 0.089629 |
| rho_1 | 0.316828 | 0.075912 |
| rho_2 | 0.189132 | 0.098331 |
| phi_1 | 0.539240 | 0.146728 |
| shape_p | 3.317002 | 0.459685 |
| shape_n | 0.228062 | 0.140601 |
| sigma_p | 0.246538 | 0.029822 |
| sigma_n | 0.928462 | 0.416507 |

### Rank 10: Seed 6, Draw 29

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398213`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090895)}{0.039260} + \underset{(0.076158)}{0.316820}\,\pi_t + \underset{(0.098231)}{0.189151}\,\pi_{t-1} + \underset{(0.148207)}{0.539216}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042953)}{0.246532}\,\omega_{p,t} - \underset{(0.418743)}{0.928453}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.778082)}{3.317205},\qquad \bar{n} = \underset{(0.140903)}{0.228068},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.042953)}{0.246532})^2\,\underset{(0.778082)}{3.317205} + (\underset{(0.418743)}{0.928453})^2\,\underset{(0.140903)}{0.228068}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039260 | 0.090895 |
| rho_1 | 0.316820 | 0.076158 |
| rho_2 | 0.189151 | 0.098231 |
| phi_1 | 0.539216 | 0.148207 |
| shape_p | 3.317205 | 0.778082 |
| shape_n | 0.228068 | 0.140903 |
| sigma_p | 0.246532 | 0.042953 |
| sigma_n | 0.928453 | 0.418743 |

### Rank 11: Seed 9, Draw 22

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398218`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089989)}{0.039221} + \underset{(0.076112)}{0.316815}\,\pi_t + \underset{(0.098045)}{0.189149}\,\pi_{t-1} + \underset{(0.147696)}{0.539262}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039186)}{0.246526}\,\omega_{p,t} - \underset{(0.412467)}{0.928446}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.684771)}{3.317262},\qquad \bar{n} = \underset{(0.139615)}{0.228084},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.039186)}{0.246526})^2\,\underset{(0.684771)}{3.317262} + (\underset{(0.412467)}{0.928446})^2\,\underset{(0.139615)}{0.228084}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039221 | 0.089989 |
| rho_1 | 0.316815 | 0.076112 |
| rho_2 | 0.189149 | 0.098045 |
| phi_1 | 0.539262 | 0.147696 |
| shape_p | 3.317262 | 0.684771 |
| shape_n | 0.228084 | 0.139615 |
| sigma_p | 0.246526 | 0.039186 |
| sigma_n | 0.928446 | 0.412467 |

### Rank 12: Seed 14, Draw 26

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398194`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090181)}{0.039259} + \underset{(0.076204)}{0.316820}\,\pi_t + \underset{(0.098445)}{0.189151}\,\pi_{t-1} + \underset{(0.148660)}{0.539219}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043535)}{0.246542}\,\omega_{p,t} - \underset{(0.422113)}{0.928277}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.785524)}{3.316930},\qquad \bar{n} = \underset{(0.142890)}{0.228133},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.043535)}{0.246542})^2\,\underset{(0.785524)}{3.316930} + (\underset{(0.422113)}{0.928277})^2\,\underset{(0.142890)}{0.228133}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039259 | 0.090181 |
| rho_1 | 0.316820 | 0.076204 |
| rho_2 | 0.189151 | 0.098445 |
| phi_1 | 0.539219 | 0.148660 |
| shape_p | 3.316930 | 0.785524 |
| shape_n | 0.228133 | 0.142890 |
| sigma_p | 0.246542 | 0.043535 |
| sigma_n | 0.928277 | 0.422113 |

### Rank 13: Seed 21, Draw 20

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398186`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090051)}{0.039260} + \underset{(0.076114)}{0.316830}\,\pi_t + \underset{(0.098408)}{0.189153}\,\pi_{t-1} + \underset{(0.147644)}{0.539203}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039187)}{0.246533}\,\omega_{p,t} - \underset{(0.423070)}{0.928306}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.686080)}{3.317050},\qquad \bar{n} = \underset{(0.142833)}{0.228118},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.039187)}{0.246533})^2\,\underset{(0.686080)}{3.317050} + (\underset{(0.423070)}{0.928306})^2\,\underset{(0.142833)}{0.228118}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039260 | 0.090051 |
| rho_1 | 0.316830 | 0.076114 |
| rho_2 | 0.189153 | 0.098408 |
| phi_1 | 0.539203 | 0.147644 |
| shape_p | 3.317050 | 0.686080 |
| shape_n | 0.228118 | 0.142833 |
| sigma_p | 0.246533 | 0.039187 |
| sigma_n | 0.928306 | 0.423070 |

### Rank 14: Seed 29, Draw 31

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398213`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089685)}{0.039243} + \underset{(0.076047)}{0.316803}\,\pi_t + \underset{(0.098493)}{0.189150}\,\pi_{t-1} + \underset{(0.147669)}{0.539247}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.039941)}{0.246546}\,\omega_{p,t} - \underset{(0.417047)}{0.928401}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.701158)}{3.316871},\qquad \bar{n} = \underset{(0.141516)}{0.228089},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.039941)}{0.246546})^2\,\underset{(0.701158)}{3.316871} + (\underset{(0.417047)}{0.928401})^2\,\underset{(0.141516)}{0.228089}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039243 | 0.089685 |
| rho_1 | 0.316803 | 0.076047 |
| rho_2 | 0.189150 | 0.098493 |
| phi_1 | 0.539247 | 0.147669 |
| shape_p | 3.316871 | 0.701158 |
| shape_n | 0.228089 | 0.141516 |
| sigma_p | 0.246546 | 0.039941 |
| sigma_n | 0.928401 | 0.417047 |

### Rank 15: Seed 24, Draw 1

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398214`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089768)}{0.039245} + \underset{(0.076153)}{0.316838}\,\pi_t + \underset{(0.098400)}{0.189138}\,\pi_{t-1} + \underset{(0.148129)}{0.539227}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043735)}{0.246525}\,\omega_{p,t} - \underset{(0.422569)}{0.928432}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.787550)}{3.317370},\qquad \bar{n} = \underset{(0.143387)}{0.228081},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.043735)}{0.246525})^2\,\underset{(0.787550)}{3.317370} + (\underset{(0.422569)}{0.928432})^2\,\underset{(0.143387)}{0.228081}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039245 | 0.089768 |
| rho_1 | 0.316838 | 0.076153 |
| rho_2 | 0.189138 | 0.098400 |
| phi_1 | 0.539227 | 0.148129 |
| shape_p | 3.317370 | 0.787550 |
| shape_n | 0.228081 | 0.143387 |
| sigma_p | 0.246525 | 0.043735 |
| sigma_n | 0.928432 | 0.422569 |

### Rank 16: Seed 23, Draw 10

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398194`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089781)}{0.039264} + \underset{(0.076182)}{0.316804}\,\pi_t + \underset{(0.098184)}{0.189136}\,\pi_{t-1} + \underset{(0.147983)}{0.539256}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043905)}{0.246535}\,\omega_{p,t} - \underset{(0.417780)}{0.928316}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.794329)}{3.317140},\qquad \bar{n} = \underset{(0.141704)}{0.228113},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.043905)}{0.246535})^2\,\underset{(0.794329)}{3.317140} + (\underset{(0.417780)}{0.928316})^2\,\underset{(0.141704)}{0.228113}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039264 | 0.089781 |
| rho_1 | 0.316804 | 0.076182 |
| rho_2 | 0.189136 | 0.098184 |
| phi_1 | 0.539256 | 0.147983 |
| shape_p | 3.317140 | 0.794329 |
| shape_n | 0.228113 | 0.141704 |
| sigma_p | 0.246535 | 0.043905 |
| sigma_n | 0.928316 | 0.417780 |

### Rank 17: Seed 49, Draw 37

- LogLik: `-181.884824`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398225`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089328)}{0.039233} + \underset{(0.075873)}{0.316820}\,\pi_t + \underset{(0.098266)}{0.189154}\,\pi_{t-1} + \underset{(0.146854)}{0.539229}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.030346)}{0.246549}\,\omega_{p,t} - \underset{(0.411540)}{0.928448}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.472102)}{3.316859},\qquad \bar{n} = \underset{(0.139193)}{0.228077},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.030346)}{0.246549})^2\,\underset{(0.472102)}{3.316859} + (\underset{(0.411540)}{0.928448})^2\,\underset{(0.139193)}{0.228077}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039233 | 0.089328 |
| rho_1 | 0.316820 | 0.075873 |
| rho_2 | 0.189154 | 0.098266 |
| phi_1 | 0.539229 | 0.146854 |
| shape_p | 3.316859 | 0.472102 |
| shape_n | 0.228077 | 0.139193 |
| sigma_p | 0.246549 | 0.030346 |
| sigma_n | 0.928448 | 0.411540 |

### Rank 18: Seed 31, Draw 28

- LogLik: `-181.884825`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398198`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.089864)}{0.039225} + \underset{(0.076139)}{0.316829}\,\pi_t + \underset{(0.098307)}{0.189129}\,\pi_{t-1} + \underset{(0.147657)}{0.539257}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.037841)}{0.246537}\,\omega_{p,t} - \underset{(0.416216)}{0.928321}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.654879)}{3.316937},\qquad \bar{n} = \underset{(0.140639)}{0.228125},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.037841)}{0.246537})^2\,\underset{(0.654879)}{3.316937} + (\underset{(0.416216)}{0.928321})^2\,\underset{(0.140639)}{0.228125}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039225 | 0.089864 |
| rho_1 | 0.316829 | 0.076139 |
| rho_2 | 0.189129 | 0.098307 |
| phi_1 | 0.539257 | 0.147657 |
| shape_p | 3.316937 | 0.654879 |
| shape_n | 0.228125 | 0.140639 |
| sigma_p | 0.246537 | 0.037841 |
| sigma_n | 0.928321 | 0.416216 |

### Rank 19: Seed 38, Draw 18

- LogLik: `-181.884825`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398191`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.090273)}{0.039242} + \underset{(0.076113)}{0.316829}\,\pi_t + \underset{(0.098491)}{0.189153}\,\pi_{t-1} + \underset{(0.147804)}{0.539207}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.044426)}{0.246523}\,\omega_{p,t} - \underset{(0.424282)}{0.928330}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.805910)}{3.317293},\qquad \bar{n} = \underset{(0.143278)}{0.228113},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.044426)}{0.246523})^2\,\underset{(0.805910)}{3.317293} + (\underset{(0.424282)}{0.928330})^2\,\underset{(0.143278)}{0.228113}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039242 | 0.090273 |
| rho_1 | 0.316829 | 0.076113 |
| rho_2 | 0.189153 | 0.098491 |
| phi_1 | 0.539207 | 0.147804 |
| shape_p | 3.317293 | 0.805910 |
| shape_n | 0.228113 | 0.143278 |
| sigma_p | 0.246523 | 0.044426 |
| sigma_n | 0.928330 | 0.424282 |

### Rank 20: Seed 35, Draw 3

- LogLik: `-181.884825`; AIC: `379.769649`; BIC: `406.734753`
- Implied variance: `0.398187`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.091462)}{0.039263} + \underset{(0.076359)}{0.316818}\,\pi_t + \underset{(0.098279)}{0.189143}\,\pi_{t-1} + \underset{(0.150738)}{0.539215}\,SPF_t + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.064714)}{0.246539}\,\omega_{p,t} - \underset{(0.415743)}{0.928344}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.265844)}{3.316919},\qquad \bar{n} = \underset{(0.140942)}{0.228098},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.064714)}{0.246539})^2\,\underset{(1.265844)}{3.316919} + (\underset{(0.415743)}{0.928344})^2\,\underset{(0.140942)}{0.228098}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.039263 | 0.091462 |
| rho_1 | 0.316818 | 0.076359 |
| rho_2 | 0.189143 | 0.098279 |
| phi_1 | 0.539215 | 0.150738 |
| shape_p | 3.316919 | 1.265844 |
| shape_n | 0.228098 | 0.140942 |
| sigma_p | 0.246539 | 0.064714 |
| sigma_n | 0.928344 | 0.415743 |

## ARX(2,2)

Top 20 admissible estimates ranked by log likelihood.

| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var | Above -150 Diagnostic | SE Status |
|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 18 | 1 | -181.688415 | 381.376830 | 411.712572 | 0.395222 | no | `computed` |
| 2 | 1 | 20 | -181.688415 | 381.376830 | 411.712572 | 0.395120 | no | `computed` |
| 3 | 13 | 30 | -181.688415 | 381.376830 | 411.712572 | 0.395224 | no | `computed` |
| 4 | 31 | 28 | -181.688415 | 381.376831 | 411.712573 | 0.395088 | no | `computed` |
| 5 | 10 | 38 | -181.688416 | 381.376832 | 411.712575 | 0.395182 | no | `computed` |
| 6 | 50 | 17 | -181.688416 | 381.376832 | 411.712575 | 0.395122 | no | `computed` |
| 7 | 33 | 25 | -181.688416 | 381.376833 | 411.712575 | 0.395223 | no | `computed` |
| 8 | 23 | 24 | -181.688417 | 381.376833 | 411.712575 | 0.395156 | no | `computed` |
| 9 | 27 | 11 | -181.688417 | 381.376833 | 411.712576 | 0.395147 | no | `computed` |
| 10 | 9 | 19 | -181.688417 | 381.376833 | 411.712576 | 0.395313 | no | `computed` |
| 11 | 34 | 2 | -181.688417 | 381.376834 | 411.712576 | 0.395197 | no | `computed` |
| 12 | 47 | 9 | -181.688417 | 381.376835 | 411.712577 | 0.395146 | no | `computed` |
| 13 | 29 | 30 | -181.688418 | 381.376835 | 411.712578 | 0.395228 | no | `computed` |
| 14 | 17 | 8 | -181.688418 | 381.376835 | 411.712578 | 0.395226 | no | `computed` |
| 15 | 4 | 28 | -181.688418 | 381.376835 | 411.712578 | 0.395217 | no | `computed` |
| 16 | 13 | 31 | -181.688418 | 381.376835 | 411.712578 | 0.395087 | no | `computed` |
| 17 | 28 | 3 | -181.688418 | 381.376835 | 411.712578 | 0.395154 | no | `computed` |
| 18 | 31 | 25 | -181.688418 | 381.376835 | 411.712578 | 0.395147 | no | `computed` |
| 19 | 48 | 33 | -181.688418 | 381.376835 | 411.712578 | 0.395051 | no | `computed` |
| 20 | 46 | 1 | -181.688418 | 381.376836 | 411.712578 | 0.395257 | no | `computed` |

### Rank 1: Seed 18, Draw 1

- LogLik: `-181.688415`; AIC: `381.376830`; BIC: `411.712572`
- Implied variance: `0.395222`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.107935)}{0.004171} + \underset{(0.119587)}{0.323173}\,\pi_t + \underset{(0.083401)}{0.163315}\,\pi_{t-1} + \underset{(0.306968)}{0.410717}\,SPF_t + \underset{(0.293473)}{0.194172}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.033788)}{0.271204}\,\omega_{p,t} - \underset{(0.342804)}{0.840368}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.409081)}{2.664771},\qquad \bar{n} = \underset{(0.158553)}{0.282099},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.033788)}{0.271204})^2\,\underset{(0.409081)}{2.664771} + (\underset{(0.342804)}{0.840368})^2\,\underset{(0.158553)}{0.282099}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004171 | 0.107935 |
| rho_1 | 0.323173 | 0.119587 |
| rho_2 | 0.163315 | 0.083401 |
| phi_1 | 0.410717 | 0.306968 |
| phi_2 | 0.194172 | 0.293473 |
| shape_p | 2.664771 | 0.409081 |
| shape_n | 0.282099 | 0.158553 |
| sigma_p | 0.271204 | 0.033788 |
| sigma_n | 0.840368 | 0.342804 |

### Rank 2: Seed 1, Draw 20

- LogLik: `-181.688415`; AIC: `381.376830`; BIC: `411.712572`
- Implied variance: `0.395120`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.107490)}{0.004263} + \underset{(0.121097)}{0.323278}\,\pi_t + \underset{(0.081796)}{0.163371}\,\pi_{t-1} + \underset{(0.330511)}{0.410449}\,SPF_t + \underset{(0.314779)}{0.194160}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.024792)}{0.271157}\,\omega_{p,t} - \underset{(0.337429)}{0.839817}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.301750)}{2.665408},\qquad \bar{n} = \underset{(0.152777)}{0.282355},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.024792)}{0.271157})^2\,\underset{(0.301750)}{2.665408} + (\underset{(0.337429)}{0.839817})^2\,\underset{(0.152777)}{0.282355}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004263 | 0.107490 |
| rho_1 | 0.323278 | 0.121097 |
| rho_2 | 0.163371 | 0.081796 |
| phi_1 | 0.410449 | 0.330511 |
| phi_2 | 0.194160 | 0.314779 |
| shape_p | 2.665408 | 0.301750 |
| shape_n | 0.282355 | 0.152777 |
| sigma_p | 0.271157 | 0.024792 |
| sigma_n | 0.839817 | 0.337429 |

### Rank 3: Seed 13, Draw 30

- LogLik: `-181.688415`; AIC: `381.376830`; BIC: `411.712572`
- Implied variance: `0.395224`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.110267)}{0.004179} + \underset{(0.119669)}{0.323231}\,\pi_t + \underset{(0.081376)}{0.163357}\,\pi_{t-1} + \underset{(0.293176)}{0.410464}\,SPF_t + \underset{(0.278948)}{0.194290}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.035012)}{0.271228}\,\omega_{p,t} - \underset{(0.373150)}{0.840331}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.507414)}{2.664400},\qquad \bar{n} = \underset{(0.170847)}{0.282117},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.035012)}{0.271228})^2\,\underset{(0.507414)}{2.664400} + (\underset{(0.373150)}{0.840331})^2\,\underset{(0.170847)}{0.282117}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004179 | 0.110267 |
| rho_1 | 0.323231 | 0.119669 |
| rho_2 | 0.163357 | 0.081376 |
| phi_1 | 0.410464 | 0.293176 |
| phi_2 | 0.194290 | 0.278948 |
| shape_p | 2.664400 | 0.507414 |
| shape_n | 0.282117 | 0.170847 |
| sigma_p | 0.271228 | 0.035012 |
| sigma_n | 0.840331 | 0.373150 |

### Rank 4: Seed 31, Draw 28

- LogLik: `-181.688415`; AIC: `381.376831`; BIC: `411.712573`
- Implied variance: `0.395088`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.177087)}{0.004208} + \underset{(0.121627)}{0.323297}\,\pi_t + \underset{(0.108819)}{0.163370}\,\pi_{t-1} + \underset{(0.308428)}{0.410312}\,SPF_t + \underset{(0.367488)}{0.194314}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.324586)}{0.271150}\,\omega_{p,t} - \underset{(1.093741)}{0.839907}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(5.011105)}{2.665276},\qquad \bar{n} = \underset{(0.557766)}{0.282277},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.324586)}{0.271150})^2\,\underset{(5.011105)}{2.665276} + (\underset{(1.093741)}{0.839907})^2\,\underset{(0.557766)}{0.282277}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004208 | 0.177087 |
| rho_1 | 0.323297 | 0.121627 |
| rho_2 | 0.163370 | 0.108819 |
| phi_1 | 0.410312 | 0.308428 |
| phi_2 | 0.194314 | 0.367488 |
| shape_p | 2.665276 | 5.011105 |
| shape_n | 0.282277 | 0.557766 |
| sigma_p | 0.271150 | 0.324586 |
| sigma_n | 0.839907 | 1.093741 |

### Rank 5: Seed 10, Draw 38

- LogLik: `-181.688416`; AIC: `381.376832`; BIC: `411.712575`
- Implied variance: `0.395182`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.112315)}{0.004169} + \underset{(0.120630)}{0.323162}\,\pi_t + \underset{(0.081940)}{0.163312}\,\pi_{t-1} + \underset{(0.321129)}{0.410592}\,SPF_t + \underset{(0.305106)}{0.194342}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.043444)}{0.271107}\,\omega_{p,t} - \underset{(0.357103)}{0.840465}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.655175)}{2.666229},\qquad \bar{n} = \underset{(0.161379)}{0.282024},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.043444)}{0.271107})^2\,\underset{(0.655175)}{2.666229} + (\underset{(0.357103)}{0.840465})^2\,\underset{(0.161379)}{0.282024}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004169 | 0.112315 |
| rho_1 | 0.323162 | 0.120630 |
| rho_2 | 0.163312 | 0.081940 |
| phi_1 | 0.410592 | 0.321129 |
| phi_2 | 0.194342 | 0.305106 |
| shape_p | 2.666229 | 0.655175 |
| shape_n | 0.282024 | 0.161379 |
| sigma_p | 0.271107 | 0.043444 |
| sigma_n | 0.840465 | 0.357103 |

### Rank 6: Seed 50, Draw 17

- LogLik: `-181.688416`; AIC: `381.376832`; BIC: `411.712575`
- Implied variance: `0.395122`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.113091)}{0.004234} + \underset{(0.119023)}{0.323238}\,\pi_t + \underset{(0.081922)}{0.163301}\,\pi_{t-1} + \underset{(0.306264)}{0.410889}\,SPF_t + \underset{(0.301121)}{0.193892}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.049765)}{0.271224}\,\omega_{p,t} - \underset{(0.369930)}{0.839874}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.754820)}{2.664288},\qquad \bar{n} = \underset{(0.165800)}{0.282300},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.049765)}{0.271224})^2\,\underset{(0.754820)}{2.664288} + (\underset{(0.369930)}{0.839874})^2\,\underset{(0.165800)}{0.282300}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004234 | 0.113091 |
| rho_1 | 0.323238 | 0.119023 |
| rho_2 | 0.163301 | 0.081922 |
| phi_1 | 0.410889 | 0.306264 |
| phi_2 | 0.193892 | 0.301121 |
| shape_p | 2.664288 | 0.754820 |
| shape_n | 0.282300 | 0.165800 |
| sigma_p | 0.271224 | 0.049765 |
| sigma_n | 0.839874 | 0.369930 |

### Rank 7: Seed 33, Draw 25

- LogLik: `-181.688416`; AIC: `381.376833`; BIC: `411.712575`
- Implied variance: `0.395223`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.114336)}{0.004029} + \underset{(0.122917)}{0.323245}\,\pi_t + \underset{(0.092810)}{0.163364}\,\pi_{t-1} + \underset{(0.303864)}{0.410335}\,SPF_t + \underset{(0.307448)}{0.194526}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.105724)}{0.271138}\,\omega_{p,t} - \underset{(0.385196)}{0.840256}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.595019)}{2.665527},\qquad \bar{n} = \underset{(0.181162)}{0.282232},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.105724)}{0.271138})^2\,\underset{(1.595019)}{2.665527} + (\underset{(0.385196)}{0.840256})^2\,\underset{(0.181162)}{0.282232}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004029 | 0.114336 |
| rho_1 | 0.323245 | 0.122917 |
| rho_2 | 0.163364 | 0.092810 |
| phi_1 | 0.410335 | 0.303864 |
| phi_2 | 0.194526 | 0.307448 |
| shape_p | 2.665527 | 1.595019 |
| shape_n | 0.282232 | 0.181162 |
| sigma_p | 0.271138 | 0.105724 |
| sigma_n | 0.840256 | 0.385196 |

### Rank 8: Seed 23, Draw 24

- LogLik: `-181.688417`; AIC: `381.376833`; BIC: `411.712575`
- Implied variance: `0.395156`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.108158)}{0.004179} + \underset{(0.116723)}{0.323148}\,\pi_t + \underset{(0.081860)}{0.163402}\,\pi_{t-1} + \underset{(0.302886)}{0.410884}\,SPF_t + \underset{(0.291474)}{0.193899}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.083047)}{0.271156}\,\omega_{p,t} - \underset{(0.337961)}{0.839896}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.295659)}{2.665569},\qquad \bar{n} = \underset{(0.153078)}{0.282337},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.083047)}{0.271156})^2\,\underset{(1.295659)}{2.665569} + (\underset{(0.337961)}{0.839896})^2\,\underset{(0.153078)}{0.282337}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004179 | 0.108158 |
| rho_1 | 0.323148 | 0.116723 |
| rho_2 | 0.163402 | 0.081860 |
| phi_1 | 0.410884 | 0.302886 |
| phi_2 | 0.193899 | 0.291474 |
| shape_p | 2.665569 | 1.295659 |
| shape_n | 0.282337 | 0.153078 |
| sigma_p | 0.271156 | 0.083047 |
| sigma_n | 0.839896 | 0.337961 |

### Rank 9: Seed 27, Draw 11

- LogLik: `-181.688417`; AIC: `381.376833`; BIC: `411.712576`
- Implied variance: `0.395147`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.110518)}{0.004176} + \underset{(0.123555)}{0.323292}\,\pi_t + \underset{(0.084243)}{0.163353}\,\pi_{t-1} + \underset{(0.311633)}{0.410315}\,SPF_t + \underset{(0.300104)}{0.194334}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.049876)}{0.271172}\,\omega_{p,t} - \underset{(0.345890)}{0.839795}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.708790)}{2.665091},\qquad \bar{n} = \underset{(0.157574)}{0.282410},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.049876)}{0.271172})^2\,\underset{(0.708790)}{2.665091} + (\underset{(0.345890)}{0.839795})^2\,\underset{(0.157574)}{0.282410}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004176 | 0.110518 |
| rho_1 | 0.323292 | 0.123555 |
| rho_2 | 0.163353 | 0.084243 |
| phi_1 | 0.410315 | 0.311633 |
| phi_2 | 0.194334 | 0.300104 |
| shape_p | 2.665091 | 0.708790 |
| shape_n | 0.282410 | 0.157574 |
| sigma_p | 0.271172 | 0.049876 |
| sigma_n | 0.839795 | 0.345890 |

### Rank 10: Seed 9, Draw 19

- LogLik: `-181.688417`; AIC: `381.376833`; BIC: `411.712576`
- Implied variance: `0.395313`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.112004)}{0.004083} + \underset{(0.121358)}{0.323249}\,\pi_t + \underset{(0.083422)}{0.163338}\,\pi_{t-1} + \underset{(0.295892)}{0.410514}\,SPF_t + \underset{(0.290283)}{0.194354}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.028269)}{0.271230}\,\omega_{p,t} - \underset{(0.363879)}{0.840691}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.316462)}{2.664458},\qquad \bar{n} = \underset{(0.164754)}{0.281991},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.028269)}{0.271230})^2\,\underset{(0.316462)}{2.664458} + (\underset{(0.363879)}{0.840691})^2\,\underset{(0.164754)}{0.281991}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004083 | 0.112004 |
| rho_1 | 0.323249 | 0.121358 |
| rho_2 | 0.163338 | 0.083422 |
| phi_1 | 0.410514 | 0.295892 |
| phi_2 | 0.194354 | 0.290283 |
| shape_p | 2.664458 | 0.316462 |
| shape_n | 0.281991 | 0.164754 |
| sigma_p | 0.271230 | 0.028269 |
| sigma_n | 0.840691 | 0.363879 |

### Rank 11: Seed 34, Draw 2

- LogLik: `-181.688417`; AIC: `381.376834`; BIC: `411.712576`
- Implied variance: `0.395197`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.107668)}{0.004243} + \underset{(0.120027)}{0.323229}\,\pi_t + \underset{(0.081873)}{0.163328}\,\pi_{t-1} + \underset{(0.314672)}{0.411108}\,SPF_t + \underset{(0.300176)}{0.193640}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.026885)}{0.271155}\,\omega_{p,t} - \underset{(0.317411)}{0.840196}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.335620)}{2.665435},\qquad \bar{n} = \underset{(0.139723)}{0.282211},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.026885)}{0.271155})^2\,\underset{(0.335620)}{2.665435} + (\underset{(0.317411)}{0.840196})^2\,\underset{(0.139723)}{0.282211}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004243 | 0.107668 |
| rho_1 | 0.323229 | 0.120027 |
| rho_2 | 0.163328 | 0.081873 |
| phi_1 | 0.411108 | 0.314672 |
| phi_2 | 0.193640 | 0.300176 |
| shape_p | 2.665435 | 0.335620 |
| shape_n | 0.282211 | 0.139723 |
| sigma_p | 0.271155 | 0.026885 |
| sigma_n | 0.840196 | 0.317411 |

### Rank 12: Seed 47, Draw 9

- LogLik: `-181.688417`; AIC: `381.376835`; BIC: `411.712577`
- Implied variance: `0.395146`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.113119)}{0.004256} + \underset{(0.122783)}{0.323190}\,\pi_t + \underset{(0.086834)}{0.163294}\,\pi_{t-1} + \underset{(0.321653)}{0.411085}\,SPF_t + \underset{(0.312851)}{0.193747}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.067461)}{0.271239}\,\omega_{p,t} - \underset{(0.383882)}{0.840186}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.974118)}{2.664173},\qquad \bar{n} = \underset{(0.176852)}{0.282104},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.067461)}{0.271239})^2\,\underset{(0.974118)}{2.664173} + (\underset{(0.383882)}{0.840186})^2\,\underset{(0.176852)}{0.282104}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004256 | 0.113119 |
| rho_1 | 0.323190 | 0.122783 |
| rho_2 | 0.163294 | 0.086834 |
| phi_1 | 0.411085 | 0.321653 |
| phi_2 | 0.193747 | 0.312851 |
| shape_p | 2.664173 | 0.974118 |
| shape_n | 0.282104 | 0.176852 |
| sigma_p | 0.271239 | 0.067461 |
| sigma_n | 0.840186 | 0.383882 |

### Rank 13: Seed 29, Draw 30

- LogLik: `-181.688418`; AIC: `381.376835`; BIC: `411.712578`
- Implied variance: `0.395228`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.110484)}{0.004160} + \underset{(0.119905)}{0.323201}\,\pi_t + \underset{(0.084523)}{0.163362}\,\pi_{t-1} + \underset{(0.301300)}{0.410844}\,SPF_t + \underset{(0.289074)}{0.193987}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.041928)}{0.271123}\,\omega_{p,t} - \underset{(0.352477)}{0.840718}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.559682)}{2.666049},\qquad \bar{n} = \underset{(0.161816)}{0.281905},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.041928)}{0.271123})^2\,\underset{(0.559682)}{2.666049} + (\underset{(0.352477)}{0.840718})^2\,\underset{(0.161816)}{0.281905}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004160 | 0.110484 |
| rho_1 | 0.323201 | 0.119905 |
| rho_2 | 0.163362 | 0.084523 |
| phi_1 | 0.410844 | 0.301300 |
| phi_2 | 0.193987 | 0.289074 |
| shape_p | 2.666049 | 0.559682 |
| shape_n | 0.281905 | 0.161816 |
| sigma_p | 0.271123 | 0.041928 |
| sigma_n | 0.840718 | 0.352477 |

### Rank 14: Seed 17, Draw 8

- LogLik: `-181.688418`; AIC: `381.376835`; BIC: `411.712578`
- Implied variance: `0.395226`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.111548)}{0.004135} + \underset{(0.120017)}{0.323176}\,\pi_t + \underset{(0.083623)}{0.163282}\,\pi_{t-1} + \underset{(0.296512)}{0.411250}\,SPF_t + \underset{(0.291040)}{0.193704}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.024106)}{0.271149}\,\omega_{p,t} - \underset{(0.356610)}{0.840476}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.221703)}{2.665667},\qquad \bar{n} = \underset{(0.160966)}{0.282052},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.024106)}{0.271149})^2\,\underset{(0.221703)}{2.665667} + (\underset{(0.356610)}{0.840476})^2\,\underset{(0.160966)}{0.282052}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004135 | 0.111548 |
| rho_1 | 0.323176 | 0.120017 |
| rho_2 | 0.163282 | 0.083623 |
| phi_1 | 0.411250 | 0.296512 |
| phi_2 | 0.193704 | 0.291040 |
| shape_p | 2.665667 | 0.221703 |
| shape_n | 0.282052 | 0.160966 |
| sigma_p | 0.271149 | 0.024106 |
| sigma_n | 0.840476 | 0.356610 |

### Rank 15: Seed 4, Draw 28

- LogLik: `-181.688418`; AIC: `381.376835`; BIC: `411.712578`
- Implied variance: `0.395217`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.110882)}{0.004311} + \underset{(0.121594)}{0.323235}\,\pi_t + \underset{(0.082617)}{0.163437}\,\pi_{t-1} + \underset{(0.328049)}{0.410563}\,SPF_t + \underset{(0.303672)}{0.194029}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.030263)}{0.271220}\,\omega_{p,t} - \underset{(0.354510)}{0.840297}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.407675)}{2.664708},\qquad \bar{n} = \underset{(0.158760)}{0.282113},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.030263)}{0.271220})^2\,\underset{(0.407675)}{2.664708} + (\underset{(0.354510)}{0.840297})^2\,\underset{(0.158760)}{0.282113}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004311 | 0.110882 |
| rho_1 | 0.323235 | 0.121594 |
| rho_2 | 0.163437 | 0.082617 |
| phi_1 | 0.410563 | 0.328049 |
| phi_2 | 0.194029 | 0.303672 |
| shape_p | 2.664708 | 0.407675 |
| shape_n | 0.282113 | 0.158760 |
| sigma_p | 0.271220 | 0.030263 |
| sigma_n | 0.840297 | 0.354510 |

### Rank 16: Seed 13, Draw 31

- LogLik: `-181.688418`; AIC: `381.376835`; BIC: `411.712578`
- Implied variance: `0.395087`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.112123)}{0.004295} + \underset{(0.117844)}{0.323174}\,\pi_t + \underset{(0.082038)}{0.163272}\,\pi_{t-1} + \underset{(0.305615)}{0.411014}\,SPF_t + \underset{(0.310129)}{0.193856}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.108539)}{0.271190}\,\omega_{p,t} - \underset{(0.354375)}{0.839798}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(1.711752)}{2.665295},\qquad \bar{n} = \underset{(0.157669)}{0.282265},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.108539)}{0.271190})^2\,\underset{(1.711752)}{2.665295} + (\underset{(0.354375)}{0.839798})^2\,\underset{(0.157669)}{0.282265}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004295 | 0.112123 |
| rho_1 | 0.323174 | 0.117844 |
| rho_2 | 0.163272 | 0.082038 |
| phi_1 | 0.411014 | 0.305615 |
| phi_2 | 0.193856 | 0.310129 |
| shape_p | 2.665295 | 1.711752 |
| shape_n | 0.282265 | 0.157669 |
| sigma_p | 0.271190 | 0.108539 |
| sigma_n | 0.839798 | 0.354375 |

### Rank 17: Seed 28, Draw 3

- LogLik: `-181.688418`; AIC: `381.376835`; BIC: `411.712578`
- Implied variance: `0.395154`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.112623)}{0.004187} + \underset{(0.119388)}{0.323077}\,\pi_t + \underset{(0.082415)}{0.163361}\,\pi_{t-1} + \underset{(0.301485)}{0.410660}\,SPF_t + \underset{(0.294350)}{0.194311}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.020732)}{0.271080}\,\omega_{p,t} - \underset{(0.375916)}{0.840353}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.195506)}{2.666666},\qquad \bar{n} = \underset{(0.173754)}{0.282070},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.020732)}{0.271080})^2\,\underset{(0.195506)}{2.666666} + (\underset{(0.375916)}{0.840353})^2\,\underset{(0.173754)}{0.282070}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004187 | 0.112623 |
| rho_1 | 0.323077 | 0.119388 |
| rho_2 | 0.163361 | 0.082415 |
| phi_1 | 0.410660 | 0.301485 |
| phi_2 | 0.194311 | 0.294350 |
| shape_p | 2.666666 | 0.195506 |
| shape_n | 0.282070 | 0.173754 |
| sigma_p | 0.271080 | 0.020732 |
| sigma_n | 0.840353 | 0.375916 |

### Rank 18: Seed 31, Draw 25

- LogLik: `-181.688418`; AIC: `381.376835`; BIC: `411.712578`
- Implied variance: `0.395147`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.110134)}{0.004262} + \underset{(0.121393)}{0.323219}\,\pi_t + \underset{(0.084648)}{0.163351}\,\pi_{t-1} + \underset{(0.319830)}{0.410419}\,SPF_t + \underset{(0.302532)}{0.194320}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.042960)}{0.271189}\,\omega_{p,t} - \underset{(0.355522)}{0.839632}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.574472)}{2.665186},\qquad \bar{n} = \underset{(0.163603)}{0.282476},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.042960)}{0.271189})^2\,\underset{(0.574472)}{2.665186} + (\underset{(0.355522)}{0.839632})^2\,\underset{(0.163603)}{0.282476}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004262 | 0.110134 |
| rho_1 | 0.323219 | 0.121393 |
| rho_2 | 0.163351 | 0.084648 |
| phi_1 | 0.410419 | 0.319830 |
| phi_2 | 0.194320 | 0.302532 |
| shape_p | 2.665186 | 0.574472 |
| shape_n | 0.282476 | 0.163603 |
| sigma_p | 0.271189 | 0.042960 |
| sigma_n | 0.839632 | 0.355522 |

### Rank 19: Seed 48, Draw 33

- LogLik: `-181.688418`; AIC: `381.376835`; BIC: `411.712578`
- Implied variance: `0.395051`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.113774)}{0.004257} + \underset{(0.124842)}{0.323200}\,\pi_t + \underset{(0.095727)}{0.163300}\,\pi_{t-1} + \underset{(0.350739)}{0.410674}\,SPF_t + \underset{(0.336282)}{0.194153}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.277729)}{0.271118}\,\omega_{p,t} - \underset{(0.466823)}{0.839528}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(4.350205)}{2.666227},\qquad \bar{n} = \underset{(0.228477)}{0.282447},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.277729)}{0.271118})^2\,\underset{(4.350205)}{2.666227} + (\underset{(0.466823)}{0.839528})^2\,\underset{(0.228477)}{0.282447}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004257 | 0.113774 |
| rho_1 | 0.323200 | 0.124842 |
| rho_2 | 0.163300 | 0.095727 |
| phi_1 | 0.410674 | 0.350739 |
| phi_2 | 0.194153 | 0.336282 |
| shape_p | 2.666227 | 4.350205 |
| shape_n | 0.282447 | 0.228477 |
| sigma_p | 0.271118 | 0.277729 |
| sigma_n | 0.839528 | 0.466823 |

### Rank 20: Seed 46, Draw 1

- LogLik: `-181.688418`; AIC: `381.376836`; BIC: `411.712578`
- Implied variance: `0.395257`
- Selection diagnostics: `eligible`
- SE status: `computed`

Mean process:

$$
\pi_{t+1} = \underset{(0.112092)}{0.004188} + \underset{(0.121000)}{0.323155}\,\pi_t + \underset{(0.084125)}{0.163279}\,\pi_{t-1} + \underset{(0.315832)}{0.411079}\,SPF_t + \underset{(0.304727)}{0.193900}\,SPF_{t-1} + u_{t+1}
$$

BEGE volatility process:

$$
\begin{aligned}
u_t &= \underset{(0.023700)}{0.271255}\,\omega_{p,t} - \underset{(0.367650)}{0.840370}\,\omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\
\bar{p} &= \underset{(0.222829)}{2.664118},\qquad \bar{n} = \underset{(0.164983)}{0.282111},\\
\operatorname{Var}_t(u_t) &= (\underset{(0.023700)}{0.271255})^2\,\underset{(0.222829)}{2.664118} + (\underset{(0.367650)}{0.840370})^2\,\underset{(0.164983)}{0.282111}.
\end{aligned}
$$

Parameter table:

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| c | 0.004188 | 0.112092 |
| rho_1 | 0.323155 | 0.121000 |
| rho_2 | 0.163279 | 0.084125 |
| phi_1 | 0.411079 | 0.315832 |
| phi_2 | 0.193900 | 0.304727 |
| shape_p | 2.664118 | 0.222829 |
| shape_n | 0.282111 | 0.164983 |
| sigma_p | 0.271255 | 0.023700 |
| sigma_n | 0.840370 | 0.367650 |
