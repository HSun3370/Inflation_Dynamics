```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# OLS Results (Monthly)

Estimation results of the six mean models on the monthly effective sample (1969M2--2026M9, 692 observations) with HAC(12) standard errors in parentheses below each estimate. (The log-likelihood is computed under the Gaussian assumption.) ARX(1,1) has the lowest AIC and BIC.

**Constant**:

$$
\hat{\pi}_{t+1} = SPF_t
$$

No estimated coefficients. Residuals $u_{t+1} = \pi_{t+1} - SPF_t$.

- Log-likelihood: **-98.986**
- AIC: **199.971**, BIC: **204.511**
- Residual skewness: **-0.419**, excess kurtosis: **6.609**

**ARX(1,0)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{cc}
0.1230 & +\,0.6228\,\pi_t \\
(0.017) & (0.056)
\end{array}
\end{array}
$$

- Log-likelihood: **-33.819**
- AIC: **71.639**, BIC: **80.718**
- Residual skewness: **-0.004**, excess kurtosis: **4.944**

**ARX(1,1)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{ccc}
0.0188 & +\,0.4688\,\pi_t & +\,0.5653\,SPF_t \\
(0.020) & (0.053) & (0.099)
\end{array}
\end{array}
$$

- Log-likelihood: **-2.420**
- AIC: **10.839**, BIC: **24.458**
- Residual skewness: **-0.114**, excess kurtosis: **5.824**

**ARX(2,0)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{ccc}
0.1138 & +\,0.5764\,\pi_t & +\,0.0746\,\pi_{t-1} \\
(0.017) & (0.070) & (0.083)
\end{array}
\end{array}
$$

- Log-likelihood: **-31.891**
- AIC: **69.782**, BIC: **83.400**
- Residual skewness: **-0.046**, excess kurtosis: **4.884**

**ARX(2,1)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{cccc}
0.0188 & +\,0.4834\,\pi_t & -\,0.0334\,\pi_{t-1} & +\,0.5878\,SPF_t \\
(0.020) & (0.074) & (0.076) & (0.093)
\end{array}
\end{array}
$$

- Log-likelihood: **-2.050**
- AIC: **12.100**, BIC: **30.258**
- Residual skewness: **-0.094**, excess kurtosis: **5.880**

**ARX(2,2)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{ccccc}
0.0172 & +\,0.4860\,\pi_t & -\,0.0289\,\pi_{t-1} & +\,0.0436\,SPF_t & +\,0.5414\,SPF_{t-1} \\
(0.021) & (0.075) & (0.078) & (0.415) & (0.389)
\end{array}
\end{array}
$$

- Log-likelihood: **-1.208**
- AIC: **12.415**, BIC: **35.113**
- Residual skewness: **-0.089**, excess kurtosis: **5.852**
