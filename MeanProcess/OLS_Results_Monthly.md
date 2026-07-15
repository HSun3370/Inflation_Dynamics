```{raw:typst}
#set page(margin: auto)
```

# OLS Results (Monthly)

Estimation results of four mean models on the monthly effective sample (1969-02--2026-05, 688 observations) with HAC(12) standard errors in parentheses below each estimate. (The log-likelihood is computed under the Gaussian assumption.)

**Constant**:

$$
\hat{\pi}_{t+1} = SPF_t
$$

No estimated coefficients. Residuals $\mu_{t+1} = \pi_{t+1} - SPF_t$.

- Log-likelihood: **-96.715**
- AIC: **195.430**, BIC: **199.963**
- Residual skewness: **-0.413**, excess kurtosis: **6.728**

**ARX(1,1)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{ccc}
0.0195 & +\,0.4736\,\pi_t & +\,0.5577\,SPF_t \\
(0.020) & (0.054) & (0.099)
\end{array}
\end{array}
$$

- Log-likelihood: **1.564**
- AIC: **2.871**, BIC: **16.473**
- Residual skewness: **-0.074**, excess kurtosis: **5.909**

**ARX(2,1)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{cccc}
0.0195 & +\,0.4843\,\pi_t & -\,0.0242\,\pi_{t-1} & +\,0.5740\,SPF_t \\
(0.020) & (0.075) & (0.077) & (0.092)
\end{array}
\end{array}
$$

- Log-likelihood: **1.757**
- AIC: **4.486**, BIC: **22.622**
- Residual skewness: **-0.060**, excess kurtosis: **5.950**

**ARX(2,2)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{ccccc}
0.0178 & +\,0.4874\,\pi_t & -\,0.0198\,\pi_{t-1} & +\,0.0108\,SPF_t & +\,0.5601\,SPF_{t-1} \\
(0.021) & (0.075) & (0.079) & (0.417) & (0.392)
\end{array}
\end{array}
$$

- Log-likelihood: **2.663**
- AIC: **4.674**, BIC: **27.343**
- Residual skewness: **-0.054**, excess kurtosis: **5.920**

