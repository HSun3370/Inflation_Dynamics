
```{raw:typst}
#set page(margin: auto)
```


# Mean Process Specification

Four types of mean processes are of interest:

- **Constant**: $\pi_{t+1} = SPF_t + \mu_{t+1}$
- **ARX(1,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \phi_1 SPF_t + \mu_{t+1}$
- **ARX(2,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \mu_{t+1}$
- **ARX(2,2)** : $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \phi_2 SPF_{t-1} + \mu_{t+1}$

where $\mu_{t+1}$ is the residual term.


<!-- [^1]: In the GARCH family estimation exercise, we found that  ARX(2,2) is always dominated by other three specifications. Thus, we stop estimating ARX(2,2) in BEGE estimation to saving computational resources. -->

## OLS Results

Estimation results of four mean models with HAC standard errors in parentheses below each estimate. (The log-likelihood is computed under the Gaussian assumption.)

**Constant**:

$$
\hat{\pi}_{t+1} = SPF_t
$$

No estimated coefficients. Residuals $\mu_{t+1} = \pi_{t+1} - SPF_t$.

- Log-likelihood: **-222.664**
- AIC: **447.329**, BIC: **450.700**
- Residual skewness: **-0.713**, excess kurtosis: **7.301**

**ARX(1,1)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{ccc}
0.0824 & +\,0.3005\,\pi_t & +\,0.7337\,SPF_t \\
(0.086) & (0.112) & (0.167)
\end{array}
\end{array}
$$

- Log-likelihood: **-207.381**
- AIC: **420.762**, BIC: **430.874**
- Residual skewness: **-1.047**, excess kurtosis: **8.210**

**ARX(2,1)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{cccc}
0.0897 & +\,0.2892\,\pi_t & +\,0.0834\,\pi_{t-1} & +\,0.6385\,SPF_t \\
(0.086) & (0.098) & (0.126) & (0.251)
\end{array}
\end{array}
$$

- Log-likelihood: **-206.810**
- AIC: **421.619**, BIC: **435.102**
- Residual skewness: **-1.191**, excess kurtosis: **9.050**

**ARX(2,2)**:

$$
\begin{array}{rl}
\hat{\pi}_{t+1} = & \begin{array}{ccccc}
0.0856 & +\,0.2992\,\pi_t & +\,0.0914\,\pi_{t-1} & +\,0.4084\,SPF_t & +\,0.2136\,SPF_{t-1} \\
(0.086) & (0.106) & (0.125) & (0.433) & (0.297)
\end{array}
\end{array}
$$

- Log-likelihood: **-206.660**
- AIC: **423.319**, BIC: **440.172**
- Residual skewness: **-1.212**, excess kurtosis: **9.129**

For the following volatility estimation process, I impose the bounds on mean process parameters for stationarity.
**Table 1: Parameter Bounds for Mean Process Specifications**

| Model     | $c$                         | $\rho_1$            | $\rho_2$            | $\phi_1$       | $\phi_2$       |
|-----------|-----------------------------|---------------------|---------------------|----------------|----------------|
| Constant  | ---                         | ---                 | ---                 | ---            | ---            |
| ARX(1,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1,\ 1)$  | ---                 | $(-10,\ 10)$   | ---            |
| ARX(2,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-2,\ 2)$  | $(-1,\ 1)$  | $(-10,\ 10)$   | ---            |
| ARX(2,2)  | $(\min \pi_t,\ \max \pi_t)$ | $(-2,\ 2)$  | $(-1,\ 1)$  | $(-10,\ 10)$   | $(-10,\ 10)$   |