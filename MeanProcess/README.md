
```{raw:typst}
#set page(margin: auto)
```


# Mean Process Specification

Four types of mean processes are of interest:

- **Constant**: $\pi_{t+1} = SPF_t + \mu_{t+1}$
- **ARX(1,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \phi_1 SPF_t + \mu_{t+1}$
- **ARX(2,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \mu_{t+1}$
- **ARX(2,2)**[^1]: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \phi_2 SPF_{t-1} + \mu_{t+1}$

where $\mu_{t+1}$ is the residual term.


[^1]: In the GARCH family estimation exercise, we found that  ARX(2,2) is always dominated by other three specifications. Thus, we stop estimating ARX(2,2) in BEGE estimation to saving computational resources.

#  OLS Results

I report OLS estimation results of three mean models with HAC standard error[^2]:

[^2]: Standard errors are heteroskedasticity and autocorrelation robust (HAC) using 12 lags and without small sample correction.

**ARX(1,1)**:

$$
\begin{aligned}
\hat{\pi}_{t+1}
&= 0.0720 + 0.2881\,\pi_t + 0.7508\, SPF_t \\
& ~ ~~(0.087) \quad (0.114) ~~~\quad (0.168)
\end{aligned}
$$

**ARX(2,1)**:

$$
\begin{aligned}
\hat{\pi}_{t+1}
&= 0.0792 + 0.2793\,\pi_t + 0.0728\,\pi_{t-1} + 0.6661\,SPF_t \\
& ~(0.087) ~~ \quad (0.102) ~~~ \quad (0.127) \quad ~~~~~~~~ (0.257)
\end{aligned}
$$

**ARX(2,2)**:

$$
\begin{aligned}
\hat{\pi}_{t+1}
&= 0.0761 + 0.2880\,\pi_t + 0.0799\,\pi_{t-1}
   + 0.4720\,SPF_t + 0.1793\,  SPF_{t-1} \\
& ~ ~ (0.088) ~~\quad (0.109) ~~~\quad (0.127)
~~~~~~~~   \quad (0.440) \quad ~~~~~~~(0.295)
\end{aligned}
$$
