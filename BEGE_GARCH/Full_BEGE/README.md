```{raw:typst}
#set page(margin: auto)
```



## Full BEGE

BEGE GARCH has mean process and higher:

$$
\begin{aligned}
\pi_{t+1} &= \hat{\pi}_{t+1} + u_{t+1}, \\
u_{t+1} &= \sigma_p \, \omega_{p} - \sigma_n \, \omega_{n},
\end{aligned}
$$

where

$$
\omega_{p} \sim \tilde{\Gamma}(p_t, 1), \quad \omega_{n} \sim \tilde{\Gamma}(n_t, 1).
$$

Here $n_t, p_t$ is generated recursively through a GJR-type updating equation with parameters $(p_0, n_0, \rho_p, \rho_n, \phi^+_p, \phi_p^-, \phi_n^+, \phi_n^-)$:

$$
\begin{aligned}
p_{t} &= p_0 + \rho_p \, p_{t-1}
        + \frac{\phi^+_p }{2  \sigma_p^2}\, (u_{t-1}^+)^2 + \frac{\phi^-_p }{2  \sigma_p^2}\,(u_{t-1}^-)^2,\\
n_{t} &= n_0 + \rho_n \, n_{t-1}
        + \frac{\phi^+_n }{2  \sigma_n^2}\, (u_{t-1}^+)^2 + \frac{\phi^- _n}{2  \sigma_n^2} \,(u_{t-1}^-)^2
\end{aligned}
$$

The parameter bounds are chosen to be:

- $0.005 < p_0, n_0 < 10$,
- $10^{-5} < \rho_p, \rho_n < 0.999$,
- $10^{-5} < \phi^+_p, \phi_p^-, \phi_n^+, \phi_n^- < 0.999$,
- $10^{-5} < \sigma_p, \sigma_n < 2$.

According to the emails, I have set the constraints below.

- $\rho + \frac{\phi^+}{2} + \frac{\phi^-}{2} < 1$ for both BE and GE shape processes.
- $\sigma_p^2 p_0 + \sigma_n^2 n_0 < \mathrm{Var}(\pi_t)$ where $\mathrm{Var}(\pi_t) = 0.75$ is the unconditional variance of inflation.
- $\max\{p_t, n_t\} < 200$.