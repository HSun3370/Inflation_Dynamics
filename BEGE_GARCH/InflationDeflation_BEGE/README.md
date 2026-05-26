```{raw:typst}
#set page(margin: auto)
```


## Inflation and Deflation Model

Here $n_t, p_t$ is generated recursively with parameters $(p_0, n_0, \rho_p, \rho_n, \phi^+_p, \phi_n^-)$:

$$
\begin{aligned}
p_{t} &= p_0 + \rho_p \, p_{t-1}
        + \frac{\phi^+_p }{2  \sigma_p^2}\, (u_{t-1}^+)^2,\\
n_{t} &= n_0 + \rho_n \, n_{t-1}
      + \frac{\phi^- _n}{2  \sigma_n^2} \,(u_{t-1}^-)^2
\end{aligned}
$$

According to the emails, I have set the constraints below.

- $\rho_p + \frac{\phi_p^+}{2} < 1$ and $\rho_n + \frac{\phi_n^-}{2} < 1$.
- $\sigma_p^2 p_0 + \sigma_n^2 n_0 < \mathrm{Var}(\pi_t)$.
- $\max\{p_t, n_t\} < 200$