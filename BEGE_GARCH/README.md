
```{raw:typst}
#set page(margin: auto)
```



#  BEGE Density

BEGE density function $f_{BEGE}(\mu | p, n, \sigma_p, \sigma_n)$ is the function that calculates the density of the observation $\mu$ given the parameters $\{p, n, \sigma_p, \sigma_n\}$.

To compare Justin's analytic BEGE density code with the old numerical one, I conducted an experiment using synthetic data. I generated $200$ independent observations from a standard normal distribution and fixed the BEGE parameters to

$$
p = 5, \qquad n = 7, \qquad \sigma_p = 0.8, \qquad \sigma_n = 1.2.
$$

I computed the sum of log-likelihood using two approaches:

1. The *old BEGE function*, which approximates the density via numerical integration on a uniform grid, and whose accuracy depends strongly on the number of grid points; and
2. *Justin's analytic BEGE function*.

The old numerical method was evaluated over a wide range of grid resolutions (up to $50{,}000$ points), and the resulting log-likelihood sums were plotted alongside the benchmark value computed from Justin's code.

![BEGE LLF Comparison](BEGELLF_comparison.png)

**Findings:**

- The old numerical likelihood converges toward Justin's analytic likelihood as the number of grid points increases, but the convergence is slow and requires very fine grids.
- For small or moderate grid sizes, the numerical integration *overestimates* the log-likelihood. This explains why, for a given set of parameters, Justin's implementation produces a lower log-likelihood than the old code.
- Justin's analytic BEGE formula provides a more accurate and more computationally efficient evaluation.


# BEGE GARCH Family




## Constant BEGE

The first model that I estimated is BEGE with invariant shape parameters, which is

$$
\begin{aligned}
\pi_{t+1} &= \hat{\pi}_{t+1} + u_{t+1}\\
u_{t+1} &= \sigma_p \omega_{p} - \sigma_n \omega_{n}
\end{aligned}
$$

where

$$
\omega_{p} \sim \tilde{\Gamma}(\bar{p}, 1), \quad \omega_{n} \sim \tilde{\Gamma}(\bar{n}, 1).
$$

$\bar{p}$ and $\bar{n}$ are unconditional shape parameters. The random search and bounds are set to be:

- $0.05 < \sigma_p, \sigma_n < 2$
- $0.1 < p, n < 10$

## Symmetric BEGE

The model assumes different scaled parameters and different constants in GJR recursions for both components:

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

Here $n_t, p_t$ is generated recursively through a GJR-type updating equation with parameters $(p_0, n_0, \rho, \phi^+, \phi^-)$:

$$
\begin{aligned}
p_{t} &= p_0 + \rho \, p_{t-1}
        + \frac{\phi^+ }{2  \sigma_p^2}\, (u_{t-1}^+)^2 + \frac{\phi^- }{2  \sigma_p^2}\,(u_{t-1}^-)^2,\\
n_{t} &= n_0 + \rho \, n_{t-1}
        + \frac{\phi^+ }{2  \sigma_n^2}\, (u_{t-1}^+)^2 + \frac{\phi^- }{2  \sigma_n^2} \,(u_{t-1}^-)^2
\end{aligned}
$$

The parameter bounds are chosen to be:

- $0.005 < p_0, n_0 < 10$,
- $10^{-5} < \rho < 0.999$,
- $10^{-5} < \phi^+, \phi^- < 0.999$,
- $10^{-5} < \sigma_p, \sigma_n < 2$.

## Bad and Good Symmetric GARCH Model

Here $n_t, p_t$ is generated recursively with parameters $(p_0, n_0, \rho_p, \rho_n, \phi_p, \phi_n)$:

$$
\begin{aligned}
p_{t} &= p_0 + \rho_p \, p_{t-1}
        + \frac{\phi_p }{2  \sigma_p^2}\, (u_{t-1} )^2,\\
n_{t} &= n_0 + \rho_n \, n_{t-1}
        + \frac{\phi_n }{2  \sigma_n^2}\, (u_{t-1} )^2
\end{aligned}
$$

I have set the constraints below.

- $\rho + \phi < 1$ for both good and bad shape processes.
- $\sigma_p^2 p_0 + \sigma_n^2 n_0 < \mathrm{Var}(\pi_t)$.
- $\max\{p_t, n_t\} < 200$

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

#  Result Summary

Overall, most of the estimations with random initial values converge. I randomly sampled $100{,}000$ starting values for the constant BEGE model, and sampled $10{,}000$ for symmetric BEGE because it is more time consuming. I report the best estimation below.