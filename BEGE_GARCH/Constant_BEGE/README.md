
```{raw:typst}
#set page(margin: auto)
```

## Constant BEGE 


BEGE with invariant shape parameters is

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

The constant implied variance path

$$
\sigma_p^2 \bar{p} + \sigma_n^2 \bar{n}
$$

is screened against the same project EWMA lower and upper bounds used by the
dynamic BEGE specifications during optimization and result collection.
