
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

## Result Collection

`collect_constant_results.py` merges the raw seed files, writes the cleaned
combined output to `results/all_estimations.csv`, and splits the same cleaned
output by mean process under `results/by_mean/` as `constant.csv`,
`ARX_1_1.csv`, `ARX_2_1.csv`, and `ARX_2_2.csv`. It ranks admissible estimates
by log likelihood within each mean process, computes standard errors for the
top 20 rows per mean process in `results/best_loglik_top20_with_se.csv`, and
writes `results/best_model.md` with substituted mean and fixed-shape BEGE
equations. Standard errors are shown below the parameter values in those
reported equations.
