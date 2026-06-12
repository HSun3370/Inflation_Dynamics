
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

- $10^{-5} < \sigma_p, \sigma_n < 2$,
- $0.1 < p, n < 10$.

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
`ARX_1_1.csv`, `ARX_2_1.csv`, and `ARX_2_2.csv`. The by-mean files retain
empirical 5%, median, and 95% path quantiles for $p_t$, $n_t$,
$\sigma_t^2$, $s_t^2$, and $k_t^2$. The collector also writes
`results/path_quantile_diagnostics.csv` with the same path quantiles beside
each stored parameter vector. It ranks admissible estimates by log likelihood,
computes standard errors for the likelihood-best admissible fit in
`results/best_loglik_with_se.csv`, and writes `results/best_model.md` with
only that selected fit as substituted mean and fixed-shape BEGE equations.
Standard errors are shown below the parameter values in those reported
equations.
