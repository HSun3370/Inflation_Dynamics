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

The parameter bounds follow the Full BEGE bounds:

- $0 \leq p_0, n_0 < 10$,
- $0 \leq \rho_n, \rho_p \leq 1$,
- $0 \leq \phi_n^+, \phi_n^- \leq 2$,
- $10^{-5} < \sigma_p, \sigma_n < 2$.

For the current random-search estimation runs, the likelihood is evaluated under
the parameter bounds used in code, finite-recursion checks, and the documented
ID-GARCH stability restrictions:

- $\rho_p + \frac{\phi_p^+}{2} < 1$ and $\rho_n + \frac{\phi_n^-}{2} < 1$.
- The implied variance path $\sigma_p^2 p_t + \sigma_n^2 n_t$ must satisfy the project EWMA lower and upper bounds at every effective-sample observation.

The legacy hard shape cap on $\max\{p_t, n_t\}$ has been removed from
`ID_GARCH`; large shape states are retained as diagnostics rather than used as
an exclusion rule.

For reported best-model selection, `collect_id_results.py` keeps the raw search
rows, recomputes each likelihood with the stabilized BEGE density, and uses
finite corrected criteria, optimizer convergence, and the documented
parameter/stability constraints. The row-level selection
outcome, stored likelihood, corrected likelihood, and maximum shape diagnostic
are written to `results/selection_diagnostics.csv`.

## Estimation Workflow

The Slurm workflow in `SimulationID.sh` submits 50 seed jobs by default. Each
seed estimates all four mean processes on the fixed effective sample
`1969Q2--2022Q4`:

- Constant
- ARX(1,1)
- ARX(2,1)
- ARX(2,2)

The job runner skips standard-error calculations and saves one CSV row after
each model fit so partial jobs leave recoverable output. Each seed uses 40
draws per mean process and 25 optimizer starts per draw, giving 50,000
optimizer starts for each mean process and Inflation/Deflation BEGE
specification. The collector merges the raw CSV files, writes the cleaned
combined output to `results/all_estimations.csv`, and writes admissible cleaned
outputs split by mean process under `results/by_mean/` as `constant.csv`,
`ARX_1_1.csv`, `ARX_2_1.csv`, and `ARX_2_2.csv`. The by-mean CSV files keep
only rows with `optimizer_success=True` and `selection_eligible=True`, and they
retain empirical 5%, median, and 95% path quantiles for $p_t$, $n_t$,
$\sigma_t^2$, $s_t^2$, and $k_t^2$. The collector also writes
`results/path_quantile_diagnostics.csv` with the same path quantiles beside
each stored parameter vector. It ranks admissible estimates by corrected log
likelihood, computes standard errors for the likelihood-best admissible fit in
`results/best_loglik_with_se.csv`, and writes `results/best_model.md` with only
that selected fit as substituted mean and BEGE volatility equations. Standard
errors are shown below the parameter values in those reported equations.
