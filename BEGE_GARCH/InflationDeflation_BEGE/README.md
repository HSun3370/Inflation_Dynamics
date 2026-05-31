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

For the current random-search estimation runs, the likelihood is evaluated under
the parameter bounds used in code, finite-recursion checks, and the documented
ID-GARCH stability restrictions:

- $\rho_p + \frac{\phi_p^+}{2} < 1$ and $\rho_n + \frac{\phi_n^-}{2} < 1$.
- $\sigma_p^2 p_0 + \sigma_n^2 n_0 < \mathrm{Var}(\pi_t)$.

The hard shape cap $\max\{p_t, n_t\} < 200$ is **not imposed by default** in
`ID_GARCH`. It can still be restored for sensitivity checks by passing
`cap_pn=200`.

For reported best-model selection, `collect_id_results.py` keeps the raw search
rows but applies the canonical $\max\{p_t, n_t\} < 200$ screen together with
optimizer-convergence checks. The row-level selection outcome is written to
`results/selection_diagnostics.csv`.

## Estimation Workflow

The Slurm workflow in `SimulationID.sh` submits 100 seed jobs by default. Each
seed estimates all four mean processes on the fixed effective sample
`1969Q2--2022Q4`:

- Constant
- ARX(1,1)
- ARX(2,1)
- ARX(2,2)

The job runner skips standard-error calculations and saves one CSV row after
each model fit so partial jobs leave recoverable output. The collector merges
the raw CSV files, picks the best results, and computes standard errors only
for the best AIC fit within each mean process.
