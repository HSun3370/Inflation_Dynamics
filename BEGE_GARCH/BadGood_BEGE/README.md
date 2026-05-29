```{raw:typst}
#set page(margin: auto)
```


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

The old hard cap $\max\{p_t, n_t\} < 200$ is no longer imposed by default. The stabilized BEGE density is evaluated directly as long as the recursive shape series are finite.

## Estimation Workflow

`BG_GJR1.py` estimates the BadGood BEGE specification on the canonical effective sample from `DataSummary/README.md`, **1969Q2--2022Q4** with **215 observations**. The runner uses the precomputed lag columns when they are available, so lagged ARX regressors do not trim the sample again.

The script estimates the four mean-process specifications:

- Constant
- ARX(1,1)
- ARX(2,1)
- ARX(2,2)

Results are kept in memory during each seed job and written once, after the seed job finishes, to `output/raw/draw_###.csv`. This avoids reporting partial result files from long-running jobs.

`collect_bg_results.py` merges the raw seed files and writes:

- `results/all_estimations.csv`, without standard-error columns or optimizer messages.
- `results/best_model.md`, with the best log-likelihood model for each mean process and standard errors for the reported parameter estimates.

`SimulationBG.sh` defaults to 400 seed jobs, 1 draw per mean process, and 25 optimizer starts per draw. This gives 10,000 optimizer starts for each mean process. With all four mean processes, that is 40,000 optimizer starts total. The `estimate` action prints a rough per-seed time estimate before submitting jobs.
