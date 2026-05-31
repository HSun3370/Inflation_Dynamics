
```{raw:typst}
#set page(margin: auto)
```



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

The current estimation code enforces the stability condition

$$
\rho + \frac{\phi^+}{2} + \frac{\phi^-}{2} < 1,
$$

and the same variance guard used by the current BEGE searches,

$$
\sigma_p^2 p_0 + \sigma_n^2 n_0 < 0.87.
$$

The hard cap $\max\{p_t, n_t\} < 200$ is not imposed during raw multi-start
optimization by default. The stabilized BEGE density is evaluated directly as
long as the recursive shape series are finite. For reported best-model
selection, however, `collect_symmetric_results.py` applies the canonical
$\max\{p_t, n_t\} < 200$ screen and writes the row-level outcome to
`results/selection_diagnostics.csv`.

## Batch Estimation

`BEGE_symmetric1.py` estimates the Symmetric BEGE specification on the canonical effective sample from `DataSummary/README.md`, **1969Q2--2022Q4** with **215 observations**. The runner uses the precomputed lag columns when they are available, so lagged ARX regressors do not trim the sample again.

The script estimates the four mean-process specifications:

- Constant
- ARX(1,1)
- ARX(2,1)
- ARX(2,2)

Results are checkpointed to `output/raw/draw_###.csv` after each draw. This keeps completed mean-process rows when a seed job is interrupted, while the final write still contains all requested rows when the seed finishes.

`collect_symmetric_results.py` merges the raw seed files and writes:

- `results/all_estimations.csv`, without standard-error columns or optimizer messages.
- `results/best_model.md`, with the best log-likelihood model for each mean process and standard errors for the reported parameter estimates.

When `START_ID` and `END_ID` are set, the collector only merges `draw_###.csv` files in that seed range. This prevents older seed files from entering a new smaller resubmission.

`SimulationSymmetric.sh` defaults to 400 seed jobs, 10 draws per mean process, 10 optimizer starts per draw, and 300 SLSQP iterations per start. This gives 40,000 optimizer starts for each mean process. With all four mean processes, that is 160,000 optimizer starts total.
