
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

- $0 \leq p_0, n_0 < 10$,
- $0 \leq \rho \leq 1$,
- $0 \leq \phi^+, \phi^- \leq 2$,
- $10^{-5} < \sigma_p, \sigma_n < 2$.

The current estimation code enforces the stability condition

$$
\rho + \frac{\phi^+}{2} + \frac{\phi^-}{2} < 1.
$$
 
The optimizer and collector also enforce the EWMA implied-variance bounds on
$\sigma_p^2 p_t + \sigma_n^2 n_t$ at every effective-sample observation.

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
- `results/by_mean/constant.csv`, `results/by_mean/ARX_1_1.csv`, `results/by_mean/ARX_2_1.csv`, and `results/by_mean/ARX_2_2.csv`, which split the cleaned estimations by mean process.
- `results/selection_diagnostics.csv`, with stored and corrected likelihood criteria plus selection diagnostics.
- `results/best_loglik_with_se.csv`, with standard errors for the likelihood-best admissible fit.
- `results/best_model.md`, with only the likelihood-best admissible fit, reported as substituted mean and BEGE volatility equations with standard errors shown below the parameter values.

When `START_ID` and `END_ID` are set, the collector only merges `draw_###.csv` files in that seed range. This prevents older seed files from entering a new smaller resubmission.

`SimulationSymmetric.sh` defaults to 50 seed jobs, 40 draws per mean process, 25 optimizer starts per draw, and 800 SLSQP iterations per start. This gives 50,000 optimizer starts for each mean process and Symmetric BEGE specification. With all four mean processes, that is 200,000 optimizer starts total.
