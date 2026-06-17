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
- The implied variance path $\sigma_p^2 p_t + \sigma_n^2 n_t$ must satisfy the project EWMA lower and upper bounds at every effective-sample observation.

The hard cap $\max\{p_t, n_t\} < 200$ is not imposed during raw multi-start
optimization or reported best-model selection by default. The stabilized BEGE
density is evaluated directly as long as the recursive shape series are finite;
large shape states use the saddlepoint density backend. The collector writes
the maximum shape path as a diagnostic in `results/selection_diagnostics.csv`.

For the resubmitted stability test, the first recursive shape states are no
longer initialized with the parameter-implied unconditional values. Instead,
each mean process uses the corresponding Constant BEGE fixed-shape estimates
from `BEGE_GARCH/Constant_BEGE/results/constant_bege_initial_shapes_by_mean.csv`:

| Mean process | Fixed $p_{\mathrm{init}}$ | Fixed $n_{\mathrm{init}}$ |
|---|---:|---:|
| Constant | 2.6759 | 0.1860 |
| ARX(1,1) | 2.6277 | 0.2811 |
| ARX(2,1) | 3.3171 | 0.2281 |
| ARX(2,2) | 2.6648 | 0.2821 |

The recursion intercept parameters named `p0` and `n0` remain freely estimated
model parameters. Only the starting states $p_{\mathrm{init}}$ and
$n_{\mathrm{init}}$ are fixed. The raw output records these fixed states as
`recursion_init_p` and `recursion_init_n`, and the collector uses those columns
when recomputing corrected likelihoods and path diagnostics. The fixed
starting states do not replace `p0` or `n0` in the parameter vector.

## Estimation Workflow

`BG_GJR1.py` estimates the BadGood BEGE specification on the canonical effective sample from `DataSummary/README.md`, **1969Q2--2022Q4** with **215 observations**. The runner uses the precomputed lag columns when they are available, so lagged ARX regressors do not trim the sample again.

The script estimates the four mean-process specifications:

- Constant
- ARX(1,1)
- ARX(2,1)
- ARX(2,2)

Results are checkpointed to `output/raw/draw_###.csv` after each draw. This keeps completed mean-process rows when a seed job is interrupted, while the final write still contains all requested rows when the seed finishes.

`collect_bg_results.py` merges the raw seed files and writes:

- `results/all_estimations.csv`, without standard-error columns or optimizer messages.
- `results/by_mean/constant.csv`, `results/by_mean/ARX_1_1.csv`, `results/by_mean/ARX_2_1.csv`, and `results/by_mean/ARX_2_2.csv`, which split the eligible cleaned estimations by mean process and retain empirical path quantiles.
- `results/selection_diagnostics.csv`, with stored and corrected likelihood criteria plus selection diagnostics.
- `results/path_quantile_diagnostics.csv`, with each estimate's parameters and empirical 5%, median, and 95% quantiles for $p_t$, $n_t$, $\sigma_t^2$, $s_t^2$, and $k_t^2$ from the fixed effective-sample recursion.
- `results/best_loglik_with_se.csv`, with standard errors for the likelihood-best admissible fit.
- `results/best_model.md`, with only the likelihood-best admissible fit, reported as substituted mean and BEGE volatility equations with standard errors shown below the parameter values.

The by-mean CSV files keep only estimates with `optimizer_success=True` and
`selection_eligible=True`; excluded raw-search rows remain in
`results/all_estimations.csv` and `results/selection_diagnostics.csv`.

When `START_ID` and `END_ID` are set, the collector only merges `draw_###.csv` files in that seed range. This prevents older seed files from entering a new smaller resubmission.

`SimulationBG.sh` defaults to 50 seed jobs, 40 draws per mean process, 25 optimizer starts per draw, and 800 SLSQP iterations per start. This gives 50,000 optimizer starts for each mean process and BadGood BEGE specification. With all four mean processes, that is 200,000 optimizer starts total. The `estimate` action prints a rough per-seed time estimate before submitting jobs.
