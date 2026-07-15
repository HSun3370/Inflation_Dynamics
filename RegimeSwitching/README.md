```{raw:typst}
#set page(margin: auto)
```

# Regime-Switching Inflation Models


Let the latent state be
$$
s_t \in \{1,\dots,K\}, \qquad \Pr(s_t = j | s_{t-1} = i) = p_{ij}.
$$

I estimate several specifications that differ in **which blocks of parameters are allowed to switch with $s_t$**. Taking the ARX(2,2) model as an example,
$$
\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1\, \mathrm{SPF}_t + \phi_2\, \mathrm{SPF}_{t-1} + \mu_{t+1},
$$
the parameters are grouped into three blocks, and each block can independently be made regime-dependent:

| Block | State-dependent parameters |
|---|---|
| AR component | $c_{s_t},\ \rho_{1,s_t},\ \rho_{2,s_t}$ |
| SPF component | $\phi_{1,s_t},\ \phi_{2,s_t}$ |
| Shock distribution | $\sigma_{s_t}$ |

A given specification is defined by selecting any subset of these three blocks
to be switching; parameters in blocks not selected are held constant across
regimes.

## Estimation And Collection

For each regime-switching specification, I run 50 optimizer starts. Each start
first uses 50 EM iterations to improve the starting values and then applies MLE
to fine tune the likelihood. The collected estimate for a model specification is
the highest log-likelihood attempt that passes all checks:

- optimizer convergence,
- AR mean-process stationarity in every regime,
- positive non-boundary variance,
- and valid non-boundary transition probabilities.

The collection step rejects selected attempts with variance estimates at or
below `max(1e-6, 1e-3 * var(y))`, where `y` is the dependent variable of the
specification (`u_t` for the Constant mean process, `\pi_t` for the ARX mean
processes), or transition probabilities within `1e-4` of 0 or 1.

For the AR stationarity check, only the inflation-lag coefficients enter the AR
polynomial. SPF coefficients are treated as exogenous loadings. Specifications
where none of the 50 attempts passes all checks are excluded from the results
table; they are not reported even as a fallback, because the highest-likelihood
attempt for such specifications is typically a degenerate solution (e.g., a
collapsed regime variance that inflates the log-likelihood artificially).

By default only normal-residual specifications are estimated. Student's $t$
variants can be estimated as a robustness exercise with the
`--include-student-t` flag, but they are always excluded from the reported
tables.

## Data Frequency

By default `estimate_regime_switching_models.py` estimates on the quarterly
effective sample (`DataSummary/Aggregate_CPI_inflation_Quarterly.pkl`,
1969Q2--2026Q1, 228 observations after the 2026-07-14 data update; the
committed quarterly results in `results/` were estimated on the previous
1969Q2--2022Q4, 215-observation sample) and writes to `results/`. The same
specification menu can be estimated on the monthly effective sample
(`DataSummary/Aggregate_CPI_inflation_Monthly.pkl`, 1969M2--2026M5,
688 observations) with the `--data-path` and `--output-dir` flags; monthly
results are stored in `results_monthly/`:

```
python estimate_regime_switching_models.py \
    --data-path ../DataSummary/Aggregate_CPI_inflation_Monthly.pkl \
    --output-dir results_monthly
```

All model settings (specification menu, starts, checks, and selection policy)
are identical across the two frequencies.
