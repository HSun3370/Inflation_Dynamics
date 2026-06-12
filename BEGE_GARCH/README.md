
```{raw:typst}
#set page(margin: auto)
```
 
# BEGE GARCH Family
 
To start the random search of initial mean parameters, I draw uniform samples from $(\mu - 2\sigma,\ \mu + 2\sigma)$ where $\mu$ and $\sigma$ are mean and standard deviation from OLS regression. I also set the AR coefficient bound to avoid the AR process to explode. We have four types of mean processes.

## Implied Moment Dynamics
The BEGE GARCH models have nice properties

- conditional variance: $\sigma_t^2 = \sigma^2_p p_t + \sigma^2_n n_t$,
- conditional skewness: $s_t^2 = 2 (\sigma^3_p p_t - \sigma^3_n n_t)$,
- conditional excess kurtosis: $k_t^2 = 6(\sigma^4_p p_t + \sigma^4_n n_t)$.

The baseline BEGE volatility menu consists of Constant, Symmetric,
InflationDeflation, BadGood, and Full BEGE. Two explicitly requested Full BEGE
restrictions are also estimated here:

- Constant-p Full BEGE: $p_t=p_0$ and $n_t$ follows the unrestricted Full BEGE GJR update.
- Constant-n Full BEGE: $n_t=n_0$ and $p_t$ follows the unrestricted Full BEGE GJR update.

**Used for recursive constraints!!**

## Random Search

For the multi-start BEGE estimators, robust numerical standard errors are optional through `compute_se`. The default is `compute_se=False` for fast model search; the result collectors recompute standard errors for the reported top log-likelihood rows after selection.

## Best-Model Reporting Screen

Raw BEGE search outputs are kept in `output/raw/draw_###.csv`. The collectors
recompute the likelihood from each stored parameter vector using the stabilized
BEGE density before writing `results/all_estimations.csv`, so stale likelihoods
from earlier density code do not determine the best model. The cleaned
estimations are also split by mean process under `results/by_mean/` as
`constant.csv`, `ARX_1_1.csv`, `ARX_2_1.csv`, and `ARX_2_2.csv`; these
by-mean files keep only rows with successful optimizer status and
`selection_eligible=True` and retain empirical 5%, median, and 95% path
quantiles for $p_t$, $n_t$, $\sigma_t^2$, $s_t^2$, and $k_t^2$.

Reported best-model
tables require finite corrected likelihood criteria, successful optimizer
convergence, finite positive shape paths, positive conditional variance paths,
mean-process stationarity, the documented parameter/stability constraints, and
the EWMA implied-variance bounds applied to
$\sigma_p^2 p_t + \sigma_n^2 n_t$.

The implied-variance bounds use the same screen during optimization and
collection. For residuals from the effective sample, compute an EWMA path with
$\lambda = 0.94$ and an initialization window of
$\tau = \min(75, T)$. The lower and upper paths are

$$
\max(EWMA_t / 10^6,\ \operatorname{Var}(u_t) / 10^8)
\leq
\sigma_p^2 p_t + \sigma_n^2 n_t
\leq
\max\left\{\min(EWMA_t \cdot 10^6,\ 10^7(1 + \max u_t^2)),\ 1 + \max u_t^2\right\}.
$$

The diagnostic value $\max_t\{p_t,n_t\}$ is recorded but is not used as a
selection exclusion rule. The companion `results/selection_diagnostics.csv`
records stored versus corrected likelihood criteria, high-shape density usage,
implied persistence, minimum scale, maximum shape path, implied-variance-bound
status, mean stationarity status, empirical 5%, median, and 95% path quantiles,
and the exclusion reason for each saved estimation row. The narrower
`results/path_quantile_diagnostics.csv` keeps the same path quantiles beside
the stored parameter vector for each estimate. A log-likelihood value above
`-150` is recorded as a manual review diagnostic but is not a selection
exclusion rule.

Each `results/best_model.md` reports only the single likelihood-best admissible
row for that BEGE specification. The markdown first shows the selected
mean-process equation and BEGE volatility-process equation with the estimated
parameters substituted directly into the equations; standard errors appear
below the substituted estimates in parentheses. The same selected row with
standard errors is written to `results/best_loglik_with_se.csv`.
 
