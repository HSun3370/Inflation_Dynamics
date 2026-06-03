
```{raw:typst}
#set page(margin: auto)
```




# BEGE GARCH Family

To start the random search of initial mean parameters, I draw uniform samples from $(\mu - 2\sigma,\ \mu + 2\sigma)$ where $\mu$ and $\sigma$ are mean and standard deviation from OLS regression. I also set the AR coefficient bound to avoid the AR process to explode. We have four types of mean processes.

**Table 1: Parameter Bounds for Mean Process Specifications**

| Model     | $c$                         | $\rho_1$            | $\rho_2$            | $\phi_1$       | $\phi_2$       |
|-----------|-----------------------------|---------------------|---------------------|----------------|----------------|
| Constant  | ---                         | ---                 | ---                 | ---            | ---            |
| ARX(1,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-0.999,\ 0.999)$  | ---                 | $(-10,\ 10)$   | ---            |
| ARX(2,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | ---            |
| ARX(2,2)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | $(-10,\ 10)$   |

## Estimation Speed Controls

The BEGE likelihood now uses the analytic hypergeometric-$U$ density for
moderate shape paths and a cumulant-generating-function saddlepoint density for
large recursive shape states. This avoids the numerical overflow/cancellation
that can make direct `hyperu` evaluation report impossible log densities when
$p_t$ or $n_t$ is large. Exact high-precision `hyperu` checks can still be
forced with `hyperu_method="mpmath"` or the estimator-specific
`density_hyperu_method="mpmath"`.

For the multi-start BEGE estimators, robust numerical standard errors are optional through `compute_se`. The default is `compute_se=False` for fast model search; the result collectors recompute standard errors for the reported top log-likelihood rows after selection.

## Best-Model Reporting Screen

Raw BEGE search outputs are kept in `output/raw/draw_###.csv`. The collectors
recompute the likelihood from each stored parameter vector using the stabilized
BEGE density before writing `results/all_estimations.csv`, so stale likelihoods
from earlier density code do not determine the best model. The cleaned
estimations are also split by mean process under `results/by_mean/` as
`constant.csv`, `ARX_1_1.csv`, `ARX_2_1.csv`, and `ARX_2_2.csv`; these
by-mean files keep only rows with successful optimizer status and
`selection_eligible=True`.

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
status, mean stationarity status, and the exclusion reason for each saved
estimation row. A log-likelihood value above `-150` is recorded as a manual
review diagnostic but is not a selection exclusion rule.

Each `results/best_model.md` reports the top 20 admissible rows by corrected
log likelihood within each mean process. For every reported row, the markdown
first shows the relevant mean-process equation and BEGE volatility-process
equation with the estimated parameters substituted directly into the equations;
standard errors appear below the substituted estimates in parentheses. The same
top-20 rows with standard errors are written to
`results/best_loglik_top20_with_se.csv`.
 
