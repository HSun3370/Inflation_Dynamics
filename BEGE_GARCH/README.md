
```{raw:typst}
#set page(margin: auto)
```



#  BEGE Density

BEGE density function $f_{BEGE}(\mu | p, n, \sigma_p, \sigma_n)$ is the function that calculates the density of the observation $\mu$ given the parameters $\{p, n, \sigma_p, \sigma_n\}$.

To compare Justin's analytic BEGE density code with the old numerical one, I conducted an experiment using synthetic data. I generated $200$ independent observations from a standard normal distribution and fixed the BEGE parameters to

$$
p = 5, \qquad n = 7, \qquad \sigma_p = 0.8, \qquad \sigma_n = 1.2.
$$

I computed the sum of log-likelihood using two approaches:

1. The *old BEGE function*, which approximates the density via numerical integration on a uniform grid, and whose accuracy depends strongly on the number of grid points; and
2. *Justin's analytic BEGE function*.

The old numerical method was evaluated over a wide range of grid resolutions (up to $50{,}000$ points), and the resulting log-likelihood sums were plotted alongside the benchmark value computed from Justin's code.

![BEGE LLF Comparison](BEGELLF_comparison.png)

**Findings:**

- The old numerical likelihood converges toward Justin's analytic likelihood as the number of grid points increases, but the convergence is slow and requires very fine grids.
- For small or moderate grid sizes, the numerical integration *overestimates* the log-likelihood. This explains why, for a given set of parameters, Justin's implementation produces a lower log-likelihood than the old code.
- Justin's analytic BEGE formula provides a more accurate and more computationally efficient evaluation.


# BEGE GARCH Family

To start the random search of initial mean parameters, I draw uniform samples from $(\mu - 2\sigma,\ \mu + 2\sigma)$ where $\mu$ and $\sigma$ are mean and standard deviation from OLS regression. I also set the AR coefficient bound to avoid the AR process to explode. We have four types of mean processes.

**Table 1: Parameter Bounds for Mean Process Specifications**

| Model     | $c$                         | $\rho_1$            | $\rho_2$            | $\phi_1$       | $\phi_2$       |
|-----------|-----------------------------|---------------------|---------------------|----------------|----------------|
| Constant  | ---                         | ---                 | ---                 | ---            | ---            |
| ARX(1,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-0.999,\ 0.999)$  | ---                 | $(-10,\ 10)$   | ---            |
| ARX(2,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | ---            |
| ARX(2,2)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | $(-10,\ 10)$   |



## Constant BEGE

The first model that I estimated is BEGE with invariant shape parameters, which is

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

- $0.05 < \sigma_p, \sigma_n < 2$
- $0.1 < p, n < 10$

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
- $\max\{p_t, n_t\} < 200$

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

According to the emails, I have set the constraints below.

- $\rho_p + \frac{\phi_p^+}{2} < 1$ and $\rho_n + \frac{\phi_n^-}{2} < 1$.
- $\sigma_p^2 p_0 + \sigma_n^2 n_0 < \mathrm{Var}(\pi_t)$.
- $\max\{p_t, n_t\} < 200$

## Full BEGE

BEGE GARCH has mean process and higher:

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

Here $n_t, p_t$ is generated recursively through a GJR-type updating equation with parameters $(p_0, n_0, \rho_p, \rho_n, \phi^+_p, \phi_p^-, \phi_n^+, \phi_n^-)$:

$$
\begin{aligned}
p_{t} &= p_0 + \rho_p \, p_{t-1}
        + \frac{\phi^+_p }{2  \sigma_p^2}\, (u_{t-1}^+)^2 + \frac{\phi^-_p }{2  \sigma_p^2}\,(u_{t-1}^-)^2,\\
n_{t} &= n_0 + \rho_n \, n_{t-1}
        + \frac{\phi^+_n }{2  \sigma_n^2}\, (u_{t-1}^+)^2 + \frac{\phi^- _n}{2  \sigma_n^2} \,(u_{t-1}^-)^2
\end{aligned}
$$

The parameter bounds are chosen to be:

- $0.005 < p_0, n_0 < 10$,
- $10^{-5} < \rho_p, \rho_n < 0.999$,
- $10^{-5} < \phi^+_p, \phi_p^-, \phi_n^+, \phi_n^- < 0.999$,
- $10^{-5} < \sigma_p, \sigma_n < 2$.

According to the emails, I have set the constraints below.

- $\rho + \frac{\phi^+}{2} + \frac{\phi^-}{2} < 1$ for both BE and GE shape processes.
- $\sigma_p^2 p_0 + \sigma_n^2 n_0 < \mathrm{Var}(\pi_t)$ where $\mathrm{Var}(\pi_t) = 0.75$ is the unconditional variance of inflation.
- $\max\{p_t, n_t\} < 200$.

#  Result Summary

I loaded the locally saved BEGE random-search outputs from the ignored `RandomDraw_*/*/summary_draws.csv` folders and converted them into compact result artifacts under `BEGE_GARCH/results/`. The raw random-draw folders and job logs remain ignored because they are large scratch outputs; the tracked files retain the final comparison tables and the selected parameter vectors.

The result loader is `BEGE_GARCH/collect_bege_results.py`. It records the source folder, mean specification, estimating script, number of local estimates, best raw AIC/BIC values, and best values after applying the documented BEGE checks. The generated tracked files are:

- `BEGE_GARCH/results/bege_model_comparison.csv`
- `BEGE_GARCH/results/bege_best_parameters.csv`
- `BEGE_GARCH/results/bege_best_models.md`

## Sample and Comparability Note

The existing BEGE estimation scripts load `Aggregate_CPI_inflation.pkl`. That file currently contains **210 observations**, from **1970Q3** to **2022Q4**. This differs from the project-wide comparable likelihood sample documented in `DataSummary/README.md`, which is **1969Q2--2022Q4** with **215 observations**. Therefore, the BEGE results reported here should be treated as imported local BEGE runs. They should not be compared one-for-one with the GARCH and regime-switching likelihoods until the BEGE estimations are rerun on the canonical 215-observation effective sample.

## Constraint-Screened Local Results

For the dynamic BEGE specifications, I screen the local estimates using the documented persistence restrictions, the variance reference bound, and the shape-path cap $\max\{p_t,n_t\}<200$. The table below reports the best AIC estimate among rows passing those checks for the current local result families. The complete audit table, including archival folders and raw best rows that fail checks, is in `BEGE_GARCH/results/bege_model_comparison.csv`.

| Model family | Mean | Source folder | Passing / total | LogLik | AIC | BIC |
|---|---|---|---:|---:|---:|---:|
| Constant BEGE | ARX(1,1) | `RandomDraw_Constant` | 50000 / 50000 | 1770.014 | -3526.028 | -3502.599 |
| Constant BEGE | ARX(2,2) | `RandomDraw_Constant` | 50000 / 50000 | 1297.510 | -2577.019 | -2546.895 |
| Constant BEGE | ARX(2,1) | `RandomDraw_Constant` | 50000 / 50000 | 372.678 | -729.355 | -702.578 |
| Inflation/Deflation BEGE-GARCH | ARX(1,1) | `RandomDraw_ID` | 1999 / 2000 | -14.515 | 51.029 | 87.848 |
| Bad/Good BEGE-GARCH | ARX(1,1) | `RandomDraw_BG_GARCH` | 1946 / 2000 | -29.540 | 81.080 | 117.898 |
| Bad/Good BEGE-GARCH | Constant | `RandomDraw_BG_GARCH` | 1222 / 2000 | -37.420 | 90.839 | 117.616 |
| Inflation/Deflation BEGE-GARCH | Constant | `RandomDraw_ID` | 1931 / 2000 | -84.888 | 185.776 | 212.553 |
| Constant BEGE | Constant | `RandomDraw_Constant` | 50000 / 50000 | -131.798 | 271.595 | 284.984 |
| Full BEGE-GJR | Constant | `RandomDraw_GJR_Oct` | 6723 / 10000 | -163.370 | 346.739 | 380.210 |
| Shared-GJR BEGE | ARX(1,1) | `RandomDraw_Symmetric_Oct` | 2000 / 2000 | -166.330 | 352.660 | 386.131 |
| Full BEGE-GJR | ARX(1,1) | `RandomDraw_GJR_Oct` | 8286 / 10000 | -163.633 | 353.266 | 396.778 |
| Shared-GJR BEGE | Constant | `RandomDraw_Symmetric_Oct` | 2000 / 2000 | -171.051 | 356.102 | 379.531 |

The ARX(2,2) entry appears because it exists in the local constant-BEGE output. For new BEGE runs, the project default remains to skip ARX(2,2) unless explicitly requested, consistent with `MeanProcess/README.md`.

## Tracked Estimation Code

The Python estimators used to generate the BEGE local runs are already tracked in this repository: `BEGE_GARCH.py`, `BEGE_density.py`, `BEGE_constant.py`, `BEGE_symmetric1.py`, `BEGE_symmetric2.py`, `BG_GJR1.py`, `BG_GJR2.py`, `ID_GJR1.py`, `ID_GJR2.py`, `BEGE_GJR1.py`, and `BEGE_GJR2.py`. The new result loader keeps the mapping from result folder to estimating script in the generated CSV files.
