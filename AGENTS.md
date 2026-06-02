# AGENTS.md

This file defines mandatory context and consistency rules for Codex contributions in this repository.

## Scope
These instructions apply to the entire project tree.

## Mandatory Context Loading
Before making or reviewing any model-related change, read these files listed in myst.yml--project--exports--articles first. If one of these files is empty or incomplete, do not invent missing model settings. Keep existing code behavior unchanged for that component and note the gap.

## Canonical Project Settings
When implementing, refactoring, documenting, or testing estimation code, keep these settings consistent unless the user explicitly requests a change.

### Shared Data and Residual Definitions
- Use common notation from `README.md`: inflation `pi_t`, expectation `hat(pi)_t`, residual `u_t`, and split residuals `u_t+`, `u_t-`.
- Keep residual definition consistent across modules: `u_t = pi_t - hat(pi)_t`.

### Effective Sample and Initialization
- Use the trimmed effective sample from `DataSummary/README.md`: `1969Q2` to `2022Q4` (`215` observations) for comparable likelihoods.
- Use the same estimation sample across mean and volatility specifications unless the task explicitly studies sample sensitivity.
- For all estimation exercises, treat the effective sample size (`215`) as fixed and comparable across models.
- If lagged regressors (for example `Inflation_lag_1`, `Inflation_lag_2`, `SPF_lag_1`) are already precomputed in the effective-sample dataset, use them directly rather than trimming observations again.
- For GARCH-type pre-sample recursion terms, initialize missing pre-sample variance states with the model-implied unconditional variance where applicable.

### Mean Process Menu
From `MeanProcess/README.md`, keep mean-model options aligned with:
- Constant
- ARX(1,1)
- ARX(2,1)
- ARX(2,2)

For BEGE runs, default to skipping ARX(2,2) unless explicitly requested, matching documented practice.

### GARCH Family Settings
From `GARCH/README.md`:
- Supported volatility families: GARCH, GJR-GARCH, EGARCH.
- Supported innovation distributions: Normal, standardized Student's t, and two-component Gaussian mixture.
- Enforce stationarity/positivity constraints appropriate to each model form.
- Keep EGARCH centering convention consistent with project documentation (fixed `sqrt(2/pi)` convention unless explicitly changed project-wide).

### BEGE Model Specification Menu

For `BEGE_GARCH`, the canonical BEGE volatility specification menu consists of exactly five specifications unless the user explicitly requests otherwise:

1. `BadGood_BEGE`
2. `Constant_BEGE`
3. `Full_BEGE`
4. `InflationDeflation_BEGE`
5. `Symmetric_BEGE`

When implementing, refactoring, documenting, or testing BEGE estimation code:

* Keep these five BEGE specifications aligned across code, scripts, result collection, and documentation.
* Do not silently add, remove, rename, or merge BEGE specifications.
* If a BEGE specification is intentionally excluded from a run, state the reason explicitly in the script output, logs, or accompanying documentation.
* For BEGE runs, continue to run all four mean processes unless explicitly requested, consistent with documented practice.
* When submitting jobs in server, don't compute SE of paramters. When reporting the best results, compute SE for the best models and report them in md file. 
### BEGE Random-Start Estimation Protocol

BEGE likelihood maximization uses randomized initial starts. For server-side production runs, the canonical search protocol is:

* Submit `50` independent jobs for each BEGE volatility specification.
* Within each job, use:

  ```python
  parser.add_argument("--n-draws", type=int, default=40, help="Number of random draws per mean specification")
  parser.add_argument("--n-starts", type=int, default=25, help="MLE restarts per draw")
  ```
* Therefore, for each mean-model and BEGE-volatility-specification pair, the intended production search covers:

  ```text
  50 jobs × 40 draws per job × 25 starts per draw = 50,000 starting points
  ```

When modifying estimation scripts:

* Preserve this job/draw/start interpretation unless the user explicitly requests a change.
* Do not reinterpret `--n-draws` or `--n-starts` in a way that changes the total number of candidate starting points.
* Ensure job-level random seeds or job identifiers generate independent randomized starts across the 50 jobs.
* Result aggregation must treat the 50 jobs as one combined search for the same model, mean process, and BEGE specification.

### BEGE Best-Model Collection Checks

When collecting, ranking, or summarizing best BEGE estimates, do not rely only on optimizer convergence or likelihood value. The best-model collection stage must also check the relevant admissibility conditions below.

#### Mean-Process Stationarity

For any estimated mean process with autoregressive inflation lags, impose the AR stationarity condition using the estimated coefficients on inflation lags.

* For `ARX(1,n)`, use the coefficient on `Inflation_lag_1`.
* For `ARX(2,n)`, use the coefficients on `Inflation_lag_1` and `Inflation_lag_2`.
* Conduct the unit-circle check for the implied AR polynomial.
* Reject or flag estimates whose AR roots do not satisfy the stationarity condition.
* Use only the inflation-lag coefficients for this check; do not include SPF or other exogenous-regressor coefficients in the AR stationarity polynomial.

#### BEGE Shape-Process and Variance-Bound Checks

For BEGE shape processes, compute the recursive shape states `p_t` and `n_t` from the estimated recursion. The implied conditional variance at time `t` is:

```text
Var_t = sigma_p^2 * p_t + sigma_n^2 * n_t
```

The collection stage must verify that the implied variance path satisfies the documented BEGE variance constraints.

When variance bounds are used in optimization or post-estimation filtering, construct them as follows:

* Compute `EWMA_t` using decay parameter `lambda = 0.94`.
* Use up to `tau = min(75, T)` observations for the EWMA initialization/window.
* Let `resids` denote the effective-sample residuals used by the BEGE likelihood.
* Define the lower bound:

  ```text
  lower[t] = max(EWMA_t / 1e6, var(resids) / 1e8)
  ```
* Define the upper bound:

  ```text
  upper[t] = min(EWMA_t * 1e6, 1e7 * (1 + max(resids^2)))
  ```
* Additionally, always enforce:

  ```text
  upper[t] >= 1 + max(resids^2)
  ```

The implied BEGE variance must satisfy:

```text
lower[t] <= sigma_p^2 * p_t + sigma_n^2 * n_t <= upper[t]
```

for all effective-sample observations `t`.

When changing BEGE variance-bound logic:

* Keep `sigma_p`, `sigma_n`, shape states, persistence terms, and shock-load terms within their documented bounds.
* Preserve documented BEGE stability restrictions, including persistence-plus-loading restrictions where applicable.
* Do not silently change the EWMA decay parameter, variance scaling constants, or upper/lower bound formulas.
* If the optimizer imposes these constraints directly, the collection stage should still re-check them before reporting final best estimates.

### BEGE Result Aggregation and Reporting

When summarizing BEGE results across random-start jobs:

* Aggregate all valid estimates across the 50 jobs for the same mean process and BEGE specification.
* Rank admissible estimates by maximized log likelihood within each mean process.
* Do not reject estimates solely because the log likelihood is above an investigatory threshold such as `-150`; such thresholds may be recorded as diagnostics for manual review, but they are not admissibility rules unless the user explicitly requests that change.
* In addition to any combined result table kept for compatibility, write cleaned estimation outputs split by mean process under `results/by_mean/`, using stable filenames:
  * `constant.csv`
  * `ARX_1_1.csv`
  * `ARX_2_1.csv`
  * `ARX_2_2.csv`
* In `results/best_model.md`, report the top 20 admissible estimates for each mean process, ranked by maximized log likelihood.
* For every reported top-20 BEGE estimate, present the relevant mean-process equation and BEGE volatility-process equation before any plain parameter table. Substitute the estimated parameter values directly into the equations.
* For substituted equation coefficients, put the standard error directly below the parameter value, for example using a display-math form like `\underset{(0.012345)}{0.123456}`.
* Compute standard errors only at the result-reporting stage for the reported top-20 estimates; do not compute standard errors in server-side raw search jobs unless the user explicitly requests it.
* Report whether the selected estimate passed:

  * optimizer convergence checks,
  * parameter-bound checks,
  * BEGE stability checks,
  * shape upper-cap checks,
  * implied variance-bound checks,
  * and mean-process stationarity checks.
* If the likelihood-best estimate fails an admissibility check, select the best admissible estimate and clearly record why the likelihood-best estimate was rejected or flagged.
* Do not compare likelihoods across models unless they use the same effective sample, residual definition, and initialization conventions required by this file.


### RS-GARCH
- Treat `RS_GARCH/README.md` as the canonical RS specification document.
- Do not introduce new RS-GARCH assumptions in code or docs unless requested.
- If RS-GARCH settings change, update `RS_GARCH/README.md` and implementation together in the same change.

## Conflict Resolution Rule
If settings appear inconsistent across files:
1. Treat the most specific model folder README as authoritative for that model.
2. Use `README.md` for global notation and shared definitions.
3. If conflict remains unresolved, stop and ask the user before changing behavior.

## Change Hygiene
When changing any model setting:
- Update all affected docs and code paths in the same change.
- Briefly state which canonical setting was applied or intentionally changed.
- Avoid silent drift in parameter bounds, sample windows, or initialization conventions.

## Markdown Output For Jupyter Book
- For generated report markdown files intended for Jupyter Book inclusion, always add this block at the very top of the file:
  ```{raw:typst}
  #set page(margin: auto)
  ```
- If a script writes markdown outputs, enforce this preamble in the script so regenerated files stay consistent automatically.
