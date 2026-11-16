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
- Always use the newest data vintage: the trimmed effective samples in `DataSummary/Aggregate_CPI_inflation_Quarterly.pkl` and `DataSummary/Aggregate_CPI_inflation_Monthly.pkl`, whose windows and sizes are recorded in `DataSummary/data_vintage.json` and the Current Vintage block of `DataSummary/README.md`. Current vintage (2026-10-20): quarterly `1969Q2` to `2026Q3` (`230` observations); monthly `1969M2` to `2026M9` (`692` months, no missing values). Never estimate from the 2023 reference workbook or an older workbook.
- Data decision (approved by the user on 2026-10-20): CPIAUCSL months that BLS did not publish inside the sample are filled by geometric (log-linear) interpolation between the nearest published months. Currently this applies only to `2025M10` (no October 2025 CPI because of the federal shutdown), so monthly inflation in `2025M10`--`2025M11` is constructed rather than observed; the quarterly series is unaffected. Do not change this convention or drop these months without the user's approval, and keep the data-decision box in `DataSummary/README.md` in sync.
- Refresh the data only through the routine in `DataSummary/README.md` (`update_inflation_data.py`, then `data_summary.py`), which verifies the construction against the reference workbook `DataSummary/Aggregate_CPI_inflation_20230513.xls`. Keep that reference workbook unchanged.
- Committed results and vintages: all committed estimation results (OLS mean processes, GARCH family, regime switching, BEGE; quarterly and monthly) are estimated on the 2026-10-20 vintage. The exception is the BEGE density-evaluator accuracy check in `BEGE_GARCH/BEGE_Density/`, which uses ARX(1,1) residuals from `1969Q2`--`2022Q4` (`215` observations, 2023 vintage). Likelihoods are not comparable across vintages.
- Use the same estimation sample across mean and volatility specifications unless the task explicitly studies sample sensitivity.
- For all estimation exercises, treat the effective sample size at a given frequency and data vintage as fixed and comparable across models.
- If lagged regressors (for example `Inflation_lag_1`, `Inflation_lag_2`, `SPF_lag_1`) are already precomputed in the effective-sample dataset, use them directly rather than trimming observations again.
- For GARCH-type pre-sample recursion terms, initialize missing pre-sample variance states with the model-implied unconditional variance where applicable.

### Mean Process Menu
From `MeanProcess/README.md`, keep mean-model options aligned with:
- Constant
- ARX(1,0)
- ARX(1,1)
- ARX(2,0)
- ARX(2,1)
- ARX(2,2)

ARX(1,0) and ARX(2,0) (pure AR, no SPF terms) were added by the user on 2026-11-10 for all volatility models (GARCH family and BEGE); they use the precomputed inflation-lag columns. Regime-switching keeps its own menu (see `RegimeSwitching/README.md`). In the GARCH script, model ids and seeds follow `MEAN_ADDITION_ROUNDS`, so the earlier 144 combinations keep their ids `M001`--`M144` and the new means are `M145`--`M216`.

For BEGE runs, default to skipping ARX(2,2) unless explicitly requested, matching documented practice.

### GARCH Family Settings
From `GARCH/README.md`:
- Supported volatility families: GARCH, GJR-GARCH, EGARCH.
- Supported innovation distributions: Normal, standardized Student's t, and two-component Gaussian mixture.
- Enforce stationarity/positivity constraints appropriate to each model form.
- Keep EGARCH centering convention consistent with project documentation (fixed `sqrt(2/pi)` convention unless explicitly changed project-wide).
- Random starts (see `GARCH/README.md`): 5,000 starts per model combination (start 1 is the `arch` default), one CBS Grid job per combination via `GARCH/submit_garch_grid.sh`. Every random start is drawn inside `arch`'s bounds/constraints and the stationary region; do not narrow the documented draw ranges without the user's approval.
- Admissibility of a reported GARCH-family estimate: SLSQP success and `arch` convergence flag 0; first-order (KKT) residual at most `0.005*T` (after up to five SLSQP restarts from the candidate's own solution); volatility stationarity with margin `1e-6`; inflation-lag AR roots inside the unit circle; and, for the Gaussian mixture, component-variance ratio `min/max >= 0.01` (user-approved 2026-10-20). If none passes, report the best converged first-order-valid fit and flag it in `selection_status`.
- For GJR-GARCH, report both the `arch` indicator coefficient (`gamma_i - alpha_i`) and the README negative-shock coefficient `gamma_i = alpha_i + (gamma_i - alpha_i)`, whose standard error uses the delta method on the robust covariance including the covariance term.

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
* For BEGE runs, continue to run all six mean processes unless explicitly requested, consistent with documented practice.
* When submitting jobs in server, don't compute SE of paramters. When reporting the best results, compute SE for the best models and report them in md file. 
User-requested extensions (2026-11-14): `SymmetricP_BEGE` imposes
`phi_p_plus = phi_p_minus = phi_p`; `PositiveOnlyP_BEGE` imposes `phi_p_minus = 0`.
Both retain the full n recursion, use nine free volatility parameters, and follow
the same bounds, initialization, 10-job/40-draw/25-start protocol, collection checks,
and quarterly/monthly samples as Full BEGE. Their six-mean research searches retain
pure-AR results; the PDF excludes those two means as previously requested.

### BEGE Random-Start Estimation Protocol

BEGE likelihood maximization uses randomized initial starts. For server-side production runs, the canonical search protocol is:

* Submit `10` independent jobs for each BEGE volatility specification and frequency (reduced from `50` by the user on 2026-11-10 to save computation).
* Within each job, use:

  ```python
  parser.add_argument("--n-draws", type=int, default=40, help="Number of random draws per mean specification")
  parser.add_argument("--n-starts", type=int, default=25, help="MLE restarts per draw")
  ```
* Therefore, for each mean-model and BEGE-volatility-specification pair, the intended production search covers:

  ```text
  10 jobs × 40 draws per job × 25 starts per draw = 10,000 starting points
  ```

When modifying estimation scripts:

* Preserve this job/draw/start interpretation unless the user explicitly requests a change.
* Do not reinterpret `--n-draws` or `--n-starts` in a way that changes the total number of candidate starting points.
* Ensure job-level random seeds or job identifiers generate independent randomized starts across the 10 jobs.
* Result aggregation must treat the 10 jobs as one combined search for the same model, mean process, and BEGE specification.
* Mean-parameter starting values are drawn uniformly from OLS-centered intervals `theta_hat ± 2 SE` computed on the estimation sample at run time (HAC(12) standard errors, as in `MeanProcess/`); do not hard-code them.
* BEGE is estimated at both frequencies. `FREQUENCY=quarterly|monthly` in the submit scripts (exported to Python as `BEGE_FREQUENCY`) selects the effective-sample pickle, writes raw output to a scratch folder named by frequency and data vintage, and writes collected results to `results/` and `README.md` (quarterly) or `results_monthly/` and `README_monthly.md` (monthly). `BEGE_GARCH/submit_bege_grid.sh` submits every specification at both frequencies.

### BEGE Best-Model Collection Checks

When collecting, ranking, or summarizing best BEGE estimates, do not rely only on optimizer convergence or likelihood value. The best-model collection stage must also check the relevant admissibility conditions below.

#### Mean-Process Stationarity

For any estimated mean process with autoregressive inflation lags, impose the AR stationarity condition using the estimated coefficients on inflation lags.

* For `ARX(1,n)`, use the coefficient on `Inflation_lag_1`.
* For `ARX(2,n)`, use the coefficients on `Inflation_lag_1` and `Inflation_lag_2`.
* Conduct the unit-circle check for the implied AR polynomial.
* Reject or flag estimates whose AR roots do not satisfy the stationarity condition.
* Use only the inflation-lag coefficients for this check; do not include SPF or other exogenous-regressor coefficients in the AR stationarity polynomial.

#### BEGE Shape-Process Checks

For BEGE shape processes, compute the recursive shape states `p_t` and `n_t` from the estimated recursion. The implied conditional variance at time `t` is:

```text
Var_t = sigma_p^2 * p_t + sigma_n^2 * n_t
```

The collection stage must verify that the shape states and the implied variance are finite and positive for all effective-sample observations `t`.

Do not impose an EWMA implied-variance bound on `Var_t`, either in optimization or in post-estimation filtering. The user dropped it on 2026-11-10. A paired test with the full 50,000 starts on the 2026-10-20 vintage showed that it never binds at a converged estimate. The parameter bounds and stability restrictions below already keep `Var_t` finite. The estimators keep `enforce_variance_bounds` only as an off-by-default option for replicating that test, and the collectors no longer check the bound.

When changing BEGE shape-process logic:

* Keep `sigma_p`, `sigma_n`, shape states, persistence terms, and shock-load terms within their documented bounds.
* Preserve documented BEGE stability restrictions, including persistence-plus-loading restrictions where applicable.
* If the optimizer imposes these constraints directly, the collection stage should still re-check them before reporting final best estimates.

### BEGE Result Aggregation and Reporting

When summarizing BEGE results across random-start jobs:

* Aggregate all valid estimates across the 10 jobs for the same mean process and BEGE specification.
* Rank admissible estimates by maximized log likelihood within each mean process.
* Write cleaned estimation outputs split by mean process under `results/`, using stable filenames:
  * `constant.csv`
  * `ARX_1_0.csv`
  * `ARX_1_1.csv`
  * `ARX_2_0.csv`
  * `ARX_2_1.csv`
  * `ARX_2_2.csv`
* In `results/best_model.md`, report only the single best admissible estimate for each BEGE specification, ranked by maximized log likelihood.
* Compute standard errors only at the result-reporting stage for the reported best estimate; do not compute standard errors in server-side raw search jobs unless the user explicitly requests it.
* In generated `results/best_model.md` markdown, format reported numeric estimates, standard errors, likelihood criteria, and diagnostic table values with exactly four digits after the decimal point.
* In every BEGE best-model summary table, report the number of observations `N` and the sample period (first--last period of the effective sample, e.g. `1969Q2--2026Q3`) next to LogLik, AIC, and BIC. This replaces the `sigma_p`, `sigma_n`, and `sqrt(Var(u))` summary columns used earlier on 2026-11-11 (user-requested 2026-11-11); `sigma_p` and `sigma_n` remain in the parameter table.
* Report whether the selected estimate passed:

  * optimizer convergence checks,
  * parameter-bound checks,
  * BEGE stability checks,
  * and mean-process stationarity checks.
* If the likelihood-best estimate fails an admissibility check, select the best admissible estimate and clearly record why the likelihood-best estimate was rejected or flagged.
* Do not compare likelihoods across models unless they use the same effective sample, residual definition, and initialization conventions required by this file.


### Regime Switching
- Treat `RegimeSwitching/README.md` as the canonical regime-switching specification document.
- Gaussian EM initialization jointly updates shared and switching mean coefficients by weighted least squares with weights equal to smoothed regime probabilities divided by the preceding innovation variances (corrected 2026-11-12 after comparing the QuantMFR implementation). Preserve coefficient-sharing restrictions, use no ridge penalty, and retain the full joint BFGS/L-BFGS optimization afterward. The optional Student-t initializer is unchanged.
- Random starts: 5,000 starts per Gaussian specification, including start 1, drawn uniformly from full-sample OLS coefficient estimates ± 2 HAC(12) standard errors, with independent draws for switching coefficients and a single draw for each shared coefficient. Draw each sigma from the positive part of its full-sample residual RMS ± 2 delta-method HAC(12) standard errors, then square it for the variance parameter. Draw each transition row independently from Dirichlet(1,...,1), uniform on the simplex (user-approved 2026-11-14). The optional Student-t starts are unchanged. Searches are run on the CBS Grid via `RegimeSwitching/submit_regime_switching_grid.sh` as blocks of 1,000 starts, one job per block. Each start keeps its own seed, so the selected estimate does not depend on the block size. Each start runs up to 1,000 EM iterations before BFGS/L-BFGS (raised from 50 by the user on 2026-11-11); raw blocks from searches with different settings are not merged.
- Identification and optimization follow Bekaert, Engstrom and Ermolov (2015), Section 4.1.2 and Appendix B: initialize regime probabilities at their unconditional distribution, label selected estimates and reports by increasing innovation variance, and perturb/refit the selected provisional optimum after combining the search blocks. Permute switching mean parameters, both transition-matrix axes, and regime-probability paths together.
- Final-solution verification uses 20 additional perturbation refits per batch in unrestricted coordinates (scale `0.05 * max(1, abs(parameter))`), with an absolute likelihood return tolerance of `1e-4` and at most three batches. A higher admissible maximum becomes the new center and requires a fresh verification batch. Report only estimates for which every refit in the final batch converges, passes the model checks, and returns within tolerance. Record all refits separately from the 5,000 initial starts. These numerical choices are this project's implementation; the paper does not specify them.
- User-approved 2026-11-10: remove the extra transition cutoff `(1e-4, 1-1e-4)` and the sample-scaled variance floor. Require finite positive innovation variances and valid stochastic transition probabilities, and retain the inflation application's regime-specific AR stationarity check. User decision 2026-11-12: keep the variance floor removed even though the Gaussian likelihood is then unbounded (degenerate optima with a regime variance near zero). Final-solution verification rejects such optima, and a selected optimum whose saved parameters do not reproduce its likelihood is recorded as `likelihood_not_reproducible`. A specification without a verified estimate is retained in diagnostic CSVs and omitted from the results tables (user-requested 2026-11-14); it is not replaced by a lower optimum. Saved searches selected using the former cutoffs must be identified as such until a fresh search is run under the updated rules.
- Do not introduce new regime-switching assumptions in code or docs unless requested.
- If regime-switching settings change, update `RegimeSwitching/README.md` and implementation together in the same change.

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
