# AGENTS.md

This file defines mandatory context and consistency rules for Codex contributions in this repository.

## Scope
These instructions apply to the entire project tree.

## Mandatory Context Loading
Before making or reviewing any model-related change, read these files first:

- `README.md`
- `DataSummary/README.md`
- `MeanProcess/README.md`
- `GARCH/README.md`
- `RegimeSwitching/README.md`
- `BEGE_GARCH/README.md`

If one of these files is empty or incomplete, do not invent missing model settings. Keep existing code behavior unchanged for that component and note the gap.

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

### BEGE Settings and Constraints
From `BEGE_GARCH/README.md`:
- Preserve documented parameter bounds for `sigma_p`, `sigma_n`, shape levels, persistence, and shock-load parameters.
- Preserve documented stability constraints (for example persistence-plus-loading restrictions), variance bound checks, and shape upper-cap checks (`max{p_t, n_t} < 200`).
- Keep the unconditional variance reference consistent with the documented BEGE settings when those checks are used.

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
