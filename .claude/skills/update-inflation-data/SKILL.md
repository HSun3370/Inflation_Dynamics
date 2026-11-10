---
name: update-inflation-data
description: Refresh the aggregate CPI inflation data (quarterly and monthly) to the newest FRED CPIAUCSL and Philadelphia Fed SPF release, check it against the advisor's 2023 reference workbook, then redo the statistical summary. Use when the user asks to update, refresh, or re-pull the inflation / SPF data, or to redo the data summary after new data.
---

# Update the inflation data (routine)

The canonical routine is documented in `DataSummary/README.md` ("Routine Data Update"). All estimation code reads `DataSummary/Aggregate_CPI_inflation_{Quarterly,Monthly}.pkl`, so these must always hold the newest vintage. Never estimate from `Aggregate_CPI_inflation_20230513.xls`; it is the reference for checking the construction and must stay unchanged.

Run everything from the repo root with the pinned env: `PY=~/.conda/envs/inflation_dynamics/bin/python`.

1. **Update the data**: `$PY DataSummary/update_inflation_data.py` (the vintage tag defaults to today's date).
   - If it aborts with "no longer replicate Aggregate_CPI_inflation_20230513.xls", stop. Do not loosen tolerances, move `CPI_REVISION_START`, or edit the reference. Report which series and dates differ (for example a Philly Fed SPF revision or a FRED rebasing) and ask the user how to proceed.
   - If it aborts with an environment mismatch, switch to the pinned env. Pass `--allow-env-mismatch` only if the user asks, because pickles written with other pandas/numpy versions do not load on the Grid.
2. **Review what changed**: run `git diff DataSummary/data_vintage.json` and compare the old and new pickles on their overlap. Expect new periods at the end. Expect revisions to existing CPI values only after the BLS February seasonal-factor revision (the last five years). Call out any other change to historical values and any change in the SPF coverage. Each `DATA DECISION` line printed by the script is an unpublished CPI month filled by geometric interpolation (the user approved this rule on 2026-10-20 for 2025M10). Tell the user about any month beyond 2025M10.
3. **Redo the statistical summary**: `$PY DataSummary/data_summary.py`. It rewrites `inflation_summary_stats_trimmed.md`, `inflation_summary_stats_monthly_trimmed.md`, both figures, and every `% BEGIN auto:` block in `DataSummary/README.md`.
4. **Update hand-written sample references** (`grep -rn` for the previous end dates and observation counts):
   - `AGENTS.md`: the "Current vintage" sentence under *Effective Sample and Initialization*.
   - `README.md`: item 1 (data summary sample windows).
   - `DataSummary/README.md` prose: only if something new appears. The `{important}` data-decision box and the Monthly Data paragraph name 2025M10 explicitly, so extend them, and the data-decision bullet in `AGENTS.md`, if another month is interpolated.
   - Leave model-result docs (MeanProcess, GARCH, RegimeSwitching, BEGE_GARCH) alone. They describe the vintage their results were estimated on. When results are re-estimated, update the "Comparability of Committed Results" paragraph in `DataSummary/README.md` and the vintage bullet in `AGENTS.md`.
5. **Verify**: the pickles load in the pinned env, neither pickle has missing values, and running `data_summary.py` again leaves `README.md` unchanged.
6. **Report** to the user: the vintage tag, the new quarterly and monthly windows and sizes, the reference-check line, any missing months, and notable changes in the summary statistics. Do not commit unless asked.
