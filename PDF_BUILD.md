# Building the econometric report

Run `python3 scripts/build_report.py` from the repository root, with `myst`
and `typst` on `PATH`. The command rebuilds `Inflation-Dynamics.pdf` and the
online appendix without running estimation or refreshing the data.

Set `REPORT_DATE=YYYY-MM-DD` when an explicit report date is needed, for
example when the build server's UTC date differs from the reader's local date.

## Report

The builder prepares a report edition under `_build/report-source/` from the
articles listed in `myst.yml`. It removes executable examples, operational
instructions, source-file references, and CSV links, and states the data date
once (from `DataSummary/data_vintage.json`). It omits the effective-sample correction footnote and the GARCH
recursion-initialization paragraph, and its list of construction rules
omits the inflation-shock and unpublished-CPI-month items. The data-decision box on the
interpolated October 2025 CPI keeps the decision and its formula but omits the
*Why* and *Scope* paragraphs of `DataSummary/README.md`. It retains model equations,
estimation restrictions, result tables, figures, and report colour styling.
Canonical source documents and estimation reports retain their implementation
details.

MyST exports Typst source, which the builder lays out before compiling with
`typst`: sections run on without page breaks, only the title page keeps the
template's wide left margin, paragraph spacing is compact, and tables may
continue on the next page. The per-file `#set page(margin: auto)` preamble is
removed from the report edition, because it would start every article on a new
page.
The footer shows the current page number; it omits the template's total-page
counter, which can be stale when Typst reaches its layout iteration limit.

Use this builder for the report PDF; a direct `myst build --pdf` in the
repository root exports the original computational documentation.

The BEGE-EGARCH specification (`BEGE_GARCH/BEGE_EGARCH/SPECIFICATION.md`,
without its script list) is printed inside the BEGE-EGARCH subsection of the
BEGE section rather than as a separate section. The regime-switching section
describes its identification and estimation procedure without citing the BEGE
paper, which is the authors' own work; the canonical
`RegimeSwitching/README.md` keeps the citation.

The regime-switching derivation uses $X_t$, $Z_t$, $Q_t$, and $\mathbb P$:
$Z_t$ generates inflation at $t+1$, $Q_t$ is its probability before that
observation, $R_t$ updates it using that observation, and $S_t$ uses the
full sample. The report retains the partial-switching restrictions, forward
filter, backward smoother, and shared-coefficient estimation derivation.
The generated best-model pages use the same notation. The notation/code
audit is in `RegimeSwitching/notation_audit/report.md`.

GARCH equations and their adjacent constraints are maintained in
`GARCH/report_model_equations.md` and included once with the quarterly results.
The monthly results refer to these shared specifications. The quarterly Estimation subsection also prints
the random-start table (parameter bounds and draw distributions) from the
*Parameter Ranges and Random Starts* section of the quarterly results file; the
monthly Estimation subsection lists its own $\bar v$ values and refers to it. The All Model Results
table of each GARCH model-selection report (every model combination) is not
printed; the section links to that table in the results file on GitHub
(`main` branch).

The abstract states which model has the lowest quarterly AIC and BIC. That
sentence is written by hand; recheck it with `python ModelComparison/compare_models.py`
(project Python environment, writes `ModelComparison/model_ranking_<frequency>.csv`)
whenever collected results change.

Each BEGE best-model summary (quarterly and monthly, every specification) replaces
its CSV list with links to the stored random-search results on GitHub (`main`
branch): `all_estimations.csv` (one estimate per random draw), the per-mean
files in `by_mean/`, `selection_diagnostics.csv`,
`path_quantile_diagnostics.csv`, and `best_loglik_with_se.csv`. The build stops
if any of these files is missing. The links resolve only after the files are
pushed.

## Online appendix

The BEGE density evaluation (`BEGE_GARCH/BEGE_Density/BEGE_Density.md`) is
published as the online appendix `OnlineAppendix/BEGE_Density_Evaluation.md`
instead of a report section. The Density Evaluation subsection of the BEGE
section links to it on GitHub (`main` branch); the link resolves only after the
appendix file is pushed. The file is generated, so do not edit it by hand. It is
GitHub-flavoured Markdown for reading on GitHub, not a Jupyter Book page, so it
deliberately omits the Typst preamble.
