# Building the econometric report

Run `python3 scripts/build_report.py` from the repository root, with `myst`
and `typst` on `PATH`. The command rebuilds `Inflation-Dynamics.pdf` and the
online appendix without running estimation or refreshing the data.

Set `REPORT_DATE=YYYY-MM-DD` when an explicit report date is needed, for
example when the build server's UTC date differs from the reader's local date.

## Report

The builder prepares a report edition under `_build/report-source/` from the
articles listed in `myst.yml`. It removes executable examples, operational
instructions, source-file references, and operational CSV lists, and states the data date
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
template's wide left margin, notation starts the normal-width body, and paragraph spacing is compact. Native Typst tables, adapted by `scripts/report_tables.typ`, replace the legacy tablex renderer so headers repeat only on continuation pages. Short tables stay together; headings and table lead-ins stay with the following content. The per-file `#set page(margin: auto)` preamble is
removed from the report edition, because it would start every article on a new
page.
Tables use thin horizontal and vertical rules, a light-gray header background,
and stronger rules above and below the header. A body row whose only content is
its first cell (the group labels of the GARCH and BEGE parameter-range tables)
spans the full table width. The footer shows the current page number.

Use this builder for the report PDF; a direct `myst build --pdf` in the
repository root exports the original computational documentation.

The PDF presents four mean processes: Constant, ARX(1,1), ARX(2,1), and
ARX(2,2). At the user's request on 2026-11-14, the pure autoregressions
ARX(1,0) and ARX(2,0) were archived from the report. Their prior discussion
and results are preserved in `Archive/PureAR_20261114/`, and estimation code
and saved results remain intact. `scripts/report_scope.py` removes these
models from the report text and adjusts the displayed counts. The GARCH
online tables, `OnlineAppendix/GARCH_Results_Quarterly.md` and
`OnlineAppendix/GARCH_Results_Monthly.md`, contain only the 144 combinations
covered by the report. BEGE count summaries use the four included means'
saved diagnostics, and their links list only those four mean processes.

The advisor's revision responses are maintained in
`RevisionResponses/responses.json`. `scripts/report_revision_notes.py` inserts
one response footnote for each of Q1--Q11 and B1--B5 beside the relevant report
passage, after the presentation edits, and regenerates
`RevisionResponses/README.md` as the completion checklist. The existing EWMA
footnote is consolidated into response Q9. These notes survive regeneration of
the estimation reports; they do not change estimates. The builder stops if a
response's insertion point changes. Recheck the dated numerical evidence when
new estimates are collected.

The BEGE-EGARCH specification (`BEGE_GARCH/BEGE_EGARCH/SPECIFICATION.md`,
without its script list) is printed inside the BEGE-EGARCH subsection of the
BEGE section rather than as a separate section. Of its *Parameters and
Stability* subsection, printed as *Persistence Restriction*, the report keeps
only the persistence restriction; the parameter-order display and the pointer to
the BEGE parameter table (which follows in the same section) are omitted. The regime-switching section
describes its identification and estimation procedure without citing the BEGE
paper, which is the authors' own work; the canonical
`RegimeSwitching/README.md` keeps the citation.

The regime-switching derivation starts with two-regime ARX(2,2), switching the
intercept, inflation-lag coefficients, and variance while sharing SPF coefficients.
$Z_t$ generates inflation at $t+1$, $Q_t$ is its predicted probability before that
observation, $Q_{t\mid t+1}$ is the updated probability, and $\widetilde{Q}_t$ uses
the full sample. General filtering and smoothing arguments precede the ARX
illustrations. The EM section explains the shared-coefficient weighted regression
and the subsequent joint MLE. The PDF omits the implementation audit and all
regime-switching moment sections, tables, and plots; the source notes retain them.

Presentation-only transformations in `scripts/report_editorial.py` shorten repeated
result introductions, reshape wide GARCH comparison panels into one row per
distribution and family (columns Distribution, Family, Order, Mean, criterion, so
that Family and Order together name the volatility specification), and wrap long fitted
equations. The BEGE path tables label the third and fourth conditional cumulants
$\kappa_{3,t}$ and $\kappa_{4,t}$; these are not standardized skewness and kurtosis.
The OLS criteria retain their saved parameter-count convention, with the adjustment
for cross-model comparison stated beside the OLS results.

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
branch): `all_estimations.csv` (every saved result, the best of 25 optimizations), the per-mean
files in `by_mean/`, `selection_diagnostics.csv`,
`path_quantile_diagnostics.csv`, and `best_loglik_with_se.csv`. The build stops
if any of these files is missing. The links resolve only after the files are
pushed. The BEGE *Estimation and Random Starts* section states only the search
size (10,000 starts, saving the best of every 25 optimizations); the job and
draw layout stays in `BEGE_GARCH/README.md`. Its parameter table omits the
log-shape cap row and the notes below the table (the lead-in defines the
mean-start symbols instead), and the Standard Errors subsection describes the
rough standard errors in one sentence; the README keeps all of these details.

## Online appendix

The BEGE density evaluation (`BEGE_GARCH/BEGE_Density/BEGE_Density.md`) is
published as the online appendix `OnlineAppendix/BEGE_Density_Evaluation.md`
instead of a report section. The Density Evaluation subsection of the BEGE
section links to it on GitHub (`main` branch); the link resolves only after the
appendix file is pushed. The file is generated, so do not edit it by hand. It is
GitHub-flavoured Markdown for reading on GitHub, not a Jupyter Book page, so it
deliberately omits the Typst preamble.

The symmetric-p/full-n and positive-only-p/full-n result pages (both frequencies)
and their comparison were withdrawn from the PDF at the user's request on
2026-11-15. Their methodology and overview references are also excluded. The saved results and preceding PDF
are preserved under `Archive/RestrictedFull_results_withdrawn_20261115/`.
The export article list excludes these result pages. The restricted-results
refinement script stops while its `report_withdrawn.json` marker exists, to
prevent inadvertently republishing the withdrawn results.
