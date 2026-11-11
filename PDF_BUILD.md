# Building the econometric report

Run `python3 scripts/build_report.py` from the repository root, with `myst`
and `typst` on `PATH`. The command rebuilds `Inflation-Dynamics.pdf` without
running estimation or refreshing the data.

Set `REPORT_DATE=YYYY-MM-DD` when an explicit report date is needed, for
example when the build server's UTC date differs from the reader's local date.

The builder prepares a report edition under `_build/report-source/` from the
articles listed in `myst.yml`. It removes executable examples, operational
instructions, source-file references, and CSV links. It retains model
equations, estimation restrictions, result tables, figures, data vintages, and
report colour styling; the BEGE density evaluation is the last section.
Canonical source documents and estimation reports retain their implementation
details.

Use this builder for the report PDF; a direct `myst build --pdf` in the
repository root exports the original computational documentation.

GARCH equations and their adjacent constraints are maintained in
`GARCH/report_model_equations.md` and included once with the quarterly results.
The monthly results refer to these shared specifications.
