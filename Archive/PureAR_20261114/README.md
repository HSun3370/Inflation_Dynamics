```{raw:typst}
#set page(margin: auto)
```

# Archived pure autoregressive mean models

On November 14, 2026, ARX(1,0) and ARX(2,0) were removed from the report
at the user's request. The advisor's revision letter did not request these
models. Its BIC suggestion was to exclude the SPF-only anchored mean from a
comparison, while retaining the models with estimated SPF coefficients.

The [previous PDF](Inflation-Dynamics-with-pure-AR.pdf) preserves the full
discussion and results before this change. Source snapshots are retained in
`sources/`, including both OLS result sections, both GARCH reports, the GARCH
equation appendix, the BEGE methodology, and the report introduction.

The current PDF presents four means: Constant, ARX(1,1), ARX(2,1), and
ARX(2,2). It omits the archived models' definitions, equations, OLS estimates,
starting-scale values, and result links. Its GARCH online tables contain
144 combinations per frequency, and its BEGE summaries count only the four
reported means: 1,600 saved draw winners from 40,000 optimizer starts per
volatility specification and frequency.

Original estimation code and saved results remain in their model folders.
In particular, each BEGE specification retains the `ARX_1_0.csv` and
`ARX_2_0.csv` files under `results/by_mean/` and
`results_monthly/by_mean/`; GARCH retains the corresponding rows in its
estimation CSVs. No models were re-estimated or deleted. Data and result
checksums before this report edit are recorded in `estimate_checksums.json`.

The report exclusion is enforced by `scripts/report_scope.py` on every
build. It does not change the estimation menu used by the research scripts.
