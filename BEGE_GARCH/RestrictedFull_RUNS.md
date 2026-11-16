```{raw:typst}
#set page(margin: auto)
```

# Restricted Full BEGE Runs (2026-11-14)

The requested symmetric-p/full-n (`SymmetricP_BEGE`) and positive-only-p/full-n
(`PositiveOnlyP_BEGE`) extensions each use 10 independent seed jobs at each
frequency, 40 draws per job and 25 fresh joint starts per draw. Each mean process
therefore receives 10,000 starts. Search jobs omit standard errors; collection
computes them only for the selected estimate. The six-mean research menu is
retained, while the PDF reports only Constant, ARX(1,1), ARX(2,1), and ARX(2,2).

Submit both frequencies with:

```bash
SPECS="SymmetricP_BEGE PositiveOnlyP_BEGE" bash BEGE_GARCH/submit_bege_grid.sh
```

These extensions are selected explicitly in the submission command; the default
submission menu of the existing specifications is unchanged. Collectors verify
that all requested seeds, draws and means are present before publishing results.

## Submitted run

The submission record and source/data hashes are in
[`manifest.json`](../RevisionResponses/restricted_full_run_20261114/manifest.json).
Forty estimation jobs and four held collectors were submitted; job `8990988`
waits for the four collectors and the worker-node reporting check. It validates
all 2,400 saved draws per specification/frequency, checks the reported likelihoods
and standard errors, writes the within-mean Full BEGE comparison, refreshes the
model rankings and revision response, and rebuilds `Inflation-Dynamics.pdf`.
A missing or incomplete search stops report publication.

Each estimation job requests 25 CPU slots and 63 GB total memory, allowing its
25 starts to run concurrently; the number of seeds, draws, and starts is unchanged.
Search jobs compute no parameter standard errors. The selected report estimate
is the likelihood-best admissible fit among the four report means; all six means
remain in the research CSVs.

Completion is recorded in
`RevisionResponses/restricted_full_run_20261114/completion.json`, including the
final PDF hash. If reporting exits unsuccessfully, its exit code is written to
`report_job_status.json` in that directory and the scheduler log gives the cause.
The report can then be regenerated with `python scripts/refine_restricted_full_report.py`
using the report tool paths in the manifest.

The PDF excludes both these specifications and their results. Their equations and additional bounds are retained in [RestrictedFull_SPECIFICATION.md](RestrictedFull_SPECIFICATION.md) for research reproducibility.
