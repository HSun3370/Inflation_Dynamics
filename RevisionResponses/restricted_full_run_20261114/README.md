```{raw:typst}
#set page(margin: auto)
```

# Restricted Full BEGE estimation run

Submitted on November 14, 2026: symmetric-p/full-n and positive-only-p/full-n,
each at quarterly and monthly frequency. The data vintage is 2026-10-20.
Each specification/frequency uses 10 seed jobs × 40 draws × 25 starts, or
10,000 starts per mean. The research search covers all six existing means;
the PDF retains the four SPF-based means. Standard errors are computed only
when reporting the selected estimates.

The [manifest](manifest.json) records all job identifiers, source/data hashes,
and the source snapshot. [Submission output](submission.log) records the
40 search jobs and four held collectors. Reporting job **8990988** waits for
the collectors, validates completeness and inference, compares the restricted
fits with Full BEGE, updates the revision response and rankings, and rebuilds
`Inflation-Dynamics.pdf`. The current PDF contains the specifications and
marks their numerical results as pending.

Six automated tests passed, covering: independent shape recursion calculations,
nesting within Full BEGE, stable initial draws, agreement between estimator
and collector at both frequencies for all means, the monthly fixed-zero
benchmark, and report rendering/completeness safeguards. A worker-node check
also computed the 14 free-parameter standard errors for the feasible monthly
positive-only ARX(2,2) benchmark and rebuilt the PDF successfully. This benchmark
check is not a replacement for the new production search.

On completion, `completion.json` records the validated selected estimates and
PDF hash, and `comparison.csv` records the comparisons for the four report means.
`report_job_status.json` records the reporting job exit code. A nonzero exit or
absence of `completion.json` must not be interpreted as a completed report.
The timestamped `progress_at_handoff.json` is only a progress snapshot, not a
live status file.
