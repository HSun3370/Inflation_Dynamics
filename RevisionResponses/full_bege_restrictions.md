```{raw:typst}
#set page(margin: auto)
```

# Full BEGE: assessment of the proposed restrictions

Checked on November 14, 2026, using the current quarterly and monthly Full
BEGE estimates on the 2026-10-20 vintage. Only the four mean processes
included in the current report are considered. This is the pre-estimation
assessment: no new optimization was run for these calculations. The subsequent
restricted searches are tracked in [the run record](restricted_full_run_20261114/manifest.json);
their completed comparison is generated separately.

The proposals impose different restrictions on Full BEGE:

- **Symmetric-p/full-n:** impose $\phi_p^+=\phi_p^-$, while retaining separate
  $\phi_n^+$ and $\phi_n^-$.
- **Positive-only-p/full-n:** impose $\phi_p^-=0$, while retaining separate
  $\phi_n^+$ and $\phi_n^-$.

Neither is the existing Symmetric BEGE model, which shares persistence and
shock-loading parameters between the two shape processes. BadGood imposes
sign symmetry on both shapes. Inflation/Deflation also sets $\phi_n^+=0$.

## Reported Full BEGE estimates

Both displayed likelihood-best Full BEGE models use ARX(2,2). Parentheses
contain the reported standard errors.

| Parameter | Quarterly | Monthly |
|---|---:|---:|
| $\rho_p$ | 0.9436 (0.0122) | 0.8587 (0.0144) |
| $\phi_p^+$ | 0.0081 (0.0490) | 0.0908 (0.0175) |
| $\phi_p^-$ | 0.0402 (0.0143) | 0.0000 (0.0205†) |
| $\phi_n^+$ | 1.0023 (0.0014) | 0.4771 (0.0544) |
| $\phi_n^-$ | 0.2684 (0.0018) | 0.2729 (0.0494) |

† A rough standard error at the nonnegative parameter boundary, not a
regular normal-approximation test of a zero coefficient.

**Quarterly:** the small good-shape loadings still motivate investigating
parsimony. They do not imply constant or irrelevant good-shape dynamics:
persistence is high, and the fitted $p_t$ 5th and 95th percentiles are
0.9649 and 2.4952. The negative-shock loading exceeds the positive one in
all three quarterly ARX Full BEGE fits. Thus the point estimates do not
specifically favor removing the negative-shock loading. Symmetry remains
a restriction to assess with a joint covariance calculation or restricted
re-estimation, rather than by comparing separate significance statements.

The archived higher-likelihood quarterly candidate reproduces this pattern:
$\rho_p=0.9441$, $\phi_p^+=0.0072$, and $\phi_p^-=0.0406$. Its likelihood
is −168.8770 versus −168.9699 for the current reported fit. Search
reconciliation remains necessary, but that candidate does not reverse the
qualitative conclusion. Small loadings alone do not establish a local-optimum
failure.

**Monthly:** the positive-only proposal is closely reflected in the saved
ARX estimates. The $p$ negative-shock coefficient is approximately zero in
ARX(1,1) and ARX(2,2), and 0.0019 in ARX(2,1); the corresponding positive
coefficients are 0.1052, 0.0908, and 0.1507. Both $n$ loadings are positive
in all three. The anchored Constant model does not share this pattern, so
the conclusion is specific to the ARX fits.

## Direct likelihood check without re-estimation

I recomputed the likelihood using the current monthly sample and evaluator,
then set only $\phi_p^-$ to exactly zero and recomputed both shape paths.
The likelihood changes by less than $3\times10^{-9}$ for ARX(1,1) and is
unchanged for ARX(2,2). The resulting parameter vector is feasible for the
positive-only-p/full-n restriction.

| Mean | LogLik at fixed zero | Original AIC | AIC counting one fewer parameter | Original BIC | BIC counting one fewer parameter |
|---|---:|---:|---:|---:|---:|
| ARX(1,1) | 106.7472 | -187.4944 | -189.4944 | -128.4798 | -135.0193 |
| ARX(2,2) | 108.0682 | -186.1364 | -188.1364 | -118.0426 | -124.5822 |

These are criterion calculations at feasible restricted parameter vectors,
not the results of a dedicated restricted-model search. Holding the likelihood
fixed, fixing one parameter lowers AIC by 2 and BIC by $\ln(692)=6.5396$.
The ARX(1,1) calculation is below the current best BEGE AIC, −188.4515 for
Symmetric BEGE with ARX(1,1), although its BIC remains above that model's
−143.0557. A restricted-model search and appropriate inference would be
needed for final reporting.

The earlier ranking of BadGood as AIC-best is no longer the current quarterly
ranking: the best Full BEGE AIC is 367.9398 versus 369.4462 for BadGood, and
BEGE-EGARCH has AIC 360.6832. Constant-p Full BEGE with ARX(1,1) has AIC
367.1311, so there is still evidence in favor of investigating simpler
good-shape dynamics.

Sources: [quarterly Full BEGE](../BEGE_GARCH/Full_BEGE/README.md),
[monthly Full BEGE](../BEGE_GARCH/Full_BEGE/README_monthly.md),
[quarterly model ranking](../ModelComparison/model_ranking_quarterly.csv),
[monthly model ranking](../ModelComparison/model_ranking_monthly.csv),
[archived candidate check](archived_full_candidate_check.csv), and
[fixed-zero likelihood calculations](full_bege_positive_only_check.csv).
