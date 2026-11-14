```{raw:typst}
#set page(margin: auto)
```

# Regime-switching estimation audit and re-estimation

Audit started 2026-11-12; completed 2026-11-13. Both frequencies use the 2026-10-20 data vintage.

## What the comparison establishes

The original inflation likelihood and filtering implementation are correct at the saved parameter vectors. The independent log-domain Hamilton filter and the original QuantMFR filter reproduce them to numerical precision. The correction made here concerns the EM initialization of models with shared mean coefficients. Final estimation still maximizes the same observed-data likelihood. Neither multiple starts nor local verification proves that a finite global maximum exists.

The comparison uses the authenticated original repository, branch `QuantMFR`, commit `f207800b7927f175839ebb74bc6c9187741afc09`. The checkout is at `/user/hs3370/TenuousBeliefs`. The relevant sources are [MLE.py](https://github.com/lphansen/TenuousBeliefs/blob/f207800b7927f175839ebb74bc6c9187741afc09/MLE.py#L1428), [EM_MSVAR.ipynb](https://github.com/lphansen/TenuousBeliefs/blob/f207800b7927f175839ebb74bc6c9187741afc09/EM_MSVAR.ipynb), and [MS_VAR.ipynb](https://github.com/lphansen/TenuousBeliefs/blob/f207800b7927f175839ebb74bc6c9187741afc09/MS_VAR.ipynb). Their access permissions also apply to these links.

## Method comparison

| Component | QuantMFR | Inflation implementation |
| --- | --- | --- |
| Model | Multivariate VAR; all regression coefficients and covariance matrices switch | Univariate Gaussian ARX; preserves shared AR/SPF blocks and anchored Constant mean |
| Initial state probabilities | Stationary distribution of P (power iteration) | Stationary distribution of P at each likelihood evaluation |
| Likelihood | Hamilton prediction/update with Gaussian densities | Same recursion; numerically stable log-domain evaluation |
| EM regression step | Separate probability-weighted regressions; ridge=1e-10 | Joint probability/variance-weighted regression with exact sharing restrictions; no ridge |
| Optimization | EM-only random search; EM notebook uses 200 starts, 500 iterations, tolerance 1e-6 | 5,000 starts; up to 1,000 EM iterations then joint BFGS/L-BFGS, up to 1,500 iterations |
| Selection | Largest finite EM likelihood; no returned convergence flag or final stationarity rejection | Optimizer success and existing parameter/AR checks, followed by 20 perturbation refits |
| Identification | EM labels arbitrary; Gibbs plots optionally align by stationary probabilities | All selected parameters, transitions, and probability paths ordered by innovation variance |
| Gibbs sampler | Posterior state/parameter draws; a different inferential procedure | Not substituted for the requested Bekaert-style likelihood estimation |

QuantMFR is a comparison implementation, not a correctness certificate. Its EM transition-count update omits the dependence of the initial stationary distribution on P, and its search can retain an iteration-limit result without checking convergence. The inflation estimator uses this type of transition update only for initialization; the final numerical optimization evaluates the full stationary-initialized likelihood. The source also exponentiates Gaussian log densities directly, which can underflow for extreme candidate parameters. No such arithmetic is copied into production.

## Correction to the previous initializer

In statsmodels 0.14.2, `_em_exog` first estimates non-switching coefficients by pooled OLS, subtracts their fitted contribution, then estimates switching coefficients separately. When coefficients are shared and innovation variances differ, this generally does not maximize the Gaussian mean-block EM objective. The corrected update solves

$$
\min_{\{\beta_k\}}\sum_{t=1}^T\sum_{k=1}^K \frac{\gamma_{k,t}}{\sigma_{k,\mathrm{old}}^2}(y_t-x_t^\prime\beta_k)^2,\qquad \gamma_{k,t}=\Pr(s_t=k\mid Y_T;\theta_{\mathrm{old}}),
$$

subject to the model's equality restrictions on shared coefficients. With all coefficients switching, the constant variance weight cancels inside each separate regression, recovering the usual regime-specific weighted least-squares update. The subsequent variance update uses weighted squared residuals. The anchored Constant model has no mean coefficients to update. Tests check the weighted first-order conditions, coefficient equality restrictions, and integration with the actual smoother.

The correction changes the starting point supplied to the final optimizer, not the likelihood or economic model. Thus a different selected result reflects the search path or verification outcome. It is not evidence that the previous likelihood formula was wrong. The optional Student-t initializer is unchanged.

## Audit of the previous search

All 34 specifications have exactly 5,000 distinct initial attempts, covering attempt numbers 1–5,000. Each frequency uses the full effective sample without additional lag trimming: quarterly 1969Q2–2026Q3 (230 observations), monthly 1969M2–2026M9 (692 observations). The monthly interpolation convention remains unchanged. The stored anchored residual is exactly inflation minus SPF.

For the source comparison, only QuantMFR's data-preparation function is replaced with the already constructed inflation response and design matrix. Its Gaussian density, forward filter, backward smoother, and likelihood functions execute unchanged. Transition subtraction roundoff below zero is removed at machine precision before passing the matrix to the reference code; no economic cutoff is imposed.

| Frequency | Models | Max independent vs statsmodels LogLik gap | Max QuantMFR vs statsmodels LogLik gap | Max filtered probability gap | Max smoothed probability gap |
| --- | --- | --- | --- | --- | --- |
| Monthly | 17 | 8.53e-14 | 5.91e-12 | 3.25e-12 | 3.11e-12 |
| Quarterly | 17 | 1.99e-13 | 2.90e-12 | 1.65e-12 | 2.10e-12 |

Two rejected quarterly fits have saved likelihoods that are not reproduced by their saved parameters. Their extremely small variances make the likelihood sensitive to floating-point changes in fitted means. The independently recomputed likelihoods agree with statsmodels at those saved parameter vectors as well. These are numerical degeneracies, not evidence for a different Hamilton likelihood.

## Fresh estimation results

The corrected search uses methodology `bekaert-identification-stability-v3-joint-em`: 170 Grid blocks × 1,000 starts = 170,000 starts across 34 specifications. Each attempt keeps the previous deterministic seed and starting-draw rule. Only the selected provisional optimum of each specification is verified after collection; every refit in its final batch must converge, pass the model checks, and return within 0.0001 log-likelihood units. Verification refits are additional to the 5,000 initial starts. A failed selected optimum is not replaced by a lower optimum.

| Frequency | Previous verified | New verified | New unverified | Additional verification refits |
| --- | --- | --- | --- | --- |
| Quarterly | 12 | 13 | 4 | 380 |
| Monthly | 12 | 11 | 6 | 360 |

| Frequency | Criterion | Mean | K | Sw.AR | Sw.SPF | N | Sample | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quarterly | AIC | ARX(1,1) | 2 | Y | N | 230 | 1969Q2–2026Q3 | -179.8177 | 377.6354 | 408.5781 |
| Quarterly | BIC | ARX(1,1) | 2 | N | N | 230 | 1969Q2–2026Q3 | -182.9057 | 379.8114 | 403.8779 |
| Monthly | AIC | ARX(1,1) | 3 | N | N | 692 | 1969-02–2026-09 | 100.3983 | -176.7965 | -122.3215 |
| Monthly | BIC | ARX(1,1) | 2 | N | N | 692 | 1969-02–2026-09 | 89.9563 | -165.9125 | -134.1354 |

AIC and BIC select among verified fits only and can favor different specifications. Likelihoods are compared within frequency; quarterly and monthly likelihoods are not ranked against each other. The full parameter tables, transition matrices, conditional-variance quantiles, and regime plots are in the [quarterly best-model report](../results/regime_switching_best_model.md) and [monthly best-model report](../results_monthly/regime_switching_best_model.md).

The former quarterly AIC winner, three-regime ARX(1,1) with switching AR coefficients, had a verified local log likelihood of −167.3355. The new search selected a higher candidate at −158.3755, with minimum variance 1.47e-7, which failed verification. Similarly, the former monthly AIC winner, three-regime ARX(2,1) with switching AR coefficients, moved from a verified local log likelihood of 116.7792 to an unverified candidate at 134.3093 (minimum variance 3.47e-7). Under the requested rule against falling back to a lower optimum, these specifications are excluded from the new ranking. This explains the change in AIC winners; it does not invalidate the older fits as local optima.

All 34 newly saved likelihoods reproduce exactly from their saved parameter values. The updated independent and QuantMFR comparisons are:

| Frequency | Max independent LogLik gap | Max QuantMFR LogLik gap | Max saved vs recomputed LogLik gap |
| --- | --- | --- | --- |
| Monthly | 1.42e-14 | 6.14e-12 | 0.00e+00 |
| Quarterly | 2.84e-14 | 5.83e-12 | 0.00e+00 |

### Quarterly: every specification

| Model | Previous LogLik | New LogLik | Change | Previous status | New status | New min variance |
| --- | --- | --- | --- | --- | --- | --- |
| ARX(1,1)_normal_r2_arY_spfN_nuN | -179.8177 | -179.8177 | 0.0000 | verified | verified | 1.08e-01 |
| ARX(1,1)_normal_r2_arY_spfY_nuN | -179.6711 | -179.6711 | 0.0000 | verified | verified | 1.09e-01 |
| ARX(1,1)_normal_r2_arN_spfN_nuN | -182.9057 | -182.9057 | 0.0000 | verified | verified | 1.28e-01 |
| ARX(1,1)_normal_r3_arY_spfN_nuN | -167.3355 | -158.3755 | 8.9599 | verified | not_verified | 1.47e-07 |
| ARX(1,1)_normal_r3_arN_spfN_nuN | -179.7370 | -179.7370 | 0.0000 | verified | verified | 1.14e-01 |
| ARX(2,1)_normal_r2_arY_spfN_nuN | -91.2733 | -179.5673 | -88.2939 | likelihood_not_reproducible | verified | 1.00e-01 |
| ARX(2,1)_normal_r2_arY_spfY_nuN | -175.4087 | -175.4087 | 0.0000 | not_verified | not_verified | 8.72e-05 |
| ARX(2,1)_normal_r2_arN_spfN_nuN | -181.6105 | -181.6105 | 0.0000 | verified | verified | 1.20e-01 |
| ARX(2,1)_normal_r3_arY_spfN_nuN | -83.8557 | -151.5653 | -67.7096 | not_verified | not_verified | 2.73e-08 |
| ARX(2,1)_normal_r3_arN_spfN_nuN | -176.7746 | -176.7746 | 0.0000 | verified | verified | 9.76e-02 |
| ARX(2,2)_normal_r2_arY_spfN_nuN | -102.8982 | -179.5338 | -76.6356 | likelihood_not_reproducible | verified | 1.02e-01 |
| ARX(2,2)_normal_r2_arN_spfN_nuN | -181.5572 | -181.5572 | 0.0000 | verified | verified | 1.21e-01 |
| ARX(2,2)_normal_r2_arY_spfY_nuN | -178.7086 | -178.7086 | 0.0000 | verified | verified | 9.87e-02 |
| ARX(2,2)_normal_r3_arY_spfN_nuN | -19.4123 | -19.3506 | 0.0617 | not_verified | not_verified | 1.70e-31 |
| ARX(2,2)_normal_r3_arN_spfN_nuN | -176.7722 | -176.7722 | 0.0000 | verified | verified | 9.79e-02 |
| Constant_normal_r2_arN_spfN_nuN | -191.8370 | -191.8370 | 0.0000 | verified | verified | 1.40e-01 |
| Constant_normal_r3_arN_spfN_nuN | -186.8808 | -186.8808 | 0.0000 | verified | verified | 4.69e-02 |

### Monthly: every specification

| Model | Previous LogLik | New LogLik | Change | Previous status | New status | New min variance |
| --- | --- | --- | --- | --- | --- | --- |
| ARX(1,1)_normal_r2_arN_spfN_nuN | 89.9563 | 89.9563 | 0.0000 | verified | verified | 1.96e-02 |
| ARX(1,1)_normal_r2_arY_spfN_nuN | 91.8501 | 91.8501 | 0.0000 | verified | verified | 1.83e-02 |
| ARX(1,1)_normal_r2_arY_spfY_nuN | 91.8590 | 91.8590 | 0.0000 | verified | verified | 1.83e-02 |
| ARX(1,1)_normal_r3_arY_spfN_nuN | 118.1786 | 118.1786 | 0.0000 | not_verified | not_verified | 8.25e-06 |
| ARX(1,1)_normal_r3_arN_spfN_nuN | 100.3983 | 100.3983 | 0.0000 | verified | verified | 1.74e-02 |
| ARX(2,1)_normal_r2_arN_spfN_nuN | 90.0731 | 90.0731 | 0.0000 | verified | verified | 1.95e-02 |
| ARX(2,1)_normal_r2_arY_spfN_nuN | 92.3528 | 92.3528 | 0.0000 | verified | verified | 1.84e-02 |
| ARX(2,1)_normal_r2_arY_spfY_nuN | 92.3793 | 92.3793 | 0.0000 | verified | verified | 1.83e-02 |
| ARX(2,1)_normal_r3_arY_spfN_nuN | 116.7792 | 134.3093 | 17.5301 | verified | not_verified | 3.47e-07 |
| ARX(2,1)_normal_r3_arN_spfN_nuN | 100.6976 | 100.6976 | 0.0000 | not_verified | not_verified | 1.75e-02 |
| ARX(2,2)_normal_r2_arN_spfN_nuN | 90.1563 | 90.1563 | 0.0000 | verified | verified | 1.94e-02 |
| ARX(2,2)_normal_r2_arY_spfN_nuN | 92.4836 | 92.4836 | 0.0000 | verified | verified | 1.85e-02 |
| ARX(2,2)_normal_r2_arY_spfY_nuN | 93.1905 | 93.1905 | 0.0000 | verified | verified | 1.83e-02 |
| ARX(2,2)_normal_r3_arY_spfN_nuN | 119.0774 | 133.3244 | 14.2471 | not_verified | not_verified | 9.15e-07 |
| ARX(2,2)_normal_r3_arN_spfN_nuN | 100.8275 | 100.8275 | 0.0000 | not_verified | not_verified | 1.75e-02 |
| Constant_normal_r2_arN_spfN_nuN | 19.5811 | 19.5811 | 0.0000 | verified | verified | 1.91e-02 |
| Constant_normal_r3_arN_spfN_nuN | 38.6851 | 38.6851 | 0.0000 | not_verified | not_verified | 1.32e-02 |

## Interpretation and reproducibility

In flexible switching-mean Gaussian specifications, a regime can fit a few observations almost exactly while its variance tends to zero. Consequently the unrestricted likelihood is unbounded. Keeping the user-approved removal of the variance floor means these models need not possess a finite global MLE. A successful local perturbation check establishes only numerical reproducibility around the selected fit; it does not make the likelihood bounded. Conversely, a failed check need not mean variance collapse: optimizer failures or convergence to another local solution can also cause rejection. The saved verification CSVs distinguish these cases.

Regime-specific AR checks are the existing inflation application's restrictions. They should not be interpreted as a separate proof of stationarity for arbitrary Markov-switching AR dynamics.

The previous summaries, parameters, verification attempts, and notes are archived under `previous/`. `likelihood_audit.csv` records the previous-fit cross-checks; `provenance.json` records their input hashes and the reference commit. The `updated/` directory contains the same audit of the fresh estimates. Fresh search blocks and scheduler logs are at `/scratch/hs3370/Inflation_Dynamics/RegimeSwitching/20261112_quantmfr_joint_em_01/`. Collected production output includes every initial attempt in `regime_switching_attempts.csv.gz` and each final refit separately in `regime_switching_stability_attempts.csv`.

```bash
~/.conda/envs/inflation_dynamics/bin/python RegimeSwitching/audit_estimation.py \
  --reference-repo /user/hs3370/TenuousBeliefs \
  --output-dir RegimeSwitching/audit_20261112/updated
~/.conda/envs/inflation_dynamics/bin/python RegimeSwitching/write_audit_report.py
```
