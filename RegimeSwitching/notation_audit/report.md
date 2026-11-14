```{raw:typst}
#set page(margin: auto)
```

# Derivation-to-code audit

Completed 2026-11-13 using the 2026-10-20 data vintage. The rewritten
[methodology](../README.md) uses the supplied VAR-regime notation.
No estimation settings, model menu, parameters, or data were changed.

## Timing and storage correspondence

The notes define $Z_t$ as the regime generating $\pi_{t+1}$. If the
implementation's observation-date regime is $s_{t+1}$, then $Z_t=e_{s_{t+1}}$.
This is a change of index, not a change of the model.

| Notes, at observation $\pi_{t+1}$ | Production result |
| --- | --- |
| Row-stochastic $\mathbb P$, origin then destination | `regime_transition[:, :, 0].T` |
| $Q_t=E(Z_t\mid\mathcal F_t)$ | `predicted_marginal_probabilities` at that observation |
| $R_t=E(Z_t\mid\mathcal F_{t+1})$ | `filtered_marginal_probabilities` at that observation |
| $S_t=E(Z_t\mid\mathcal F_T)$ | `smoothed_marginal_probabilities` at that observation |
| $J_{ij,t}$ for $Z_t$ to $Z_{t+1}$ | `smoothed_joint_probabilities[j,i,t+1]` with zero-based sample columns |

In particular, $Q_t$ must not be identified with the code's filtered
probability for the same observation. The notes update $Q_t$ to $R_t$
and then propagate to $Q_{t+1}$; the code stores all three at their
corresponding observation dates.

## Parameter sharing and estimation

`build_model` maps the AR switch to `switching_trend` and the inflation-lag
entries of `switching_exog`, the SPF switch to its own regressor entries,
and the variance switch to `switching_variance`. A shared coefficient
has one parameter index reused by all regimes. The menu contains exactly
17 Gaussian specifications, all with switching variances. Mean-only
switching is derived for explanation and tested separately; it is not
added to production estimation.

`JointMeanMarkovRegression._em_exog` solves the constrained joint weighted
regression in the notes, with weights $S_{i,t}/(\sigma_i^2)^{old}$.
The inherited `_em_variance` supplies the stated weighted residual update.
Tests check their score equations and both switching and common variances.
The likelihood, Hamilton filter, and Kim smoother are inherited from
statsmodels 0.14.2. Final BFGS/L-BFGS estimates all free parameters jointly.

There is one qualification to calling the initializer exact EM.
`MarkovSwitching._em_regime_transition` includes the pre-sample transition
in its numerator and divides by summed in-sample smoothed origin-regime
probabilities. It also omits the derivative of the stationary initial
probability term. It is consequently not the exact transition maximizer
of the finite-sample stationary-initialized objective displayed in the
notes. This existing initializer is retained and explicitly distinguished
from final likelihood maximization; the final objective does include the
stationary initial distribution. No production change or re-estimation
is needed for the notation rewrite.

For numerical log calculations, statsmodels uses a $10^{-20}$ safeguard
for zero probabilities. This is distinct from the removed economic
transition-probability cutoff and variance floor. The independent audit
uses exact zeros and removes only last-entry subtraction roundoff in the
reported transition matrix.

## Numerical comparison on all saved specifications

The independent implementation in `audit_derivation.py` constructs the
Gaussian densities, runs the notes' forward recursion and backward
recursion, and computes smoothed transition probabilities and predictive
variances. It checks all 34 saved specifications, including candidates
excluded from the model ranking. Checking their likelihood calculations
does not make those excluded candidates verified estimates.

All coefficient-sharing checks pass. Every reported gap below is the
largest absolute difference over the models and dates of that frequency:

| frequency | likelihood_gap | predicted_gap | filtered_gap | smoothed_gap | pair_gap | variance_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Monthly | 2.13e-14 | 1.78e-15 | 9.99e-16 | 7.11e-15 | 6.99e-15 | 1.87e-15 |
| Quarterly | 2.84e-14 | 1.44e-15 | 2.11e-15 | 3.00e-15 | 3.00e-15 | 4.22e-15 |

The independent recursion is additionally checked against direct
summation over every hidden-state path on short synthetic samples.
These tests cover every production configuration, mean-only switching,
and identical regimes with no signal about the state. They verify
filtering, smoothing, transition-pair marginals, and the likelihood.

Reproduce with the project Python environment:

```bash
python -m unittest discover -s RegimeSwitching -p 'test_*.py'
python RegimeSwitching/audit_derivation.py
```

`results.csv` contains every model's errors; `input_hashes.json` identifies
the unchanged estimation inputs. The audit is separate from the production
estimator and does not refit or rewrite parameter estimates.
