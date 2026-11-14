"""Check the supplied Q/R/S derivation against the production model.

This independent implementation is an audit, not a production estimator.
Rows are indexed by the observation date: row t contains Q_t, R_t and S_t
for the regime Z_t generating pi_{t+1} in the notes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import estimate_regime_switching_models as rs


def forward_backward(log_densities, transition, initial):
    n, k = log_densities.shape
    q = np.empty((n, k))
    r = np.empty_like(q)
    increments = np.empty(n)
    with np.errstate(divide='ignore'):
        log_p = np.log(transition)
        prior = np.log(initial)
    for t in range(n):
        q[t] = prior
        joint = prior + log_densities[t]
        increments[t] = logsumexp(joint)
        r[t] = joint - increments[t]
        prior = logsumexp(r[t, :, None] + log_p, axis=0)
    s = r.copy()
    pairs = np.zeros((n-1, k, k))
    for t in range(n-2, -1, -1):
        reachable = np.isfinite(q[t+1])
        ratio = np.full(k, -np.inf)
        ratio[reachable] = s[t+1, reachable] - q[t+1, reachable]
        joint = r[t, :, None] + log_p + ratio[None, :]
        pairs[t] = np.exp(joint)
        s[t] = logsumexp(joint, axis=1)
    return dict(Q=np.exp(q), R=np.exp(r), S=np.exp(s), J=pairs,
                loglike=float(increments.sum()))


def model_inputs(model, params):
    """Construct the observation densities independently from coefficient blocks."""
    params = np.asarray(params)
    k = model.k_regimes
    p = model.regime_transition_matrix(params)[:, :, 0].T.copy()
    if p.min() < -1e-12:
        raise ValueError('Invalid transition matrix')
    p = np.maximum(p, 0)  # Only implied last-row subtraction roundoff.
    p /= p.sum(axis=1, keepdims=True)
    initial = np.linalg.lstsq(np.vstack([p.T-np.eye(k), np.ones(k)]),
                              np.r_[np.zeros(k), 1.], rcond=None)[0]
    if initial.min() < -1e-12:
        raise ValueError('Invalid stationary distribution')
    initial = np.maximum(initial, 0)
    initial /= initial.sum()
    means = np.column_stack([
        model.exog @ params[model.parameters[i, 'exog']]
        if model.exog is not None else np.zeros(model.nobs) for i in range(k)])
    variances = np.array([params[model.parameters[i, 'variance']][0] for i in range(k)])
    log_densities = -0.5*(np.log(2*np.pi*variances) +
                          (model.endog[:, None]-means)**2/variances)
    return log_densities, p, initial, means, variances


def check_sharing(model, spec):
    """Check storage indices against the economic restrictions, not fit proximity."""
    k = model.k_regimes
    columns = ['const'] + list(spec.mean_process.exog_cols) if model.exog is not None else []
    for colno, col in enumerate(columns):
        switches = (spec.switch_spec.switching_ar if col == 'const' or rs.is_ar_col(col)
                    else spec.switch_spec.switching_spf)
        indices = [model.parameters[i, 'exog'][colno] for i in range(k)]
        assert len(set(indices)) == (k if switches else 1), col
    indices = [model.parameters[i, 'variance'][0] for i in range(k)]
    assert len(set(indices)) == (k if spec.switch_spec.switching_distribution else 1)


def run():
    out = rs.RS_DIR / 'notation_audit'
    out.mkdir(exist_ok=True)
    rows, hashes = [], {}
    for frequency, folder in [('Quarterly','results'),('Monthly','results_monthly')]:
        directory = rs.RS_DIR / folder
        paths = [directory/'regime_switching_results.csv', directory/'regime_switching_parameters.csv',
                 rs.ROOT/'DataSummary'/f'Aggregate_CPI_inflation_{frequency}.pkl']
        for path in paths:
            hashes[str(path.relative_to(rs.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary = pd.read_csv(paths[0], float_precision='round_trip')
        parameters = pd.read_csv(paths[1], float_precision='round_trip')
        data = rs.load_effective_sample(paths[2])
        for _, saved in summary.iterrows():
            rs._check_saved_sample(data, saved)
            spec = rs._spec_from_row(saved)
            result = rs._reconstruct_result_from_saved_params(
                data, saved, parameters[parameters.model_label == saved.model_label])
            model = result.model
            check_sharing(model, spec)
            densities, p, prior, means, variances = model_inputs(model, result.params)
            ref = forward_backward(densities, p, prior)
            prediction = (ref['Q']*means).sum(axis=1)
            variance = (ref['Q']*(variances+(means-prediction[:, None])**2)).sum(axis=1)
            pair_marginal_gap = max(np.max(abs(ref['J'].sum(axis=2)-ref['S'][:-1])),
                                    np.max(abs(ref['J'].sum(axis=1)-ref['S'][1:])))
            # statsmodels joint array is [destination, origin, observation].
            pairs = np.asarray(result.smoothed_joint_probabilities)[:, :, 1:].transpose(2,1,0)
            row = dict(frequency=frequency, model_label=saved.model_label, nobs=len(data),
                       sharing_passed=True, likelihood_gap=abs(ref['loglike']-result.llf),
                       saved_likelihood_gap=abs(ref['loglike']-saved.llf),
                       predicted_gap=np.max(abs(ref['Q']-np.asarray(result.predicted_marginal_probabilities))),
                       filtered_gap=np.max(abs(ref['R']-np.asarray(result.filtered_marginal_probabilities))),
                       smoothed_gap=np.max(abs(ref['S']-np.asarray(result.smoothed_marginal_probabilities))),
                       pair_gap=np.max(abs(ref['J']-pairs)), pair_marginal_gap=pair_marginal_gap,
                       variance_gap=np.max(abs(variance-rs._conditional_variance_path(result))))
            for key, value in row.items():
                if key.endswith('_gap'):
                    assert np.isfinite(value) and value < 1e-8, (saved.model_label, key, value)
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out/'results.csv',index=False)
    (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    metrics = ['likelihood_gap','predicted_gap','filtered_gap','smoothed_gap','pair_gap','variance_gap']
    maxima = result.groupby('frequency')[metrics].max()
    print(maxima.to_string())
    display = maxima.map(lambda x:f'{x:.2e}').reset_index()
    report = rs.TYPST_PREAMBLE + '\n\n' + r'''# Derivation-to-code audit

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

'''+rs._markdown_table(display)+r'''

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
'''
    (out/'report.md').write_text(report)


if __name__ == '__main__':
    run()
