"""Compare saved inflation fits with an independent filter and QuantMFR.

The external repository is read at runtime; none of its private code or data
is copied into this project. Only its data-preparation function is adapted to
the already trimmed inflation response/design matrix (no extra lag trimming).
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import estimate_regime_switching_models as rs


def independent_filter(y, means, variances, transition):
    k = len(variances)
    stationary = np.linalg.lstsq(
        np.vstack([transition.T - np.eye(k), np.ones(k)]),
        np.r_[np.zeros(k), 1.], rcond=None)[0]
    if stationary.min() < -1e-10:
        raise ValueError('Invalid stationary probabilities')
    stationary = np.maximum(stationary, 0)
    stationary /= stationary.sum()
    probabilities = stationary.copy()
    filtered, predicted, increments = [], [], []
    log_density = -0.5 * (np.log(2 * np.pi * variances)[None, :] +
                           (y[:, None] - means)**2 / variances[None, :])
    for density in log_density:
        predicted.append(probabilities.copy())
        with np.errstate(divide='ignore'):
            joint = np.log(probabilities) + density
        increment = logsumexp(joint)
        posterior = np.exp(joint - increment)
        filtered.append(posterior)
        increments.append(increment)
        probabilities = posterior @ transition
    return np.asarray(increments), np.asarray(predicted), np.asarray(filtered)


def run(reference_repo, output):
    spec = importlib.util.spec_from_file_location('quantmfr_reference', reference_repo / 'MLE.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'reference_commit': subprocess.check_output(
        ['git', '-C', str(reference_repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'reference_url': 'https://github.com/lphansen/TenuousBeliefs/tree/QuantMFR',
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'methodology': rs.METHODOLOGY_VERSION,
        'files_sha256': {}}
    for filename in ['estimate_regime_switching_models.py', 'joint_em.py', 'audit_estimation.py']:
        path = rs.RS_DIR / filename
        provenance['files_sha256'][str(path.relative_to(rs.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for frequency, folder in [('Quarterly', 'results'), ('Monthly', 'results_monthly')]:
        data_path = rs.ROOT / 'DataSummary' / f'Aggregate_CPI_inflation_{frequency}.pkl'
        data = rs.load_effective_sample(data_path)
        directory = rs.RS_DIR / folder
        paths = [data_path, directory / 'regime_switching_results.csv',
                 directory / 'regime_switching_parameters.csv',
                 directory / 'regime_switching_stability_attempts.csv']
        for path in paths:
            provenance['files_sha256'][str(path.relative_to(rs.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary = pd.read_csv(paths[1], float_precision='round_trip')
        parameters = pd.read_csv(paths[2], float_precision='round_trip')
        attempts = pd.read_csv(directory / 'regime_switching_attempts.csv.gz')
        np.testing.assert_allclose(data.SPF_shock, data.Inflation-data.SPF, atol=0, rtol=0)
        for _, saved in summary.iterrows():
            rs._check_saved_sample(data, saved)
            subset = parameters[parameters.model_label == saved.model_label]
            result = rs._reconstruct_result_from_saved_params(data, saved, subset)
            model, params = result.model, np.asarray(result.params)
            k = model.k_regimes
            design = model.exog if model.exog is not None else np.empty((len(data), 0))
            coefficients = [params[model.parameters[j, 'exog']][None, :] for j in range(k)]
            means = np.column_stack([design @ b.ravel() for b in coefficients])
            variances = np.array([params[model.parameters[j, 'variance']][0] for j in range(k)])
            transition = result.regime_transition[:, :, 0].T.copy()
            # Remove only subtraction roundoff in the last implied probability.
            if transition.min() < -1e-12:
                raise ValueError('Transition matrix has negative probabilities')
            transition = np.maximum(transition, 0)
            transition /= transition.sum(axis=1, keepdims=True)
            increments, predicted, filtered = independent_filter(
                model.endog, means, variances, transition)
            with patch.object(source, 'prepare_var_data', return_value=([model.endog], design)):
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    ref_filt, ref_smooth, ref_llf = source.em_e_step(
                        [], [], coefficients, [np.array([[np.sqrt(v)]]) for v in variances], transition)
            a = attempts[attempts.model_label == saved.model_label]
            coverage = len(a) == 5000 and set(a.attempt) == set(range(1, 5001))
            rows.append({
                'frequency': frequency, 'model_label': saved.model_label,
                'nobs': len(data), 'sample_start': str(data.index[0]), 'sample_end': str(data.index[-1]),
                'saved_llf': saved.llf, 'statsmodels_llf': result.llf,
                'independent_llf': increments.sum(), 'quantmfr_llf': ref_llf,
                'independent_minus_statsmodels': increments.sum()-result.llf,
                'quantmfr_minus_statsmodels': ref_llf-result.llf,
                'saved_minus_recomputed': saved.llf-result.llf,
                'max_filtered_probability_difference': np.max(np.abs(ref_filt.T-result.filtered_marginal_probabilities)),
                'max_smoothed_probability_difference': np.max(np.abs(ref_smooth.T-result.smoothed_marginal_probabilities)),
                'min_variance': variances.min(), 'full_search_coverage': coverage,
                'initial_starts': len(a), 'stability_status': saved.stability_status,
                'aic': saved.aic, 'bic': saved.bic,
            })
    pd.DataFrame(rows).to_csv(output / 'likelihood_audit.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(pd.DataFrame(rows).groupby('frequency')[
        ['independent_minus_statsmodels', 'quantmfr_minus_statsmodels',
         'max_filtered_probability_difference', 'max_smoothed_probability_difference']].agg(lambda x: abs(x).max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=rs.RS_DIR / 'audit_20261112')
    args = parser.parse_args()
    run(args.reference_repo, args.output_dir)
