"""Check nesting, restricted recursions, free counts and search/collector agreement."""
import importlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from BEGE_GARCH import bege_batch as batch
from BEGE_GARCH.full_restrictions import expand_volatility, parameter_names, sample_volatility

model = importlib.import_module('BEGE_GARCH.BEGE_GARCH')
ROOT = Path(__file__).resolve().parents[1]


class RestrictedFullTests(unittest.TestCase):
    def test_direct_recurrence_and_full_nesting(self):
        residuals = np.array([1., -2., 0., .5, -1.])
        vol = np.array([.2, .1, .5, .4, .15, .2, .3, .7, .6])
        for family in ('symmetric_p', 'positive_p'):
            p, n, sp, sn, _ = batch._vol_paths_from_theta(
                vol, mean_type='constant', model_family=family, residuals=residuals)
            want_p = [.2 / (1 - .5 - (.15 if family == 'symmetric_p' else .075))]
            want_n = [.1 / (1 - .4 - .25)]
            for u in residuals[:-1]:
                shock = u*u if family == 'symmetric_p' else max(u, 0)**2
                want_p.append(.2 + .5*want_p[-1] + .15/(2*sp*sp)*shock)
                want_n.append(.1 + .4*want_n[-1] + (.2 if u > 0 else .3)/(2*sn*sn)*u*u)
            np.testing.assert_allclose(p, want_p)
            np.testing.assert_allclose(n, want_n)
            full = batch._vol_paths_from_theta(expand_volatility(vol, family),
                mean_type='constant', model_family='full', residuals=residuals)
            np.testing.assert_array_equal(p, full[0])
            np.testing.assert_array_equal(n, full[1])

    def test_starts_and_stability_rejection(self):
        for family in ('symmetric_p', 'positive_p'):
            rng = np.random.default_rng(123)
            for _ in range(500):
                vol = sample_volatility(rng, family, p0_bounds=(0, 10), rho_bounds=(0, 1),
                    phi_bounds=(0, 2), sigma_bounds=(1e-5, 2), floor_eps=1e-6)
                self.assertEqual(len(vol), 9)
                self.assertTrue(batch._constraints_ok_for_theta(vol, mean_type='constant', model_family=family))
                self.assertTrue(np.all(vol >= 0))
            vol[2] = 1.
            self.assertFalse(batch._constraints_ok_for_theta(vol, mean_type='constant', model_family=family))

    def test_search_objective_matches_collection_both_frequencies_all_means(self):
        # Replace optimization, not its objective: evaluate the actual submitted
        # random starts and verify the collector reproduces the same likelihood.
        def evaluate_starts(starts, objective, **kwargs):
            return [(OptimizeResult(x=x, fun=objective(x), success=True, status=0), None)
                    for x in starts]
        for frequency, expected_n in [('quarterly', 230), ('monthly', 692)]:
            with patch.dict(os.environ, BEGE_FREQUENCY=frequency):
                specs = batch.build_model_specs(batch.load_effective_sample(ROOT), True)
                for spec in specs:
                    self.assertEqual(len(spec['Y']), expected_n)
                    for family in ('symmetric_p', 'positive_p'):
                        with patch.object(model, '_run_multistart_minimize', evaluate_starts):
                            result = model.BEGE_FullGJR_MLE(spec['Y'], spec['X'], spec['mean_type'],
                                n_starts=2, random_state=10201, print_summary=False, p_restriction=family)
                        row = pd.Series(batch.result_to_row(result, spec['mean_type'], 1, 1, 10201, parameter_names(family)))
                        metrics = batch._selection_metrics_for_row(row, model_family=family,
                            specs_by_mean={spec['mean_type']: spec})
                        for key in ('loglik', 'AIC', 'BIC'):
                            self.assertAlmostEqual(result[key], metrics['corrected_'+key], places=8)
                        k = len(batch.MEAN_PARAM_NAMES[spec['mean_type']]) + 9
                        self.assertEqual(len(result['params']), k)
                        self.assertEqual(len(batch._bounds_for_row(spec, spec['mean_type'], family)), k)
                        nll, _ = batch._row_likelihood_functions(spec=spec, mean_type=spec['mean_type'], model_family=family)
                        self.assertAlmostEqual(nll(result['params']), -result['loglik'], places=8)

    def test_monthly_full_boundary_benchmark(self):
        with patch.dict(os.environ, BEGE_FREQUENCY='monthly'):
            row = pd.read_csv(ROOT/'BEGE_GARCH/Full_BEGE/results_monthly/best_loglik_with_se.csv').iloc[0]
            specs = {s['mean_type']: s for s in batch.build_model_specs(batch.load_effective_sample(ROOT), True)}
            metrics = batch._selection_metrics_for_row(row, model_family='positive_p', specs_by_mean=specs)
            self.assertAlmostEqual(metrics['corrected_loglik'], row['loglik'], places=6)
            self.assertAlmostEqual(metrics['corrected_AIC'], row['AIC']-2, places=6)
            self.assertAlmostEqual(metrics['corrected_BIC'], row['BIC']-np.log(692), places=6)


if __name__ == '__main__':
    unittest.main()
