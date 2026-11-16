"""OLS/HAC draws, shared coefficients, and transition orientation."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import estimate_regime_switching_models as rs
from ols_initialization import draw_ols_start, ols_reference
from test_estimation_methodology import synthetic_data


class OLSInitializationTests(unittest.TestCase):
    def test_reference_matches_independent_ols_and_hac(self):
        for _, spec in rs.model_entries(False)[0]:
            model, _, _ = rs.build_model(synthetic_data(), spec)
            ref = ols_reference(model)
            y = np.asarray(model.endog)
            if model.exog is not None:
                x = np.asarray(model.exog)
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                u = y - x @ beta
                scores = x * u[:, None]
                meat = scores.T @ scores
                for lag in range(1, 13):
                    cross = scores[lag:].T @ scores[:-lag]
                    meat += (1 - lag / 13) * (cross + cross.T)
                bread = np.linalg.inv(x.T @ x)
                np.testing.assert_allclose(ref.coefficients, beta, atol=1e-12)
                np.testing.assert_allclose(ref.standard_errors, np.sqrt(np.diag(bread @ meat @ bread)))
            else:
                u = y
                self.assertEqual(ref.coefficients, [])
            scores = u**2 - np.mean(u**2)
            long_run_sum = scores @ scores
            for lag in range(1, 13):
                long_run_sum += 2 * (1 - lag / 13) * (scores[lag:] @ scores[:-lag])
            sigma = np.sqrt(np.mean(u**2))
            self.assertAlmostEqual(ref.sigma, sigma)
            self.assertAlmostEqual(ref.sigma_standard_error, np.sqrt(long_run_sum) / len(u) / (2 * sigma))

    def test_draws_respect_sharing_scale_and_transition_orientation(self):
        for _, spec in rs.model_entries(False)[0]:
            with self.subTest(spec=rs.model_label(spec)):
                model, _, _ = rs.build_model(synthetic_data(), spec)
                ref = ols_reference(model)
                rng = np.random.default_rng(123)
                expected = {}
                for regime in range(model.k_regimes):
                    for feature, index in enumerate(model.parameters[regime, "exog"]):
                        if index not in expected:
                            expected[index] = rng.uniform(
                                ref.coefficients[feature] - 2 * ref.standard_errors[feature],
                                ref.coefficients[feature] + 2 * ref.standard_errors[feature],
                            )
                scales = rng.uniform(max(0, ref.sigma - 2 * ref.sigma_standard_error),
                                     ref.sigma + 2 * ref.sigma_standard_error, model.k_regimes)
                rows = rng.dirichlet(np.ones(model.k_regimes), size=model.k_regimes)
                params = draw_ols_start(model, np.random.default_rng(123))
                for index, value in expected.items():
                    self.assertEqual(params[index], value)
                np.testing.assert_allclose(params[model.parameters["variance"]], scales**2)
                np.testing.assert_allclose(model.regime_transition_matrix(params)[:, :, 0].T, rows)
                self.assertTrue(np.isfinite(model.loglike(params)))

    def test_search_records_reference(self):
        spec = rs.model_entries(False)[0][0][1]
        record = rs.search_model(synthetic_data(), spec, range(1, 2))
        self.assertEqual(record["initialization_reference"]["nobs"], 180)
        self.assertEqual(record["search_methodology"], rs.METHODOLOGY_VERSION)

    def test_failed_cases_are_not_printed(self):
        rows = pd.read_csv(Path(rs.RS_DIR) / "results" / "regime_switching_results.csv")
        rows.loc[:, "checks_passed"] = False
        markdown = rs.build_results_markdown(rows)
        self.assertNotIn("rejected estimate", markdown)
        self.assertNotIn("Refits returned", markdown)


if __name__ == "__main__":
    unittest.main()
