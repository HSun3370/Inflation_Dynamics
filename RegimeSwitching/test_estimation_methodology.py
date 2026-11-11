"""Identification, independent likelihood, and final-optimum regression checks.

Run with the project environment:
    python -m unittest discover -s RegimeSwitching -p 'test_*.py'
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, t

import estimate_regime_switching_models as rs


def synthetic_data(n=180):
    rng = np.random.default_rng(8927)
    spf = rng.normal(0.5, 0.2, n)
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        state[i] = 1 - state[i - 1] if rng.uniform() < 0.06 else state[i - 1]
    inflation = spf + rng.normal(size=n) * np.where(state == 0, 0.2, 0.8)
    return pd.DataFrame({
        "Inflation": inflation,
        "Inflation_lag_1": np.r_[0, inflation[:-1]],
        "Inflation_lag_2": np.r_[0, 0, inflation[:-2]],
        "SPF": spf, "SPF_lag_1": np.r_[0.5, spf[:-1]],
        "SPF_shock": inflation - spf,
    }, index=pd.period_range("1970Q1", periods=n, freq="Q"))


def independent_likelihood(model, params, student_t=False):
    """Forward recursion separate from statsmodels, including its initial prior."""
    matrix = model.regime_transition_matrix(params)[:, :, 0].T
    k = model.k_regimes
    probabilities = np.linalg.lstsq(
        np.vstack([matrix.T - np.eye(k), np.ones(k)]), np.r_[np.zeros(k), 1], rcond=None
    )[0]
    densities = []
    for regime in range(k):
        mean = (model.exog @ params[model.parameters[regime, "exog"]]
                if model.exog is not None else np.zeros(model.nobs))
        variance = params[model.parameters[regime, "variance"]][0]
        if student_t:
            nu = params[model.parameters[regime, "nu"]][0]
            density = t.logpdf(model.endog, nu, loc=mean, scale=np.sqrt(variance * (nu - 2) / nu))
        else:
            density = norm.logpdf(model.endog, loc=mean, scale=np.sqrt(variance))
        densities.append(density)
    densities = np.column_stack(densities)
    total = 0.0
    for density in densities:
        joint = np.log(probabilities) + density
        increment = logsumexp(joint)
        total += increment
        probabilities = np.exp(joint - increment) @ matrix
    return total


class IdentificationTests(unittest.TestCase):
    def test_every_menu_entry_preserves_density_parameters_and_state_paths(self):
        df = synthetic_data()
        specs, _ = rs.build_model_specs(include_student_t=True)
        for spec in specs:
            with self.subTest(model=rs.model_label(spec)):
                model, _, _ = rs.build_model(df, spec)
                params = model.start_params
                k = model.k_regimes
                matrix = (np.array([[0.9, 0.15], [0.1, 0.85]]) if k == 2 else
                          np.array([[0.85, 0.06, 0.03], [0.10, 0.90, 0.07], [0.05, 0.04, 0.90]]))
                params[model.parameters["regime_transition"]] = matrix[:-1].ravel()
                params[model.parameters["variance"]] = [0.9, 0.1] if k == 2 else [0.4, 0.9, 0.1]
                original = model.smooth(params, cov_type="none")
                ordered, order = rs._variance_ordered_result(original)
                expected_llf = independent_likelihood(model, params, spec.error_distribution == "Student t")
                self.assertAlmostEqual(original.llf, expected_llf, places=9)
                self.assertAlmostEqual(ordered.llf, expected_llf, places=9)
                self.assertAlmostEqual(original.aic, ordered.aic, places=9)
                np.testing.assert_allclose(
                    ordered.smoothed_marginal_probabilities,
                    np.asarray(original.smoothed_marginal_probabilities)[:, order], atol=1e-12,
                )
                np.testing.assert_allclose(
                    ordered.regime_transition[:, :, 0], matrix[np.ix_(order, order)], atol=1e-12,
                )
                for new, old in enumerate(order):
                    for block in model.parameters.k_parameters:
                        if block != "regime_transition":
                            np.testing.assert_allclose(
                                ordered.params.iloc[model.parameters[new, block]],
                                params[model.parameters[int(old), block]],
                            )

    def test_boundary_transition_logits_and_small_positive_variances(self):
        spec = rs.ModelSpec(rs.mean_processes()[0], "Normal", rs.SwitchingSpec(3, False, False))
        model, _, _ = rs.build_model(synthetic_data(), spec)
        params = model.start_params
        params[model.parameters["variance"]] = [1e-12, 0.2, 0.5]
        matrix = np.array([[1, 0, 0.1], [0, 0.8, 0.2], [0, 0.2, 0.7]])
        params[model.parameters["regime_transition"]] = matrix[:-1].ravel()
        coordinates = rs._optimization_coordinates(model, params)
        recovered = model.transform_params(coordinates)
        np.testing.assert_allclose(recovered, params, atol=3e-12, rtol=1e-10)
        result = SimpleNamespace(model=model, params=params, regime_transition=matrix[:, :, None])
        self.assertTrue(rs._distribution_check(result)[0])
        self.assertTrue(rs._transition_check(result)[0])
        params[model.parameters["variance"]][0] = 0
        self.assertFalse(rs._distribution_check(result)[0])


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_data(400)
        cls.spec = rs.ModelSpec(rs.mean_processes()[0], "Normal", rs.SwitchingSpec(2, False, False))
        cls.model, cls.mask, _ = rs.build_model(cls.df, cls.spec)
        cls.fitted = cls.model.fit(method="lbfgs", em_iter=50, maxiter=1500, cov_type="none", disp=False)

    def test_improved_optimum_requires_fresh_verification_batch(self):
        params = np.asarray(self.fitted.params).copy()
        params[self.model.parameters["variance"]] *= 1.10
        provisional = self.model.smooth(params, cov_type="none")
        provisional.mle_retvals = {"converged": True}
        final, diagnostics, audit = rs._verify_solution(provisional, self.spec, self.mask, n_reps=4)
        self.assertGreater(final.llf, provisional.llf + rs.STABILITY_LLF_TOL)
        self.assertGreaterEqual(diagnostics["stability_rounds"], 2)
        self.assertTrue(diagnostics["stability_verified"])
        self.assertEqual(diagnostics["stability_returns"], 4)
        self.assertTrue(all(row["returned"] for row in audit[-4:]))

    def test_optimizer_failure_does_not_count_as_return(self):
        failed = self.model.smooth(self.fitted.params, cov_type="none")
        failed.mle_retvals = {"converged": False}
        with patch.object(self.model, "fit", return_value=failed):
            _, diagnostics, audit = rs._verify_solution(self.fitted, self.spec, self.mask, n_reps=2)
        self.assertFalse(diagnostics["stability_verified"])
        self.assertEqual(diagnostics["stability_returns"], 0)
        self.assertTrue(all(not row["returned"] for row in audit))

    def test_improvement_at_round_limit_is_not_verified_without_new_batch(self):
        params = np.asarray(self.fitted.params).copy()
        params[self.model.parameters["variance"]] *= 1.1
        provisional = self.model.smooth(params, cov_type="none")
        with patch.object(rs, "STABILITY_MAX_ROUNDS", 1):
            final, diagnostics, _ = rs._verify_solution(provisional, self.spec, self.mask, n_reps=2)
        self.assertGreater(final.llf, provisional.llf + rs.STABILITY_LLF_TOL)
        self.assertFalse(diagnostics["stability_verified"])

    def test_grid_block_partition_preserves_search_and_selection(self):
        together = rs.search_model(self.df, self.spec, range(1, 5))
        separate = [rs.search_model(self.df, self.spec, range(1, 3)),
                    rs.search_model(self.df, self.spec, range(3, 5))]
        one, params_one, attempts_one = rs.select_model([together])
        two, params_two, attempts_two = rs.select_model(separate)
        self.assertEqual(one, two)
        self.assertEqual(params_one, params_two)
        self.assertEqual(attempts_one, attempts_two)
        self.assertEqual(one["fit_attempts"], 4)

    def test_reports_preserve_failed_verification_and_reject_wrong_sample(self):
        candidate = rs._candidate(self.df, self.spec, self.mask, False,
                                  (self.fitted.llf, 1, "lbfgs#1", self.fitted))
        row = candidate["summary_row"]
        row.update(fit_attempts=5000, fit_strategy="lbfgs#1", stability_verified=False,
                   stability_status="not_verified", verification_methodology=rs.METHODOLOGY_VERSION)
        finalized, _, _ = rs._finalize_estimates(
            self.df, pd.DataFrame([row]), pd.DataFrame(candidate["parameter_rows"]), verify=False,
        )
        self.assertFalse(finalized.iloc[0]["checks_passed"])
        with self.assertRaisesRegex(ValueError, "Saved sample"):
            rs._check_saved_sample(self.df.iloc[1:], pd.Series(row))

    def test_report_only_does_not_invent_legacy_verification(self):
        candidate = rs._candidate(self.df, self.spec, self.mask, False,
                                  (self.fitted.llf, 1, "lbfgs#1", self.fitted))
        row = candidate["summary_row"]
        row.update(fit_attempts=5000, fit_strategy="lbfgs#1")
        finalized, _, audit = rs._finalize_estimates(
            self.df, pd.DataFrame([row]), pd.DataFrame(candidate["parameter_rows"]), verify=False,
        )
        self.assertFalse(finalized.iloc[0]["stability_verified"])
        self.assertTrue(audit.empty)
        markdown = rs.build_results_markdown(finalized)
        self.assertIn("has not yet been verified", markdown)
        self.assertIn("former transition and variance cutoffs", markdown)

    def test_old_filtered_grid_blocks_cannot_masquerade_as_new_search(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            pd.to_pickle({"model_label": rs.model_label(self.spec)}, directory / "R01_0001-0004.pkl")
            with self.assertRaisesRegex(SystemExit, "older collection rules"):
                rs.collect_model_jobs(self.df, None, directory, directory, [("R01", self.spec)], [], 4, False)

    def test_collector_verifies_once_after_combining_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw = directory / "raw"
            raw.mkdir()
            for first, last in [(1, 2), (3, 4)]:
                record = rs.search_model(self.df, self.spec, range(first, last + 1))
                pd.to_pickle(record, raw / f"R01_{first:04d}-{last:04d}.pkl")
            with patch.object(rs, "_verify_solution", wraps=rs._verify_solution) as verify:
                rs.collect_model_jobs(
                    self.df, None, raw, directory, [("R01", self.spec)], [], 4, False, stability_reps=2,
                )
            self.assertEqual(verify.call_count, 1)
            summary = pd.read_csv(directory / "regime_switching_results.csv")
            self.assertEqual(summary.iloc[0]["fit_attempts"], 4)
            self.assertEqual(summary.iloc[0]["search_methodology"], rs.METHODOLOGY_VERSION)
            self.assertEqual(len(pd.read_csv(directory / "regime_switching_attempts.csv.gz")), 4)
            self.assertGreaterEqual(len(pd.read_csv(directory / "regime_switching_stability_attempts.csv")), 2)
            markdown = (directory / "regime_switching_results.md").read_text()
            self.assertTrue(markdown.startswith(rs.TYPST_PREAMBLE))
            self.assertNotIn("former transition and variance cutoffs", markdown)


if __name__ == "__main__":
    unittest.main()
