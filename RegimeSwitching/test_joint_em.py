"""Check the EM regression step against its complete-data objective."""

import unittest
import numpy as np
from types import SimpleNamespace
from joint_em import JointMeanMarkovRegression


class JointEMTests(unittest.TestCase):
    def test_real_smoother_supplies_old_variances_to_shared_mean_update(self):
        rng = np.random.default_rng(93)
        x = rng.normal(size=(80, 2))
        model = JointMeanMarkovRegression(
            x[:, 0] + rng.normal(size=80), k_regimes=3, exog=x,
            switching_trend=True, switching_exog=[True, False], switching_variance=True)
        old = model.start_params.copy()
        result, new = model._em_iteration(old)
        self.assertTrue(np.all(np.isfinite(new)))
        coefficients = np.stack([new[model.parameters[k, 'exog']] for k in range(3)])
        score = np.stack([
            model.exog.T @ (result.smoothed_marginal_probabilities[k] /
                           old[model.parameters[k, 'variance']][0] *
                           (model.endog - model.exog @ coefficients[k]))
            for k in range(3)])
        np.testing.assert_allclose(score[:, :2], 0, atol=1e-10)
        self.assertAlmostEqual(score[:, 2].sum(), 0, places=9)

    def test_shared_and_switching_coefficients_jointly_minimize_q(self):
        rng = np.random.default_rng(20261112)
        n = 120
        x = rng.normal(size=(n, 2))
        y = 0.8 + x @ np.array([0.4, 1.2]) + rng.normal(size=n)
        for switching in ([False, False, False], [True, True, False], [True, True, True]):
            with self.subTest(switching=switching):
                model = JointMeanMarkovRegression(
                    y, k_regimes=3, exog=x, switching_trend=switching[0],
                    switching_exog=switching[1:], switching_variance=True)
                params = model.start_params.copy()
                params[model.parameters['variance']] = [0.05, 0.4, 2.0]
                gamma = rng.dirichlet(np.ones(3), size=n).T
                result = SimpleNamespace(params=params, smoothed_marginal_probabilities=gamma)
                b = model._em_exog(result, y, model.exog, switching)
                score = np.stack([
                    model.exog.T @ ((gamma[k] / [0.05, 0.4, 2.0][k]) *
                                    (y - model.exog @ b[k])) for k in range(3)])
                for j, switches in enumerate(switching):
                    if switches:
                        np.testing.assert_allclose(score[:, j], 0, atol=1e-10)
                    else:
                        self.assertAlmostEqual(score[:, j].sum(), 0, places=9)
                        np.testing.assert_array_equal(b[:, j], np.repeat(b[0, j], 3))


if __name__ == '__main__':
    unittest.main()
