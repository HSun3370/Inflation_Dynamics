"""Joint Gaussian mean update for Markov-regression EM initialization.

The final optimizer still maximizes the stationary-initialized observed-data
likelihood. This changes only the regression-coefficient update in the EM
initializer, retaining shared coefficients exactly across regimes.
"""

import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching


class JointMeanMarkovRegression(MarkovRegression):
    def _em_iteration(self, params0):
        result, params1 = MarkovSwitching._em_iteration(self, params0)
        # The raw KimSmootherResults does not retain its parameter vector.
        result.params = np.asarray(params0)
        tmp = np.sqrt(result.smoothed_marginal_probabilities)
        coeffs = None
        if self._k_exog > 0:
            coeffs = self._em_exog(result, self.endog, self.exog,
                                   self.parameters.switching['exog'], tmp)
            for regime in range(self.k_regimes):
                params1[self.parameters[regime, 'exog']] = coeffs[regime]
        params1[self.parameters['variance']] = self._em_variance(
            result, self.endog, self.exog, coeffs, tmp)
        return result, params1

    def _em_exog(self, result, endog, exog, switching, tmp=None):
        """Maximize the Gaussian mean-block Q function at the old variances.

        Minimize sum_{t,k} gamma_{k,t}/sigma_k^2 (y_t-x_t'b_k)^2,
        imposing equality of every non-switching coefficient. A stacked
        weighted least-squares solve avoids normal-equation conditioning.
        """
        switching = np.asarray(switching, dtype=bool)
        if np.all(switching):
            # Each variance is constant within its own independent regression.
            return super()._em_exog(result, endog, exog, switching, tmp)
        k, n_features = self.k_regimes, exog.shape[1]
        indices = np.empty((k, n_features), dtype=int)
        n_coeffs = 0
        for feature in range(n_features):
            if switching[feature]:
                indices[:, feature] = np.arange(n_coeffs, n_coeffs + k)
                n_coeffs += k
            else:
                indices[:, feature] = n_coeffs
                n_coeffs += 1
        variances = np.array([
            result.params[self.parameters[regime, 'variance']][0]
            for regime in range(k)
        ])
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError('EM requires positive finite innovation variances')
        probabilities = np.asarray(result.smoothed_marginal_probabilities)
        # Multiplying all weights by min(variance) leaves the minimizer intact
        # and avoids forming a potentially overflowing inverse variance.
        weights = np.sqrt(np.maximum(probabilities, 0) *
                          (variances.min() / variances)[:, None])
        design = np.zeros((k * len(endog), n_coeffs))
        response = (weights * endog).reshape(-1)
        for regime in range(k):
            block = design[regime * len(endog):(regime + 1) * len(endog)]
            block[:, indices[regime]] = weights[regime, :, None] * exog
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        return coefficients[indices]
