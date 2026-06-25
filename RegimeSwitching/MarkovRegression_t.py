"""Markov switching regression with Student's t innovations."""

from __future__ import annotations

import numpy as np
import statsmodels.base.wrapper as wrap

from scipy.special import loggamma
from statsmodels.tsa.regime_switching import markov_regression

NU_LOWER = 2.05
NU_UPPER = 200.0
NU_GAUSSIAN_CUTOFF = 80.0


class MarkovRegression_t(markov_regression.MarkovRegression):
    """Markov switching regression with optional regime-specific t degrees of freedom."""

    def __init__(
        self,
        endog,
        k_regimes,
        trend="c",
        exog=None,
        order=0,
        exog_tvtp=None,
        switching_trend=True,
        switching_exog=True,
        switching_variance=False,
        switching_nu=False,
        dates=None,
        freq=None,
        missing="none",
    ):
        self.switching_nu = switching_nu

        super().__init__(
            endog=endog,
            k_regimes=k_regimes,
            trend=trend,
            exog=exog,
            order=order,
            exog_tvtp=exog_tvtp,
            switching_trend=switching_trend,
            switching_exog=switching_exog,
            switching_variance=switching_variance,
            dates=dates,
            freq=freq,
            missing=missing,
        )

        self.parameters["nu"] = [1] if self.switching_nu else [0]

    def _conditional_loglikelihoods(self, params):
        resid = self._resid(params)

        # Keep the native dtype so complex-step derivatives are not truncated.
        variance = np.asarray(params[self.parameters["variance"]].squeeze())
        nu = np.asarray(params[self.parameters["nu"]].squeeze())

        if self.switching_variance:
            variance = np.reshape(variance, (self.k_regimes, 1, 1))
        if self.switching_nu:
            nu = np.reshape(nu, (self.k_regimes, 1, 1))

        # Standardized Student's t log-likelihood with variance parameterization
        conditional_loglikelihoods = (
            loggamma((nu + 1.0) / 2.0)
            - loggamma(nu / 2.0)
            - 0.5 * np.log(np.pi * (nu - 2.0))
            - 0.5 * np.log(variance)
            - ((nu + 1.0) / 2.0)
            * np.log(1.0 + (resid**2.0) / (variance * (nu - 2.0)))
        )
        return conditional_loglikelihoods

    @property
    def start_params(self):
        params = markov_regression.MarkovRegression.start_params.fget(self)
        if self.switching_nu:
            params[self.parameters["nu"]] = 4.5 + np.cumsum(np.random.rand(self.k_regimes))
        else:
            params[self.parameters["nu"]] = 6.0
        return params

    def _em_iteration(self, params0):
        result, params1 = markov_regression.MarkovRegression._em_iteration(self, params0)
        # Keep nu fixed during EM; final optimizer updates it.
        params1[self.parameters["nu"]] = params0[self.parameters["nu"]]
        return result, params1

    @property
    def param_names(self):
        param_names = np.array(
            markov_regression.MarkovRegression.param_names.fget(self),
            dtype=object,
        )

        if self.switching_nu:
            for i in range(self.k_regimes):
                param_names[self.parameters[i, "nu"]] = f"nu[{i}]"
        else:
            param_names[self.parameters["nu"]] = "nu"

        return param_names.tolist()

    def transform_params(self, unconstrained):
        constrained = super().transform_params(unconstrained)
        # Bounded map for numerical stability: NU_LOWER < nu < NU_UPPER.
        nu_u = unconstrained[self.parameters["nu"]]
        constrained[self.parameters["nu"]] = NU_LOWER + (NU_UPPER - NU_LOWER) / (1.0 + np.exp(-nu_u))
        return constrained

    def untransform_params(self, constrained):
        unconstrained = super().untransform_params(constrained)
        nu_c = constrained[self.parameters["nu"]]
        eps = 1e-10
        nu_c = np.clip(nu_c, NU_LOWER + eps, NU_UPPER - eps)
        ratio = (nu_c - NU_LOWER) / (NU_UPPER - nu_c)
        unconstrained[self.parameters["nu"]] = np.log(ratio)
        return unconstrained

    @property
    def _res_classes(self):
        return {
            "fit": (
                MarkovRegressiontResults,
                MarkovRegressiontResultsWrapper,
            )
        }


class MarkovRegressiontResults(markov_regression.MarkovRegressionResults):
    pass


class MarkovRegressiontResultsWrapper(markov_regression.MarkovRegressionResultsWrapper):
    pass


wrap.populate_wrapper(MarkovRegressiontResultsWrapper, MarkovRegressiontResults)
