"""Full-sample OLS/HAC reference and uniform Gaussian regime-model starts."""

from dataclasses import asdict, dataclass

import numpy as np
import statsmodels.api as sm


@dataclass(frozen=True)
class OLSReference:
    coefficient_names: list[str]
    coefficients: list[float]
    standard_errors: list[float]
    sigma: float
    sigma_standard_error: float
    nobs: int
    hac_lags: int = 12

    def as_dict(self):
        return asdict(self)


def ols_reference(model):
    """Cache one reference fit on the model's complete effective sample.

    HAC uses Bartlett weights and no small-sample correction, matching
    MeanProcess. The sigma standard error is the delta-method transform of
    the HAC standard error of the mean squared OLS residual. With no exog,
    the model response already equals inflation minus its fixed SPF mean.
    """
    if hasattr(model, "_ols_start_reference"):
        return model._ols_start_reference
    y = np.asarray(model.endog, dtype=float)
    coefficients, standard_errors, names = [], [], []
    residuals = y
    if model.exog is not None and model.exog.shape[1]:
        fit = sm.OLS(y, model.exog).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        coefficients = np.asarray(fit.params).tolist()
        standard_errors = np.asarray(fit.bse).tolist()
        names = list(model.exog_names)
        residuals = np.asarray(fit.resid)
    second_moment = sm.OLS(residuals**2, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 12},
    )
    sigma = float(np.sqrt(second_moment.params[0]))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("OLS initialization requires a positive finite residual RMS")
    sigma_se = float(second_moment.bse[0] / (2 * sigma))
    if (not np.isfinite(sigma_se) or sigma_se < 0
            or not np.all(np.isfinite(coefficients))
            or not np.all(np.isfinite(standard_errors))
            or np.any(np.asarray(standard_errors) < 0)):
        raise ValueError("OLS initialization requires finite estimates and nonnegative standard errors")
    reference = OLSReference(names, coefficients, standard_errors, sigma, sigma_se, len(y))
    model._ols_start_reference = reference
    return reference


def draw_ols_start(model, rng):
    """Draw every free Gaussian parameter, respecting coefficient sharing."""
    reference = ols_reference(model)
    params = np.asarray(model.start_params, dtype=float).copy()
    assigned = set()
    for regime in range(model.k_regimes):
        for feature, index in enumerate(model.parameters[regime, "exog"]):
            if index not in assigned:
                center = reference.coefficients[feature]
                width = 2 * reference.standard_errors[feature]
                params[index] = rng.uniform(center - width, center + width)
                assigned.add(index)
    lower = max(0.0, reference.sigma - 2 * reference.sigma_standard_error)
    upper = reference.sigma + 2 * reference.sigma_standard_error
    for index in range(*model.parameters["variance"].indices(len(params))):
        sigma = rng.uniform(lower, upper)
        while sigma <= 0:  # Exclude the endpoint zero; do not impose a fitted floor.
            sigma = rng.uniform(lower, upper)
        params[index] = sigma**2
    # The notes use row-stochastic P; statsmodels stores P' and omits its last row.
    transition = rng.dirichlet(np.ones(model.k_regimes), size=model.k_regimes)
    params[model.parameters["regime_transition"]] = transition.T[:-1].ravel()
    return params
