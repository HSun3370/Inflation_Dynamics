#!/usr/bin/env python3
"""Estimate GARCH-family models under multiple mean processes and distributions.

Estimated combinations
- Mean processes: Constant anchor, ARX(1,1), ARX(2,1), ARX(2,2)
- Volatility families: GARCH, GJR-GARCH, EGARCH
- Orders: (1,1), (1,2), (2,1), (2,2)
- Distributions: normal, studentst, mix_normal

Outputs
- CSV summary for all models
- CSV parameters for all models (GJR rows include the negative-shock
  coefficient alpha+gamma with a delta-method standard error)
- CSV diagnostics for combinations with no stable converged fit
- gzipped CSV of every optimizer start (restart audit; not tracked in git)
- TXT line-by-line printout (AIC/BIC/LogLik)
- Markdown report: parameter ranges and random-start draws, then the best
  model by AIC and by BIC

Run modes
- Sequential (default): estimate every combination, then write the outputs.
- Grid: ``--model-id Mxxx --raw-dir DIR`` estimates one combination and
  pickles it to DIR; ``--collect --raw-dir DIR`` merges all combinations and
  writes the outputs. ``submit_garch_grid.sh`` drives both on the CBS Grid.
"""

from __future__ import annotations

# ============================================================
# 1) Imports and runtime environment
# ============================================================
import argparse
import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import trapezoid
from scipy.optimize import nnls

MPL_CACHE = Path(tempfile.gettempdir()) / "inflation_dynamics_mplconfig"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
XDG_CACHE = Path(tempfile.gettempdir()) / "inflation_dynamics_cache"
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

from arch import arch_model
from arch.typing import ArrayLike, ArrayLike1D, Float64Array
from arch.univariate.base import StartingValueWarning
from arch.univariate.distribution import Distribution
from arch.utility.array import AbstractDocStringInheritor, ensure1d
from numpy import asarray
from numpy.random import Generator, RandomState


# ============================================================
# 2) Configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "DataSummary" / "Aggregate_CPI_inflation_Quarterly.pkl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_garch_distributions"

ORDERS = ((1, 1), (1, 2), (2, 1), (2, 2))
DISTRIBUTIONS = ("normal", "studentst", "mix_normal")
VOL_FAMILIES = ("GARCH", "GJR-GARCH", "EGARCH")
COMMON_HOLD_BACK = 0
STABILITY_MARGIN = 1e-6
DEFAULT_N_STARTS = 5000
DEFAULT_RETRY_FAILED_STARTS = 0
DEFAULT_MAXITER = 8000
DEFAULT_SEED = 20260622

# Random-start draw ranges (documented in the report). Every draw lies inside
# arch's own bounds/constraints and the stationary region; parameters arch
# leaves unbounded (mean coefficients, EGARCH alpha/gamma) get the finite
# ranges below.
AR_COLUMNS = ("Inflation_lag_1", "Inflation_lag_2")
EGARCH_ALPHA_DRAW = (-2.0, 2.0)
EGARCH_GAMMA_DRAW = (-2.0, 2.0)
MIX_MIN_COMPONENT_VARIANCE = 1e-6
MAX_DRAW_TRIES = 10_000
LOGLIK_HIT_TOL = 0.01
AT_BOUND_TOL = 1e-6
# Admissibility beyond SLSQP's success flag (see the report's selection notes):
# first-order (KKT) conditions within KKT_TOL_PER_OBS * T, and for the mixture a
# bounded component-variance ratio, which rules out single-observation spikes.
KKT_TOL_PER_OBS = 0.005
MAX_POLISH_ROUNDS = 5
MIX_MIN_VARIANCE_RATIO = 0.01
SUMMARY_SORT_COLUMNS = ["distribution", "mean_spec", "vol_family", "p", "q"]
MIN_SUMMARY_COLUMNS = [
    *SUMMARY_SORT_COLUMNS,
    "optimizer_success",
    "stable_by_proxy",
    "loglikelihood",
    "aic",
    "bic",
]
FAILED_SUMMARY_COLUMNS = [
    "model_id",
    "distribution",
    "distribution_label",
    "mean_spec",
    "mean_label",
    "vol_family",
    "vol_spec",
    "vol_order",
    "attempts",
    "optimizer_success_attempts",
    "stable_attempts",
    "best_attempt_loglikelihood",
    "best_optimizer_success_loglikelihood",
    "last_error_message",
]

DISTRIBUTION_LABEL = {
    "normal": "Normal",
    "studentst": "Student's $t$",
    "mix_normal": "Gaussian mixture",
}
DISTRIBUTION_ORDER = ["normal", "studentst", "mix_normal"]

MEAN_LABEL = {
    "Constant_anchor": "Constant",
    "ARX_1_1": "ARX(1,1)",
    "ARX_2_1": "ARX(2,1)",
    "ARX_2_2": "ARX(2,2)",
}
MEAN_START_SE_MULTIPLIER = 4.0


# ============================================================
# 3) Data structures
# ============================================================
@dataclass(frozen=True)
class MeanSpec:
    name: str
    y_col: str
    y_transform: str  # "identity" or "inflation_minus_spf"
    mean: str
    lags: int | list[int]
    x_cols: list[str]


@dataclass(frozen=True)
class VolSpec:
    family: str
    p: int
    q: int
    o: int
    vol_keyword: str

    @property
    def label(self) -> str:
        return f"{self.family}({self.p},{self.q})"


# ============================================================
# 4) Gaussian mixture distribution for ARCH
# ============================================================
class MixNormal(Distribution, metaclass=AbstractDocStringInheritor):
    """Two-component Gaussian mixture with implied second component."""

    def __init__(
        self,
        random_state: RandomState | None = None,
        *,
        seed: None | int | RandomState | Generator = None,
    ) -> None:
        super().__init__(random_state=random_state, seed=seed)
        self._name = "Two-component Gaussian mixture"
        self.num_params = 3

    @staticmethod
    def _implied_params(parameters: Sequence[float] | ArrayLike1D):
        params = asarray(parameters, dtype=float)
        p1, mu1, sigma1_sq = params
        p2 = 1.0 - p1
        if p2 <= 0.0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        mu2 = -p1 * mu1 / p2
        sigma2_sq = (1.0 - p1 * (sigma1_sq + mu1 * mu1) - p2 * mu2 * mu2) / p2
        return p1, p2, mu1, mu2, sigma1_sq, sigma2_sq

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return np.empty((0, 3)), np.empty(0)

    def bounds(self, resids: Float64Array) -> list[tuple[float, float]]:
        return [(1e-4, 1 - 1e-4), (-50.0, 50.0), (1e-6, 100.0)]

    def loglikelihood(
        self,
        parameters: Sequence[float] | ArrayLike1D,
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> float | Float64Array:
        p1, p2, mu1, mu2, sigma1_sq, sigma2_sq = self._implied_params(parameters)
        r = np.asarray(resids, dtype=float)
        v = np.asarray(sigma2, dtype=float)

        if (
            not np.isfinite(p1)
            or sigma1_sq <= 0.0
            or not np.isfinite(sigma2_sq)
            or sigma2_sq <= 0.0
            or np.any(v <= 0.0)
        ):
            penalty = np.full_like(r, -1e12, dtype=float)
            return penalty if individual else float(np.sum(penalty))

        z = r / np.sqrt(v)
        pdf1 = p1 * stats.norm.pdf(z, loc=mu1, scale=np.sqrt(sigma1_sq))
        pdf2 = p2 * stats.norm.pdf(z, loc=mu2, scale=np.sqrt(sigma2_sq))
        mix_pdf = np.maximum(pdf1 + pdf2, 1e-300)

        lls = -0.5 * np.log(v) + np.log(mix_pdf)
        return lls if individual else float(np.sum(lls))

    def starting_values(self, std_resid: Float64Array) -> Float64Array:
        return np.array([0.3, 0.1, 1.0], dtype=float)

    def _simulator(self, size: int | tuple[int, ...]) -> Float64Array:
        assert self._parameters is not None
        p1, p2, mu1, mu2, sigma1_sq, sigma2_sq = self._implied_params(self._parameters)
        draws = self._generator.uniform(size=size)
        z1 = self._generator.normal(loc=mu1, scale=np.sqrt(sigma1_sq), size=size)
        z2 = self._generator.normal(loc=mu2, scale=np.sqrt(sigma2_sq), size=size)
        return np.where(draws <= p1, z1, z2)

    def simulate(
        self, parameters: int | float | Sequence[float | int] | ArrayLike1D
    ) -> Callable[[int | tuple[int, ...]], Float64Array]:
        params = ensure1d(parameters, "parameters", False)
        self._parameters = asarray(params, dtype=float)
        return self._simulator

    def parameter_names(self) -> list[str]:
        return ["p_1", "mu_1", "sigma_1_sq"]

    def cdf(
        self,
        resids: Sequence[float] | ArrayLike1D,
        parameters: None | Sequence[float] | ArrayLike1D = None,
    ) -> Float64Array:
        params = self._check_constraints(parameters)
        p1, p2, mu1, mu2, sigma1_sq, sigma2_sq = self._implied_params(params)
        x = np.asarray(resids, dtype=float)
        return p1 * stats.norm.cdf(x, loc=mu1, scale=np.sqrt(sigma1_sq)) + p2 * stats.norm.cdf(
            x, loc=mu2, scale=np.sqrt(sigma2_sq)
        )

    def ppf(
        self,
        pits: float | Sequence[float] | ArrayLike1D,
        parameters: None | Sequence[float] | ArrayLike1D = None,
    ) -> Float64Array:
        params = self._check_constraints(parameters)
        q = np.asarray(pits, dtype=float)
        scalar = np.isscalar(pits)
        if scalar:
            q = np.array([float(pits)], dtype=float)

        p1, p2, mu1, mu2, sigma1_sq, sigma2_sq = self._implied_params(params)
        lo = min(mu1 - 10 * np.sqrt(sigma1_sq), mu2 - 10 * np.sqrt(sigma2_sq))
        hi = max(mu1 + 10 * np.sqrt(sigma1_sq), mu2 + 10 * np.sqrt(sigma2_sq))

        grid = np.linspace(lo, hi, 20001)
        cvals = p1 * stats.norm.cdf(grid, loc=mu1, scale=np.sqrt(sigma1_sq)) + p2 * stats.norm.cdf(
            grid, loc=mu2, scale=np.sqrt(sigma2_sq)
        )
        out = np.interp(q, cvals, grid)
        return out[0] if scalar else out

    def moment(
        self, n: int, parameters: None | Sequence[float] | ArrayLike1D = None
    ) -> float:
        if n < 0:
            return float("nan")
        params = self._check_constraints(parameters)
        p1, p2, mu1, mu2, sigma1_sq, sigma2_sq = self._implied_params(params)
        m1 = stats.norm.moment(n, loc=mu1, scale=np.sqrt(sigma1_sq))
        m2 = stats.norm.moment(n, loc=mu2, scale=np.sqrt(sigma2_sq))
        return float(p1 * m1 + p2 * m2)

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: None | Sequence[float] | ArrayLike1D = None,
    ) -> float:
        if n < 0:
            return float("nan")
        params = self._check_constraints(parameters)
        p1, p2, mu1, mu2, sigma1_sq, sigma2_sq = self._implied_params(params)

        if n == 0:
            return float(
                p1 * stats.norm.cdf(z, loc=mu1, scale=np.sqrt(sigma1_sq))
                + p2 * stats.norm.cdf(z, loc=mu2, scale=np.sqrt(sigma2_sq))
            )

        x = np.linspace(-20.0, z, 4001)
        pdf = p1 * stats.norm.pdf(x, loc=mu1, scale=np.sqrt(sigma1_sq)) + p2 * stats.norm.pdf(
            x, loc=mu2, scale=np.sqrt(sigma2_sq)
        )
        return float(trapezoid((x**n) * pdf, x))


# ============================================================
# 5) Data loading and model specification builders
# ============================================================
def load_quarterly_data(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    required = {
        "Inflation",
        "Inflation_lag_1",
        "Inflation_lag_2",
        "SPF",
        "SPF_lag_1",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")
    return df.copy()


def build_mean_specs() -> list[MeanSpec]:
    return [
        MeanSpec(
            name="Constant_anchor",
            y_col="Inflation",
            y_transform="inflation_minus_spf",
            mean="Zero",
            lags=0,
            x_cols=[],
        ),
        MeanSpec(
            name="ARX_1_1",
            y_col="Inflation",
            y_transform="identity",
            mean="LS",
            lags=0,
            x_cols=["Inflation_lag_1", "SPF"],
        ),
        MeanSpec(
            name="ARX_2_1",
            y_col="Inflation",
            y_transform="identity",
            mean="LS",
            lags=0,
            x_cols=["Inflation_lag_1", "Inflation_lag_2", "SPF"],
        ),
        MeanSpec(
            name="ARX_2_2",
            y_col="Inflation",
            y_transform="identity",
            mean="LS",
            lags=0,
            x_cols=["Inflation_lag_1", "Inflation_lag_2", "SPF", "SPF_lag_1"],
        ),
    ]


def build_vol_specs() -> list[VolSpec]:
    specs: list[VolSpec] = []
    for p, q in ORDERS:
        specs.append(VolSpec(family="GARCH", p=p, q=q, o=0, vol_keyword="GARCH"))
        specs.append(VolSpec(family="GJR-GARCH", p=p, q=q, o=p, vol_keyword="GARCH"))
        specs.append(VolSpec(family="EGARCH", p=p, q=q, o=p, vol_keyword="EGARCH"))
    return specs


def prepare_mean_inputs(
    df: pd.DataFrame, mean_spec: MeanSpec
) -> tuple[pd.Series, pd.DataFrame | None]:
    if mean_spec.y_transform == "inflation_minus_spf":
        y = df["Inflation"] - df["SPF"]
    elif mean_spec.y_transform == "identity":
        y = df[mean_spec.y_col]
    else:
        raise ValueError(f"Unsupported y_transform: {mean_spec.y_transform}")

    x = df[mean_spec.x_cols].copy() if mean_spec.x_cols else None
    return y.astype(float), x


def ols_with_se(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and homoskedastic standard errors."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(int(len(y) - design.shape[1]), 1)
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return beta, np.sqrt(np.maximum(np.diag(cov), 0.0))


def ar_max_root_modulus(rho: Sequence[float]) -> float:
    """Largest modulus of the roots of z^k - rho_1 z^{k-1} - ... - rho_k; the AR
    part of the mean process is stationary when this is below one."""
    rho = [float(r) for r in rho]
    if not rho:
        return 0.0
    return float(np.max(np.abs(np.roots([1.0, *[-r for r in rho]]))))


def mean_ar_coefficients(params: pd.Series) -> list[float]:
    """Inflation-lag coefficients only (SPF terms are exogenous)."""
    return [float(params[c]) for c in AR_COLUMNS if c in params.index]


def draw_stationary_ar(k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform draw over the AR(k) stationarity region (k <= 2): (-1, 1) for
    AR(1); the triangle |rho_1| < 1 - rho_2, rho_2 > -1 for AR(2)."""
    box = np.array([2.0, 1.0]) if k == 2 else np.ones(k)  # bounding box of the region
    for _ in range(MAX_DRAW_TRIES):
        rho = rng.uniform(-box, box)
        if ar_max_root_modulus(rho) < 1.0 - STABILITY_MARGIN:
            return rho
    raise RuntimeError(f"Could not draw a stationary AR({k}) vector")


# ============================================================
# 6) Unconditional variance helper logic
# ============================================================
def _sum_params(params: pd.Series, prefix: str, count: int) -> float:
    total = 0.0
    for i in range(1, count + 1):
        total += float(params.get(f"{prefix}[{i}]", 0.0))
    return total


def implied_initial_variance(
    params: pd.Series,
    vol_spec: VolSpec,
) -> tuple[float, float, str]:
    """Return implied initial variance and persistence proxy.

    For GARCH: omega / (1 - sum alpha - sum beta)
    For GJR-GARCH: omega / (1 - sum alpha - 0.5*sum gamma - sum beta)
      (under symmetric innovations)
    For EGARCH: exp( omega / (1 - sum beta) )
      (proxy based on E[log sigma^2], with arch-package centering convention)
    """
    omega = float(params.get("omega", np.nan))
    beta_sum = _sum_params(params, "beta", vol_spec.q)

    if vol_spec.family == "GARCH":
        alpha_sum = _sum_params(params, "alpha", vol_spec.p)
        persistence = alpha_sum + beta_sum
        if (not np.isfinite(omega)) or persistence >= 1.0:
            return np.nan, persistence, "undefined_or_nonstationary"
        return omega / (1.0 - persistence), persistence, "exact"

    if vol_spec.family == "GJR-GARCH":
        alpha_sum = _sum_params(params, "alpha", vol_spec.p)
        gamma_sum = _sum_params(params, "gamma", vol_spec.o)
        persistence = alpha_sum + 0.5 * gamma_sum + beta_sum
        if (not np.isfinite(omega)) or persistence >= 1.0:
            return np.nan, persistence, "undefined_or_nonstationary"
        return omega / (1.0 - persistence), persistence, "exact_under_symmetry"

    # EGARCH proxy
    persistence = beta_sum
    if (not np.isfinite(omega)) or persistence >= 1.0:
        return np.nan, persistence, "proxy_undefined_or_nonstationary"
    return float(np.exp(omega / (1.0 - persistence))), persistence, "proxy_from_log_variance"


# ============================================================
# 7) Fit wrappers (ARCH and SPARCH-style mix-normal)
# ============================================================
def fit_rejecting_replaced_starts(model: Any, kwargs: dict[str, Any]) -> Any:
    """Fit, but fail instead of letting arch silently swap starting values that
    violate its bounds/constraints for its own default start."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=StartingValueWarning)
        result = model.fit(**kwargs)
    if any(issubclass(w.category, StartingValueWarning) for w in caught):
        raise RuntimeError("arch rejected the starting values (outside its bounds/constraints)")
    return result


def make_arch_fit_func(
    y: pd.Series,
    x: pd.DataFrame | None,
    mean: str,
    lags: int | list[int],
    vol_spec: VolSpec,
    dist: str,
    hold_back: int,
    maxiter: int,
):
    am = arch_model(
        y=y,
        x=x,
        mean=mean,
        lags=lags,
        vol=vol_spec.vol_keyword,
        p=vol_spec.p,
        o=vol_spec.o,
        q=vol_spec.q,
        dist=dist,
        hold_back=hold_back,
        rescale=False,
    )

    def _fit(starting_values: np.ndarray | None):
        kwargs: dict[str, Any] = {
            "disp": "off",
            "update_freq": 0,
            "cov_type": "robust",
            "show_warning": False,
            "options": {"maxiter": maxiter},
        }
        if starting_values is not None:
            kwargs["starting_values"] = starting_values
        return fit_rejecting_replaced_starts(am, kwargs)

    return _fit, am


def make_sparch_fit_func(
    y: pd.Series,
    x: pd.DataFrame | None,
    mean: str,
    lags: int | list[int],
    vol_spec: VolSpec,
    hold_back: int,
    maxiter: int,
):
    benchmark_model = arch_model(
        y=y,
        x=x,
        mean=mean,
        lags=lags,
        vol=vol_spec.vol_keyword,
        p=vol_spec.p,
        o=vol_spec.o,
        q=vol_spec.q,
        dist="studentst",
        hold_back=hold_back,
        rescale=False,
    )
    benchmark = benchmark_model.fit(
        disp="off",
        update_freq=0,
        cov_type="robust",
        show_warning=False,
        options={"maxiter": maxiter},
    )

    initial = np.concatenate(
        [np.asarray(benchmark.params, dtype=float)[:-1], np.array([0.3, 0.1, 1.0])]
    )

    mix_model = arch_model(
        y=y,
        x=x,
        mean=mean,
        lags=lags,
        vol=vol_spec.vol_keyword,
        p=vol_spec.p,
        o=vol_spec.o,
        q=vol_spec.q,
        hold_back=hold_back,
        rescale=False,
    )
    mix_model.distribution = MixNormal()

    def _fit(starting_values: np.ndarray | None):
        kwargs: dict[str, Any] = {
            "disp": "off",
            "update_freq": 0,
            "cov_type": "robust",
            "show_warning": False,
            "options": {"maxiter": maxiter},
        }
        if starting_values is not None:
            kwargs["starting_values"] = starting_values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            return fit_rejecting_replaced_starts(mix_model, kwargs)

    return _fit, initial, mix_model


# ============================================================
# 8) Feasible set, random starting values, and restart search
# ============================================================
def optimizer_converged(result: Any) -> bool:
    return bool(result.optimization_result.success) and int(result.convergence_flag) == 0


def model_checks(result: Any, vol_spec: VolSpec) -> dict[str, Any]:
    """Convergence, volatility stationarity, and mean-process AR stationarity."""
    unc_var, persistence, unc_status = implied_initial_variance(result.params, vol_spec)
    vol_stationary = bool(np.isfinite(persistence) and persistence < 1.0 - STABILITY_MARGIN)
    ar_root = ar_max_root_modulus(mean_ar_coefficients(result.params))
    mean_stationary = bool(ar_root < 1.0 - STABILITY_MARGIN)
    converged = optimizer_converged(result)
    ratio = mixture_variance_ratio(result.params)
    mixture_ok = bool(not np.isfinite(ratio) or ratio >= MIX_MIN_VARIANCE_RATIO)
    return {
        "unc_var": float(unc_var),
        "persistence": float(persistence),
        "unc_status": unc_status,
        "vol_stationary": vol_stationary,
        "ar_max_root_modulus": ar_root,
        "mean_stationary": mean_stationary,
        "mixture_variance_ratio": ratio,
        "mixture_admissible": mixture_ok,
        "converged": converged,
        "passed": converged and vol_stationary and mean_stationary and mixture_ok,
    }


def mixture_variance_ratio(params: pd.Series) -> float:
    """min/max of the two standardized component variances (NaN without a mixture)."""
    if "p_1" not in params.index:
        return np.nan
    p1, mu1, s1 = float(params["p_1"]), float(params["mu_1"]), float(params["sigma_1_sq"])
    p2 = 1.0 - p1
    s2 = (1.0 - p1 * s1 - p1 * mu1 * mu1 / p2) / p2
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0
    return min(s1, s2) / max(s1, s2)


def first_order_violation(model: Any, feasible: FeasibleSet, x: np.ndarray) -> float:
    """Largest KKT residual of the log likelihood at x.

    The gradient (central differences; one-sided at an active bound) must be
    offset by non-negative multipliers on the active bounds and linear
    constraints; the residual of that non-negative least-squares fit is zero at
    a genuine constrained maximum. SLSQP sometimes reports success at points
    where it is large."""
    x = np.asarray(x, dtype=float)
    resids0 = np.asarray(model.resids(model.starting_values()), dtype=float)
    vol = model.volatility
    backcast, var_bounds = vol.backcast(resids0), vol.variance_bounds(resids0)
    sigma2 = np.zeros_like(resids0)

    def loglik(z: np.ndarray) -> float:
        return -float(model._loglikelihood(z, sigma2, backcast, var_bounds))

    n = len(x)
    grad = np.zeros(n)
    active: list[np.ndarray] = []
    base = loglik(x)
    for i, (lo, hi) in enumerate(feasible.bounds):
        h = 1e-6 * max(abs(x[i]), 1e-2)
        step = np.zeros(n)
        step[i] = h
        tol = AT_BOUND_TOL * max(1.0, abs(x[i]))
        if np.isfinite(lo) and x[i] - lo <= tol:
            grad[i] = (loglik(x + step) - base) / h
            active.append(np.eye(n)[i])
        elif np.isfinite(hi) and hi - x[i] <= tol:
            grad[i] = (base - loglik(x - step)) / h
            active.append(-np.eye(n)[i])
        else:
            grad[i] = (loglik(x + step) - loglik(x - step)) / (2.0 * h)
    for a_row, b_val in zip(feasible.a, feasible.b):
        if a_row @ x - b_val <= AT_BOUND_TOL:
            active.append(a_row)
    if active:
        jac = np.array(active).T
        lam, _ = nnls(jac, -grad)
        grad = grad + jac @ lam
    return float(np.max(np.abs(grad)))


@dataclass(frozen=True)
class FeasibleSet:
    """Parameter bounds and linear constraints (a @ x - b >= 0) imposed by arch's fit."""

    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    a: np.ndarray
    b: np.ndarray
    resid_scale: float  # mean squared residual at arch's mean starting values (v-bar)

    def contains(self, x: np.ndarray) -> bool:
        if not all(lo <= v <= hi for v, (lo, hi) in zip(x, self.bounds)):
            return False
        return bool(np.all(self.a @ x - self.b >= 0.0)) if self.a.size else True

    def bound(self, name: str) -> tuple[float, float]:
        return self.bounds[self.names.index(name)]

    def at_bound(self, params: pd.Series) -> list[str]:
        """Parameters of an estimate that sit on one of arch's bounds."""
        out = []
        for name, (lo, hi) in zip(self.names, self.bounds):
            v = float(params[name])
            tol = AT_BOUND_TOL * max(1.0, abs(v))
            if (np.isfinite(lo) and v - lo <= tol) or (np.isfinite(hi) and hi - v <= tol):
                out.append(name)
        return out


def arch_feasible_set(model: Any) -> FeasibleSet:
    """Replicate how arch's ARCHModel.fit assembles bounds and constraints.

    arch replaces any starting vector outside this set with its default start,
    so every random start is drawn inside it."""
    model._adjust_sample(None, None)  # what fit() does before computing residuals
    resids = np.asarray(model.resids(model.starting_values()), dtype=float)
    vol, dist = model.volatility, model.distribution
    # fit() passes standardized residuals to dist.bounds; the bounds of the
    # normal, Student's t, and MixNormal distributions do not depend on them.
    bounds = model.bounds() + vol.bounds(resids) + dist.bounds(resids)
    blocks = [model.constraints(), vol.constraints(), dist.constraints()]
    sizes = [model.num_params, vol.num_params, dist.num_params]
    a = np.zeros((sum(c[0].shape[0] for c in blocks), sum(sizes)))
    b = np.zeros(a.shape[0])
    row = col = 0
    for (block_a, block_b), size in zip(blocks, sizes):
        a[row : row + block_a.shape[0], col : col + size] = block_a
        b[row : row + block_a.shape[0]] = block_b
        row += block_a.shape[0]
        col += size
    names = model.parameter_names() + vol.parameter_names() + dist.parameter_names()
    return FeasibleSet(tuple(names), tuple(bounds), a, b, float(np.mean(resids**2)))


class StartDrawer:
    """Random starting vectors inside arch's feasible set and the stationary region.

    Mean: AR coefficients uniform over the stationarity region; intercept and
    SPF loadings uniform over the conditional OLS estimate (given the drawn AR
    coefficients) +/- MEAN_START_SE_MULTIPLIER standard errors.
    Volatility: omega over its full arch bound (log-uniform for GARCH/GJR,
    uniform for EGARCH's log-scale omega); ARCH/GARCH loadings uniform over
    the stationary simplex; EGARCH alpha/gamma uniform on fixed ranges.
    Distribution: nu log-uniform on its arch bound; mixture parameters uniform
    over the values that keep both component variances positive.
    """

    def __init__(
        self,
        feasible: FeasibleSet,
        vol_spec: VolSpec,
        dist: str,
        y: pd.Series,
        x: pd.DataFrame | None,
    ) -> None:
        self.fs = feasible
        self.vol_spec = vol_spec
        self.dist = dist
        self.idx = {name: i for i, name in enumerate(feasible.names)}
        self.y = np.asarray(y, dtype=float)
        cols = list(x.columns) if x is not None else []
        self.ar_cols = [c for c in AR_COLUMNS if c in cols]
        self.other_cols = [c for c in cols if c not in self.ar_cols]
        self.lags = x[self.ar_cols].to_numpy(dtype=float) if self.ar_cols else None
        self.design = (
            np.column_stack([np.ones(len(self.y)), x[self.other_cols].to_numpy(dtype=float)])
            if x is not None
            else None
        )
        self.rejected = 0

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(MAX_DRAW_TRIES):
            sv = np.full(len(self.fs.names), np.nan)
            self._draw_mean(sv, rng)
            self._draw_volatility(sv, rng)
            self._draw_distribution(sv, rng)
            if np.all(np.isfinite(sv)) and self.fs.contains(sv) and self._stationary(sv):
                return sv
            self.rejected += 1
        raise RuntimeError(f"No admissible random start after {MAX_DRAW_TRIES} draws")

    def _set(self, sv: np.ndarray, name: str, value: float) -> None:
        sv[self.idx[name]] = value

    def _stationary(self, sv: np.ndarray) -> bool:
        params = pd.Series(sv, index=list(self.fs.names))
        _, persistence, _ = implied_initial_variance(params, self.vol_spec)
        return bool(np.isfinite(persistence) and persistence < 1.0 - STABILITY_MARGIN)

    def _draw_mean(self, sv: np.ndarray, rng: np.random.Generator) -> None:
        if self.design is None:  # Constant anchor: no mean parameters
            return
        rho = draw_stationary_ar(len(self.ar_cols), rng)
        target = self.y - (self.lags @ rho if self.lags is not None else 0.0)
        coef, se = ols_with_se(self.design, target)
        half = MEAN_START_SE_MULTIPLIER * se
        for name, c, h in zip(["Const", *self.other_cols], coef, half):
            self._set(sv, name, float(rng.uniform(c - h, c + h)))
        for name, r in zip(self.ar_cols, rho):
            self._set(sv, name, float(r))

    def _draw_volatility(self, sv: np.ndarray, rng: np.random.Generator) -> None:
        vs = self.vol_spec
        lo, hi = self.fs.bound("omega")
        if vs.family == "EGARCH":
            self._set(sv, "omega", float(rng.uniform(lo, hi)))
            for i in range(1, vs.p + 1):
                self._set(sv, f"alpha[{i}]", float(rng.uniform(*EGARCH_ALPHA_DRAW)))
            for j in range(1, vs.o + 1):
                self._set(sv, f"gamma[{j}]", float(rng.uniform(*EGARCH_GAMMA_DRAW)))
            betas = rng.dirichlet(np.ones(vs.q + 1))[: vs.q]
        else:
            self._set(sv, "omega", float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
            if vs.family == "GJR-GARCH":
                # Uniform over {alpha_i/2, gamma_i/2, beta_k >= 0, sum < 1} in README
                # notation (alpha_i on (u+)^2, gamma_i on (u-)^2); arch's own
                # gamma is gamma_i - alpha_i, and arch caps alpha_i at 1.
                w = rng.dirichlet(np.ones(2 * vs.p + vs.q + 1))
                for i in range(1, vs.p + 1):
                    alpha = 2.0 * float(w[i - 1])
                    neg = 2.0 * float(w[vs.p + i - 1])
                    self._set(sv, f"alpha[{i}]", alpha)
                    self._set(sv, f"gamma[{i}]", neg - alpha)
                betas = w[2 * vs.p : 2 * vs.p + vs.q]
            else:
                w = rng.dirichlet(np.ones(vs.p + vs.q + 1))
                for i in range(1, vs.p + 1):
                    self._set(sv, f"alpha[{i}]", float(w[i - 1]))
                betas = w[vs.p : vs.p + vs.q]
        for k, beta in enumerate(betas, start=1):
            self._set(sv, f"beta[{k}]", float(beta))

    def _draw_distribution(self, sv: np.ndarray, rng: np.random.Generator) -> None:
        if self.dist == "studentst":
            lo, hi = self.fs.bound("nu")
            self._set(sv, "nu", float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        elif self.dist == "mix_normal":
            s_min = MIX_MIN_COMPONENT_VARIANCE
            p1 = float(rng.uniform(*self.fs.bound("p_1")))
            p2 = 1.0 - p1
            mu_lo, mu_hi = self.fs.bound("mu_1")
            mu_max = min(mu_hi, -mu_lo, np.sqrt(p2 * (1.0 - s_min) / p1))
            mu1 = float(rng.uniform(-mu_max, mu_max))
            s_lo, s_hi = self.fs.bound("sigma_1_sq")
            # sigma_2^2 = (1 - p1 sigma_1^2 - p1 mu1^2 / p2) / p2 >= s_min
            s_max = min(s_hi, (1.0 - p1 * mu1 * mu1 / p2 - p2 * s_min) / p1)
            self._set(sv, "p_1", p1)
            self._set(sv, "mu_1", mu1)
            self._set(sv, "sigma_1_sq", float(rng.uniform(s_lo, s_max)) if s_max > s_lo else np.nan)


def fit_with_restarts(
    fit_func: Callable[[np.ndarray | None], Any],
    model: Any,
    drawer: StartDrawer,
    vol_spec: VolSpec,
    rng: np.random.Generator,
    n_starts: int,
    initial_starting_values: np.ndarray | None = None,
    phase: str = "primary",
    attempt_offset: int = 0,
) -> tuple[Any | None, list[dict[str, Any]], str]:
    """Attempt 1 uses arch's default start (or the supplied initial vector);
    the rest use random draws. Only the best results are kept in memory.

    A start that would become the new best (admissible or fallback) must also
    satisfy the first-order conditions; the KKT residual is computed only then,
    which suffices because every unchecked start has a lower log likelihood
    than a checked one. If SLSQP stopped short of the conditions, it is
    restarted from its own solution (up to MAX_POLISH_ROUNDS times); points
    on steep ridges do not move and stay rejected."""
    attempts: list[dict[str, Any]] = []
    best_valid: Any | None = None
    best_fallback: Any | None = None
    fs = drawer.fs
    kkt_tol = KKT_TOL_PER_OBS * len(drawer.y)
    lower = np.array([lo for lo, _ in fs.bounds])
    upper = np.array([hi for _, hi in fs.bounds])

    def verify_first_order(result: Any) -> tuple[Any, float, bool, int]:
        violation = first_order_violation(model, fs, np.asarray(result.params, dtype=float))
        rounds = 0
        while violation > kkt_tol and rounds < MAX_POLISH_ROUNDS:
            x = np.clip(np.asarray(result.params, dtype=float), lower, upper)
            if not fs.contains(x):
                break
            polished = fit_func(x)
            rounds += 1
            if not (
                optimizer_converged(polished)
                and float(polished.loglikelihood) >= float(result.loglikelihood) - 1e-8
            ):
                break
            result = polished
            violation = first_order_violation(model, fs, np.asarray(result.params, dtype=float))
        return result, violation, bool(violation <= kkt_tol), rounds

    for attempt_idx in range(max(1, n_starts)):
        attempt_number = attempt_offset + attempt_idx + 1
        if attempt_idx == 0:
            sv = initial_starting_values
            label = "default" if sv is None else "initial"
        else:
            sv = drawer.draw(rng)
            label = f"random_{attempt_number}"
        try:
            result = fit_func(sv)
            checks = model_checks(result, vol_spec)
            loglik = float(result.loglikelihood)
            kkt_violation, kkt_ok, polish_rounds = np.nan, True, 0

            def improves(res: Any, passed: bool) -> bool:
                incumbent = best_valid if passed else best_fallback
                return incumbent is None or float(res.loglikelihood) > float(incumbent.loglikelihood)

            if checks["converged"] and np.isfinite(loglik) and improves(result, checks["passed"]):
                result, kkt_violation, kkt_ok, polish_rounds = verify_first_order(result)
                checks = model_checks(result, vol_spec)
                loglik = float(result.loglikelihood)
                if kkt_ok and improves(result, checks["passed"]):
                    if checks["passed"]:
                        best_valid = result
                    else:
                        best_fallback = result
            attempts.append(
                {
                    "phase": phase,
                    "attempt": attempt_number,
                    "start_type": label,
                    "loglikelihood": loglik,
                    "optimizer_success": bool(result.optimization_result.success),
                    "convergence_flag": int(result.convergence_flag),
                    "stable_by_proxy": checks["vol_stationary"],
                    "persistence_proxy": checks["persistence"],
                    "mean_stationary": checks["mean_stationary"],
                    "ar_max_root_modulus": checks["ar_max_root_modulus"],
                    "mixture_variance_ratio": checks["mixture_variance_ratio"],
                    "mixture_admissible": checks["mixture_admissible"],
                    "kkt_violation": kkt_violation,
                    "polish_rounds": polish_rounds,
                    "passed_checks": bool(checks["passed"] and kkt_ok),
                    "implied_initial_variance": checks["unc_var"],
                    "implied_initial_variance_status": checks["unc_status"],
                    "error_message": "",
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "phase": phase,
                    "attempt": attempt_number,
                    "start_type": label,
                    "loglikelihood": np.nan,
                    "optimizer_success": False,
                    "convergence_flag": np.nan,
                    "stable_by_proxy": False,
                    "persistence_proxy": np.nan,
                    "mean_stationary": False,
                    "ar_max_root_modulus": np.nan,
                    "mixture_variance_ratio": np.nan,
                    "mixture_admissible": False,
                    "kkt_violation": np.nan,
                    "polish_rounds": 0,
                    "passed_checks": False,
                    "implied_initial_variance": np.nan,
                    "implied_initial_variance_status": "fit_failed",
                    "error_message": str(exc),
                }
            )

    if best_valid is not None:
        return best_valid, attempts, "passed_checks"
    if best_fallback is not None:
        return best_fallback, attempts, "fallback_best_converged_no_checks_passed"
    return None, attempts, "failed_no_converged_attempt"


def model_phase_rng(seed: int, model_counter: int, phase_id: int) -> np.random.Generator:
    """Create an independent reproducible RNG for one model and search phase."""
    return np.random.default_rng(np.random.SeedSequence([seed, model_counter, phase_id]))


# ============================================================
# 9) Output row builders
# ============================================================
def residual_diagnostics(result: Any) -> dict[str, float]:
    resid = pd.Series(np.asarray(result.resid, dtype=float)).dropna()
    std_resid = pd.Series(np.asarray(result.std_resid, dtype=float)).dropna()

    out = {
        "resid_skewness": np.nan,
        "resid_excess_kurtosis": np.nan,
        "std_resid_skewness": np.nan,
        "std_resid_excess_kurtosis": np.nan,
    }
    if len(resid) > 3:
        out["resid_skewness"] = float(stats.skew(resid, bias=False))
        out["resid_excess_kurtosis"] = float(stats.kurtosis(resid, fisher=True, bias=False))
    if len(std_resid) > 3:
        out["std_resid_skewness"] = float(stats.skew(std_resid, bias=False))
        out["std_resid_excess_kurtosis"] = float(
            stats.kurtosis(std_resid, fisher=True, bias=False)
        )
    return out


def summary_row(
    model_id: str,
    distribution: str,
    mean_spec: MeanSpec,
    vol_spec: VolSpec,
    result: Any,
    checks: dict[str, Any],
    attempts: list[dict[str, Any]],
    selection_status: str,
    feasible: FeasibleSet,
    drawer: StartDrawer,
    n_starts: int,
    kkt_violation: float,
) -> dict[str, Any]:
    diag = residual_diagnostics(result)
    kkt_tol = KKT_TOL_PER_OBS * len(drawer.y)
    loglik = float(result.loglikelihood)
    valid_lls = [
        float(a["loglikelihood"]) for a in attempts if bool(a.get("passed_checks"))
    ]
    # Converged, first-order-valid fits that fail stationarity or the mixture rule.
    inadmissible_lls = [
        float(a["loglikelihood"])
        for a in attempts
        if not bool(a.get("passed_checks")) and float(a.get("kkt_violation", np.nan)) <= kkt_tol
    ]
    omega_lo, omega_hi = feasible.bound("omega")
    return {
        "model_id": model_id,
        "distribution": distribution,
        "distribution_label": DISTRIBUTION_LABEL[distribution],
        "mean_spec": mean_spec.name,
        "mean_label": MEAN_LABEL[mean_spec.name],
        "vol_family": vol_spec.family,
        "vol_spec": vol_spec.label,
        "vol_order": f"({vol_spec.p},{vol_spec.q})",
        "p": vol_spec.p,
        "o": vol_spec.o,
        "q": vol_spec.q,
        "nobs": int(result.nobs),
        "num_parameters": int(len(result.params)),
        "loglikelihood": loglik,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "persistence_proxy": checks["persistence"],
        "stable_by_proxy": checks["vol_stationary"],
        "ar_max_root_modulus": checks["ar_max_root_modulus"],
        "mean_stationary": checks["mean_stationary"],
        "mixture_variance_ratio": checks["mixture_variance_ratio"],
        "mixture_admissible": checks["mixture_admissible"],
        "kkt_violation": kkt_violation,
        "kkt_tolerance": kkt_tol,
        "implied_initial_variance": checks["unc_var"],
        "implied_initial_variance_status": checks["unc_status"],
        "initialization_method": "arch_default",
        "n_starts_requested": int(n_starts),
        "restart_attempts": len(attempts),
        "optimizer_success_attempts": attempt_count(attempts, "optimizer_success"),
        "valid_restart_attempts": len(valid_lls),
        "best_loglik_hits": int(sum(ll >= loglik - LOGLIK_HIT_TOL for ll in valid_lls)),
        "best_inadmissible_loglikelihood": max(inadmissible_lls) if inadmissible_lls else np.nan,
        "start_draw_rejections": int(drawer.rejected),
        "params_at_arch_bound": ";".join(feasible.at_bound(result.params)),
        "resid_scale": feasible.resid_scale,
        "omega_lower_bound": omega_lo,
        "omega_upper_bound": omega_hi,
        "optimizer_success": bool(result.optimization_result.success),
        "convergence_flag": int(result.convergence_flag),
        "covariance_type": "robust",
        "selection_status": selection_status,
        "selected_checks_passed": bool(checks["passed"] and kkt_violation <= kkt_tol),
        **diag,
    }


def parameter_rows(model_id: str, result: Any, vol_spec: VolSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in result.params.index:
        rows.append(
            {
                "model_id": model_id,
                "parameter": str(name),
                "coef": float(result.params[name]),
                "std_err": float(result.std_err[name]),
                "t_value": float(result.tvalues[name]),
                "p_value": float(result.pvalues[name]),
                "derived_from": "",
            }
        )
    if vol_spec.family == "GJR-GARCH":
        rows.extend(gjr_negative_shock_rows(model_id, result, vol_spec))
    return rows


def gjr_negative_shock_rows(model_id: str, result: Any, vol_spec: VolSpec) -> list[dict[str, Any]]:
    """README-notation negative-shock coefficient gamma_i = alpha_i + gamma_i^arch.

    arch estimates sigma^2_t = omega + alpha u^2 + gamma^arch u^2 1(u<0) + ...,
    so the coefficient on (u^-)^2 is alpha + gamma^arch. Its standard error uses
    the delta method on arch's (robust) parameter covariance:
    Var = Var(alpha) + Var(gamma^arch) + 2 Cov(alpha, gamma^arch).
    """
    cov = result.param_cov
    rows: list[dict[str, Any]] = []
    for i in range(1, vol_spec.o + 1):
        a, g = f"alpha[{i}]", f"gamma[{i}]"
        coef = float(result.params[a] + result.params[g])
        var = float(cov.loc[a, a] + cov.loc[g, g] + 2.0 * cov.loc[a, g])
        se = float(np.sqrt(var)) if var > 0.0 else np.nan
        t_value = coef / se if np.isfinite(se) and se > 0.0 else np.nan
        rows.append(
            {
                "model_id": model_id,
                "parameter": f"alpha_plus_gamma[{i}]",
                "coef": coef,
                "std_err": se,
                "t_value": t_value,
                "p_value": float(2.0 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan,
                "derived_from": f"{a}+{g}",
            }
        )
    return rows


def attempt_detail_rows(
    model_id: str,
    distribution: str,
    mean_spec: MeanSpec,
    vol_spec: VolSpec,
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        rows.append(
            {
                "model_id": model_id,
                "distribution": distribution,
                "distribution_label": DISTRIBUTION_LABEL[distribution],
                "mean_spec": mean_spec.name,
                "mean_label": MEAN_LABEL[mean_spec.name],
                "vol_family": vol_spec.family,
                "vol_spec": vol_spec.label,
                "vol_order": f"({vol_spec.p},{vol_spec.q})",
                **attempt,
            }
        )
    return rows


def attempt_count(attempts: list[dict[str, Any]], key: str) -> int:
    return sum(1 for attempt in attempts if bool(attempt.get(key)))


def best_attempt_loglikelihood(
    attempts: list[dict[str, Any]], *, optimizer_success_only: bool = False
) -> float:
    vals: list[float] = []
    for attempt in attempts:
        if optimizer_success_only and not bool(attempt.get("optimizer_success")):
            continue
        val = float(attempt.get("loglikelihood", np.nan))
        if np.isfinite(val):
            vals.append(val)
    return max(vals) if vals else np.nan


# ============================================================
# 10) Markdown best-model reporting
# ============================================================
def _format_value(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{_no_negative_zero(x, digits):.{digits}f}"


def _signed(x: float, digits: int = 4) -> str:
    return f"{_no_negative_zero(x, digits):+.{digits}f}"


def _no_negative_zero(x: float, digits: int) -> float:
    return 0.0 if round(float(x), digits) == 0.0 else float(x)


def _mean_label(row: pd.Series) -> str:
    mean_spec = str(row.get("mean_spec", ""))
    if mean_spec in MEAN_LABEL:
        return MEAN_LABEL[mean_spec]
    return str(row.get("mean_label", mean_spec))


def _candidate_pool(summary_df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    crit_values = pd.to_numeric(summary_df[criterion], errors="coerce")
    finite_criterion = np.isfinite(crit_values.to_numpy(dtype=float))
    admissible = (
        summary_df["selected_checks_passed"] == True
        if "selected_checks_passed" in summary_df
        else (summary_df["optimizer_success"] == True) & (summary_df["stable_by_proxy"] == True)
    )
    pool = summary_df[admissible & finite_criterion].copy()
    if pool.empty:
        pool = summary_df[
            (summary_df["optimizer_success"] == True) & finite_criterion
        ].copy()
    if pool.empty:
        pool = summary_df[finite_criterion].copy()
    return pool


def _best_row(df: pd.DataFrame, criterion: str) -> pd.Series | None:
    if df.empty:
        return None
    return df.loc[df[criterion].idxmin()]


def build_family_panel_table(summary_df: pd.DataFrame, criterion: str) -> str:
    pool = _candidate_pool(summary_df, criterion)
    fams = ["GARCH", "GJR-GARCH", "EGARCH"]
    best_rows: dict[tuple[str, str], pd.Series | None] = {}
    values: list[float] = []

    for dist in DISTRIBUTION_ORDER:
        for fam in fams:
            sub = pool[(pool["distribution"] == dist) & (pool["vol_family"] == fam)]
            b = _best_row(sub, criterion)
            best_rows[(dist, fam)] = b
            if b is not None:
                values.append(float(b[criterion]))

    global_min = min(values) if values else np.nan

    if criterion == "aic":
        title = "**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**[^garch-aic-sample-note]"
        c_g = "GARCH AIC"
        c_gjr = "GJR AIC"
        c_eg = "EGARCH AIC"
    else:
        title = "**Table 3: Model selection by BIC: GARCH vs GJR vs EGARCH**"
        c_g = "GARCH BIC"
        c_gjr = "GJR BIC"
        c_eg = "EGARCH BIC"

    lines: list[str] = []
    lines.append(title)
    lines.append("")
    lines.append(
        f"| Distribution | GARCH Mean | GARCH Vol | {c_g} | GJR Mean | GJR Vol | {c_gjr} | EGARCH Mean | EGARCH Vol | {c_eg} |"
    )
    lines.append("|---|---|---|---:|---|---|---:|---|---|---:|")

    for dist in DISTRIBUTION_ORDER:
        gar = best_rows[(dist, "GARCH")]
        gjr = best_rows[(dist, "GJR-GARCH")]
        ega = best_rows[(dist, "EGARCH")]

        def crit_fmt(row: pd.Series | None) -> str:
            if row is None:
                return "NA"
            val = float(row[criterion])
            txt = _format_value(val, 4)
            if np.isfinite(global_min) and abs(val - global_min) <= 1e-10:
                return f"<span style=\"color:red\">{txt}</span>"
            return txt

        gar_mean = _mean_label(gar) if gar is not None else "NA"
        gar_vol = gar["vol_order"] if gar is not None else "NA"
        gjr_mean = _mean_label(gjr) if gjr is not None else "NA"
        gjr_vol = gjr["vol_order"] if gjr is not None else "NA"
        ega_mean = _mean_label(ega) if ega is not None else "NA"
        ega_vol = ega["vol_order"] if ega is not None else "NA"

        lines.append(
            "| "
            + f"{DISTRIBUTION_LABEL[dist]} | {gar_mean} | {gar_vol} | {crit_fmt(gar)} | "
            + f"{gjr_mean} | {gjr_vol} | {crit_fmt(gjr)} | "
            + f"{ega_mean} | {ega_vol} | {crit_fmt(ega)} |"
        )
    if criterion == "aic":
        nobs_values = pd.unique(summary_df["nobs"].dropna()) if "nobs" in summary_df else []
        nobs_text = (
            f"**{int(nobs_values[0])} observations**"
            if len(nobs_values) == 1
            else "a common number of observations"
        )
        lines.append("")
        lines.append(
            "[^garch-aic-sample-note]: These numbers differ from earlier GARCH tables because "
            "earlier code used inconsistent effective sample sizes across model settings due "
            "to a sample-trimming error. The correction is discussed in the "
            "[Data Summary effective-sample section](../../DataSummary/README.md#effective-sample). "
            f"The current estimates use a common effective sample with {nobs_text}."
        )
    return "\n".join(lines)


def build_all_model_results_table(summary_df: pd.DataFrame) -> str:
    rows = summary_df.copy()
    rows["_distribution_order"] = rows["distribution"].map(
        {dist: idx for idx, dist in enumerate(DISTRIBUTION_ORDER)}
    )
    rows["_mean_order"] = rows["mean_spec"].map(
        {mean: idx for idx, mean in enumerate(MEAN_LABEL)}
    )
    rows["_family_order"] = rows["vol_family"].map(
        {family: idx for idx, family in enumerate(VOL_FAMILIES)}
    )
    rows = rows.sort_values(
        ["_distribution_order", "_mean_order", "_family_order", "p", "q"]
    )

    lines: list[str] = []
    lines.append("## All Model Results")
    lines.append("")
    lines.append("| Distribution | Mean Process | Volatility Process | Best LogLik | AIC | BIC |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + f"{_distribution_label(row)} | {_mean_label(row)} | {row['vol_spec']} | "
            + f"{_format_value(float(row['loglikelihood']))} | "
            + f"{_format_value(float(row['aic']))} | {_format_value(float(row['bic']))} |"
        )
    return "\n".join(lines)


def _param_lookup(psub: pd.DataFrame, name: str) -> tuple[float, float]:
    row = psub[psub["parameter"] == name]
    if row.empty:
        return np.nan, np.nan
    r = row.iloc[0]
    return float(r["coef"]), float(r["std_err"])


def _parameter_label(raw_name: str, family: str) -> str:
    """Map arch/raw parameter names to notation used in the report equations."""
    mean_labels = {
        "Const": "$c$",
        "Inflation_lag_1": "$\\rho_1$",
        "Inflation_lag_2": "$\\rho_2$",
        "SPF": "$\\phi_1$",
        "SPF_lag_1": "$\\phi_2$",
    }
    if raw_name in mean_labels:
        return mean_labels[raw_name]

    scalar_labels = {
        "omega": "$\\omega$",
        "nu": "$\\nu$",
        "p_1": "$p_1$",
        "mu_1": "$\\mu_1$",
        "sigma_1_sq": "$\\sigma_1^2$",
    }
    if raw_name in scalar_labels:
        return scalar_labels[raw_name]

    m = re.fullmatch(r"(alpha|beta|gamma|alpha_plus_gamma)\[(\d+)\]", raw_name)
    if m:
        group, idx = m.groups()
        if group == "alpha_plus_gamma":
            return f"$\\gamma_{idx}$"
        if group == "gamma" and family == "GJR-GARCH":
            return f"$\\gamma_{idx}-\\alpha_{idx}$"
        symbols = {"alpha": "\\alpha", "beta": "\\beta", "gamma": "\\gamma"}
        return f"${symbols[group]}_{idx}$"

    return raw_name


def _parameter_sort_key(raw_name: str, family: str) -> tuple[int, int, str]:
    mean_order = {
        "Const": 0,
        "Inflation_lag_1": 1,
        "Inflation_lag_2": 2,
        "SPF": 3,
        "SPF_lag_1": 4,
    }
    if raw_name in mean_order:
        return (0, mean_order[raw_name], raw_name)

    if raw_name == "omega":
        return (1, 0, raw_name)

    m = re.fullmatch(r"(alpha|gamma|alpha_plus_gamma|beta)\[(\d+)\]", raw_name)
    if m:
        group, idx_s = m.groups()
        idx = int(idx_s)
        if group == "beta":
            return (1, 100 + idx, raw_name)
        group_order = {"alpha": 0, "gamma": 1, "alpha_plus_gamma": 2}
        return (1, 10 * idx + group_order[group], raw_name)

    dist_order = {"nu": 0, "p_1": 1, "mu_1": 2, "sigma_1_sq": 3}
    if raw_name in dist_order:
        return (2, dist_order[raw_name], raw_name)

    return (9, 0, raw_name)


def _distribution_label(row: pd.Series) -> str:
    dist = str(row.get("distribution", ""))
    if dist in DISTRIBUTION_LABEL:
        return DISTRIBUTION_LABEL[dist]
    return str(row.get("distribution_label", dist))


def _model_params(best_row: pd.Series, param_df: pd.DataFrame) -> pd.DataFrame:
    mid = best_row["model_id"]
    family = str(best_row["vol_family"])
    psub = param_df[param_df["model_id"] == mid].copy()
    if not psub.empty:
        psub["_sort_key"] = psub["parameter"].map(lambda x: _parameter_sort_key(str(x), family))
        psub = psub.sort_values("_sort_key")
    return psub


def _mean_equation_markdown(best_row: pd.Series, psub: pd.DataFrame) -> str:
    m = best_row["mean_spec"]
    if m == "Constant_anchor":
        return (
            "$$\n"
            "\\hat{\\pi}_{t+1} = SPF_t + \\mu_{t+1}\n"
            "$$\n"
            "\nNo mean-process coefficients are estimated in this anchored specification."
        )

    def s(v: float) -> str:
        return _signed(v)

    c, _ = _param_lookup(psub, "Const")
    i1, _ = _param_lookup(psub, "Inflation_lag_1")
    spf, _ = _param_lookup(psub, "SPF")

    if m == "ARX_1_1":
        return (
            "$$\n"
            f"\\hat{{\\pi}}_{{t+1}} = {c:.4f} {s(i1)}\\,\\pi_t {s(spf)}\\,SPF_t + \\mu_{{t+1}}\n"
            "$$"
        )

    i2, _ = _param_lookup(psub, "Inflation_lag_2")
    if m == "ARX_2_1":
        return (
            "$$\n"
            f"\\hat{{\\pi}}_{{t+1}} = {c:.4f} {s(i1)}\\,\\pi_t {s(i2)}\\,\\pi_{{t-1}} {s(spf)}\\,SPF_t + \\mu_{{t+1}}\n"
            "$$"
        )

    spf1, _ = _param_lookup(psub, "SPF_lag_1")
    return (
        "$$\n"
        f"\\hat{{\\pi}}_{{t+1}} = {c:.4f} {s(i1)}\\,\\pi_t {s(i2)}\\,\\pi_{{t-1}} "
        f"{s(spf)}\\,SPF_t {s(spf1)}\\,SPF_{{t-1}} + \\mu_{{t+1}}\n"
        "$$"
    )


def _volatility_equation_markdown(best_row: pd.Series, psub: pd.DataFrame) -> str:
    fam = best_row["vol_family"]
    p = int(best_row["p"])
    q = int(best_row["q"])
    o = int(best_row["o"])
    omega, _ = _param_lookup(psub, "omega")

    def gp(name: str, i: int) -> float:
        v, _ = _param_lookup(psub, f"{name}[{i}]")
        return v

    if fam == "GARCH":
        parts = [f"{omega:.4f}"]
        for i in range(1, p + 1):
            a = gp("alpha", i)
            if np.isfinite(a):
                parts.append(f"{_signed(a)}\\,u_{{t-{i}}}^2")
        for k in range(1, q + 1):
            b = gp("beta", k)
            if np.isfinite(b):
                parts.append(f"{_signed(b)}\\,\\sigma_{{t-{k}}}^2")
        rhs = " ".join(parts)
        return "$$\n" + f"\\hat{{\\sigma}}_t^2 = {rhs}\n" + "$$"

    if fam == "GJR-GARCH":
        # Match README notation using positive/negative decompositions:
        # arch package form alpha*u^2 + gamma*u^2 I(u<0)
        # equivalent to alpha*(u^+)^2 + (alpha+gamma)*(u^-)^2
        parts = [f"{omega:.4f}"]
        for i in range(1, p + 1):
            a = gp("alpha", i)
            g = gp("gamma", i) if i <= o else 0.0
            if np.isfinite(a):
                parts.append(f"{_signed(a)}\\,(u_{{t-{i}}}^+)^2")
            if np.isfinite(a) and np.isfinite(g):
                parts.append(f"{_signed((a + g))}\\,(u_{{t-{i}}}^-)^2")
        for k in range(1, q + 1):
            b = gp("beta", k)
            if np.isfinite(b):
                parts.append(f"{_signed(b)}\\,\\sigma_{{t-{k}}}^2")
        rhs = " ".join(parts)
        return (
            "$$\n"
            + f"\\hat{{\\sigma}}_t^2 = {rhs}\n"
            + "$$\n\n"
            + "Reported in README notation. (`arch` estimates the equivalent indicator form.)"
        )

    # EGARCH (arch convention uses sqrt(2/pi) centering)
    parts = [f"{omega:.4f}"]
    for i in range(1, p + 1):
        a = gp("alpha", i)
        if np.isfinite(a):
            parts.append(f"{_signed(a)}\\,(|z_{{t-{i}}}|-\\sqrt{{2/\\pi}})")
    for j in range(1, o + 1):
        g = gp("gamma", j)
        if np.isfinite(g):
            parts.append(f"{_signed(g)}\\,z_{{t-{j}}}")
    for k in range(1, q + 1):
        b = gp("beta", k)
        if np.isfinite(b):
            parts.append(f"{_signed(b)}\\,\\ln\\sigma_{{t-{k}}}^2")
    rhs = " ".join(parts)
    return "$$\n" + f"\\ln \\hat{{\\sigma}}_t^2 = {rhs}\n" + "$$"


def _model_detail_markdown(best_row: pd.Series, param_df: pd.DataFrame) -> str:
    family = str(best_row["vol_family"])
    psub = _model_params(best_row, param_df)

    lines: list[str] = []
    lines.append("| Model ID | Distribution | Mean process | Volatility process | Log likelihood | AIC | BIC |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    lines.append(
        "| "
        + f"`{best_row['model_id']}` | {_distribution_label(best_row)} | {_mean_label(best_row)} | "
        + f"{best_row['vol_spec']} | {_format_value(float(best_row['loglikelihood']))} | "
        + f"{_format_value(float(best_row['aic']))} | {_format_value(float(best_row['bic']))} |"
    )
    lines.append("")
    lines.append("Mean process:")
    lines.append("")
    lines.append(_mean_equation_markdown(best_row, psub))
    lines.append("")
    lines.append("Volatility process:")
    lines.append("")
    lines.append(_volatility_equation_markdown(best_row, psub))
    lines.append("")
    dist_params = _distribution_parameters_markdown(best_row, psub)
    if dist_params:
        lines.append(dist_params)
        lines.append("")
    if family == "GJR-GARCH":
        lines.append(
            "For GJR-GARCH, `arch` estimates the indicator coefficient $\\gamma_i-\\alpha_i$ "
            "(row $\\gamma_i-\\alpha_i$). The negative-shock coefficient of the README "
            "specification is $\\gamma_i = \\alpha_i + (\\gamma_i-\\alpha_i)$ (row $\\gamma_i$); "
            "its standard error uses the delta method on the same robust covariance matrix, "
            "$\\operatorname{Var}(\\hat\\gamma_i) = \\operatorname{Var}(\\hat\\alpha_i) + "
            "\\operatorname{Var}(\\widehat{\\gamma_i-\\alpha_i}) + "
            "2\\operatorname{Cov}(\\hat\\alpha_i, \\widehat{\\gamma_i-\\alpha_i})$."
        )
        lines.append("")
    lines.append("| Parameter | Coef | Std Err | t-value | p-value |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in psub.iterrows():
        label = _parameter_label(str(r["parameter"]), family)
        lines.append(
            f"| {label} | {_format_value(float(r['coef']))} | {_format_value(float(r['std_err']))} | "
            + f"{_format_value(float(r['t_value']))} | {_format_value(float(r['p_value']))} |"
        )
    return "\n".join(lines)


def _distribution_parameters_markdown(best_row: pd.Series, psub: pd.DataFrame) -> str:
    dist = str(best_row.get("distribution", ""))
    if dist == "studentst":
        nu, _ = _param_lookup(psub, "nu")
        return "\n".join(
            [
                "Distribution parameters:",
                "",
                "| Parameter | Estimate |",
                "|---|---:|",
                f"| $\\nu$ | {_format_value(nu)} |",
            ]
        )

    if dist != "mix_normal":
        return ""

    p1, _ = _param_lookup(psub, "p_1")
    mu1, _ = _param_lookup(psub, "mu_1")
    sigma1_sq, _ = _param_lookup(psub, "sigma_1_sq")
    p2 = 1.0 - p1
    if np.isfinite(p1) and np.isfinite(mu1) and np.isfinite(sigma1_sq) and p2 > 0.0:
        mu2 = -p1 * mu1 / p2
        sigma2_sq = (1.0 - p1 * (sigma1_sq + mu1 * mu1) - p2 * mu2 * mu2) / p2
    else:
        p2 = mu2 = sigma2_sq = np.nan

    return "\n".join(
        [
            "Distribution parameters:",
            "",
            "| Parameter | Estimate |",
            "|---|---:|",
            f"| $p_1$ | {_format_value(p1)} |",
            f"| $\\mu_1$ | {_format_value(mu1)} |",
            f"| $\\sigma_1^2$ | {_format_value(sigma1_sq)} |",
            f"| $p_2$ | {_format_value(p2)} |",
            f"| $\\mu_2$ | {_format_value(mu2)} |",
            f"| $\\sigma_2^2$ | {_format_value(sigma2_sq)} |",
        ]
    )


def build_parameter_range_section(
    summary_df: pd.DataFrame, param_df: pd.DataFrame, n_starts: int
) -> str:
    """Document arch's parameter bounds and the random-start draw ranges."""
    egarch_ids = set(summary_df.loc[summary_df["vol_family"] == "EGARCH", "model_id"]) if "vol_family" in summary_df else set()
    eg = param_df[param_df["model_id"].isin(egarch_ids)] if not param_df.empty else param_df
    if len(eg):
        max_a = eg.loc[eg["parameter"].str.startswith("alpha["), "coef"].abs().max()
        max_g = eg.loc[eg["parameter"].str.startswith("gamma["), "coef"].abs().max()
        egarch_note = (
            f"the reported EGARCH estimates have $\\max|\\alpha_i|={max_a:.4f}$ and "
            f"$\\max|\\gamma_j|={max_g:.4f}$, well inside the draw range "
            f"$[{EGARCH_ALPHA_DRAW[0]:g},{EGARCH_ALPHA_DRAW[1]:g}]$."
        )
    else:
        egarch_note = "no EGARCH estimates are reported."
    scales = (
        summary_df.groupby("mean_spec")["resid_scale"].first()
        if "resid_scale" in summary_df
        else pd.Series(dtype=float)
    )
    vbar = ", ".join(
        f"{MEAN_LABEL[m]} {scales[m]:.4f}" for m in MEAN_LABEL if m in scales.index
    ) or "NA"
    a_lo, a_hi = EGARCH_ALPHA_DRAW
    g_lo, g_hi = EGARCH_GAMMA_DRAW
    k = MEAN_START_SE_MULTIPLIER
    lines = [
        "## Parameter Ranges and Random Starts",
        "",
        "All models are estimated with `arch` 7.0.0, which maximizes the likelihood by SLSQP "
        "subject to the bounds and linear constraints in the second column of the table below. "
        "`arch` imposes no bounds on mean-process coefficients or on the EGARCH coefficients "
        "$\\alpha_i$ and $\\gamma_j$. If a supplied starting vector violates any bound or "
        "constraint, `arch` silently replaces the whole vector with its own default start. "
        f"Each model combination uses {n_starts} starts. Start 1 is the `arch` default (Gaussian "
        "mixture: the Student's $t$ estimates plus $(p_1,\\mu_1,\\sigma_1^2)=(0.3,0.1,1)$ when "
        "that vector is feasible, otherwise the `arch` default); "
        f"starts 2--{n_starts} are random draws from the third column. Every random draw lies "
        "inside `arch`'s feasible set and inside the stationary region (volatility persistence "
        "and AR roots below $1-10^{-6}$ in modulus), so `arch` uses every random start as drawn.",
        "",
        "$\\bar v$ denotes the mean squared residual at `arch`'s mean starting values (OLS "
        f"residuals for ARX means, $\\pi_t-SPF_t$ for the Constant mean): {vbar}.",
        "",
        "| Parameter | `arch` bound / constraint | Random-start draw |",
        "|---|---|---|",
        "| **Mean process** | | |",
        "| $\\rho_1$ in ARX(1,·) | none | $U(-1,1)$ (AR stationarity) |",
        "| $(\\rho_1,\\rho_2)$ in ARX(2,·) | none | uniform on the stationarity triangle "
        "$\\rho_2>-1$, $\\lvert\\rho_1\\rvert<1-\\rho_2$ |",
        "| $c$, $\\phi_1$, $\\phi_2$ | none | uniform on $\\hat b(\\rho)\\pm "
        f"{k:g}\\,\\widehat{{SE}}(\\rho)$, the OLS fit of $\\pi_t-\\rho_1\\pi_{{t-1}}-\\rho_2\\pi_{{t-2}}$ "
        "on a constant and the SPF terms given the drawn $\\rho$ |",
        "| **GARCH** | | |",
        "| $\\omega$ | $[10^{-8}\\bar v,\\ 10\\bar v]$ | log-uniform on the bound |",
        "| $\\alpha_i$, $\\beta_k$ | $\\alpha_i,\\beta_k\\in[0,1]$; $\\sum\\alpha_i+\\sum\\beta_k\\le1$ "
        "| uniform on $\\{\\alpha,\\beta\\ge0,\\ \\sum\\alpha_i+\\sum\\beta_k<1\\}$ |",
        "| **GJR-GARCH** | | |",
        "| $\\omega$ | $[10^{-8}\\bar v,\\ 10\\bar v]$ | log-uniform on the bound |",
        "| $\\alpha_i$ (on $(u^+)^2$) | $[0,1]$ | uniform on $\\{\\alpha_i\\in[0,1],\\ \\gamma_i\\ge0,\\ "
        "\\beta_k\\ge0,\\ \\tfrac12\\sum\\alpha_i+\\tfrac12\\sum\\gamma_i+\\sum\\beta_k<1\\}$ |",
        "| $\\gamma_i$ (on $(u^-)^2$) | `arch` estimates $\\gamma_i-\\alpha_i\\in[-1,2]$ subject to "
        "$\\gamma_i\\ge0$ | joint draw above |",
        "| $\\beta_k$ | $[0,1]$; $\\tfrac12\\sum\\alpha_i+\\tfrac12\\sum\\gamma_i+\\sum\\beta_k\\le1$ "
        "| joint draw above |",
        "| **EGARCH** | | |",
        "| $\\omega$ | $[\\ln\\bar v-\\ln10^4,\\ \\ln\\bar v+\\ln10^4]$ | uniform on the bound |",
        f"| $\\alpha_i$ | none | $U({a_lo:g},{a_hi:g})$ |",
        f"| $\\gamma_j$ | none | $U({g_lo:g},{g_hi:g})$ |",
        "| $\\beta_k$ | $[0,q]$; $\\sum\\beta_k\\le1$ | uniform on $\\{\\beta\\ge0,\\ \\sum\\beta_k<1\\}$ |",
        "| **Student's $t$** | | |",
        "| $\\nu$ | $[2.05,\\ 500]$ | log-uniform on the bound |",
        "| **Gaussian mixture** | | |",
        "| $p_1$ | $[10^{-4},\\ 1-10^{-4}]$ | uniform on the bound |",
        "| $\\mu_1$ | $[-50,\\ 50]$ | uniform on the values that allow $\\sigma_2^2\\ge10^{-6}$ |",
        "| $\\sigma_1^2$ | $[10^{-6},\\ 100]$; $\\sigma_2^2>0$ through a likelihood penalty | uniform "
        "on $[10^{-6},\\ \\min(100,\\bar\\sigma_1^2)]$, where $\\bar\\sigma_1^2$ gives "
        "$\\sigma_2^2=10^{-6}$ |",
        "",
        "Notes:",
        "",
        "- *Uniform on a region* means a uniform draw over the whole region: Dirichlet(1,…,1) "
        "weights for the volatility simplices, rejection sampling for the AR triangle and for "
        "`arch`'s caps. *Log-uniform* draws are uniform in the logarithm of the parameter.",
        "- Compared with the stationarity conditions in the README, `arch`'s feasible set is the "
        "same for GARCH. For GJR-GARCH, `arch` also caps $\\alpha_i\\le1$, although stationarity "
        "alone allows $\\alpha_i<2$ when $\\gamma_i$ is small. For EGARCH, `arch` requires "
        "$\\beta_k\\ge0$, under which $\\sum\\beta_k<1$ is the stationarity condition; $\\omega$ is "
        "also bounded. The mixture bounds come from this repository's `MixNormal` class. "
        f"$\\alpha_i$ and $\\gamma_j$ in EGARCH are unbounded in `arch`; {egarch_note}",
        "- Selection: for each combination the reported estimate has the highest log likelihood "
        "among *admissible* starts, which must pass five checks. (1) SLSQP reports success and "
        "`arch` returns convergence flag 0. (2) The first-order (KKT) conditions hold: the "
        "numerical gradient of the log likelihood, net of non-negative multipliers on the active "
        f"bounds and constraints, is at most {KKT_TOL_PER_OBS:g}$\\cdot T$ in absolute value. "
        "When a start that would become the best fails this check, SLSQP is restarted from its "
        f"own solution up to {MAX_POLISH_ROUNDS} times, because SLSQP sometimes stops just short "
        "of a genuine maximum. Points on extremely steep ridges, where SLSQP reports success "
        "with gradients of $10^1$--$10^6$, do not move and stay rejected. (3) Volatility is "
        "stationary. (4) The AR "
        "roots lie inside the unit circle. (5) For the Gaussian mixture, the component "
        f"variances satisfy $\\min(\\sigma_1^2,\\sigma_2^2)/\\max(\\sigma_1^2,\\sigma_2^2)\\ge"
        f"{MIX_MIN_VARIANCE_RATIO:g}$. This Hathaway-type restriction keeps the mixture likelihood "
        "bounded; without it the best fits place a component with variance $10^{-6}$ on one or "
        "two observations. If no start passes, the best converged start that satisfies the "
        "first-order conditions is reported and flagged in the CSV `selection_status` column. "
        f"`best_loglik_hits` in the summary CSV counts the admissible starts that came within "
        f"{LOGLIK_HIT_TOL:g} of the reported log likelihood, and `params_at_arch_bound` lists "
        "parameters of the reported estimate that sit on an `arch` bound.",
    ]
    if "best_inadmissible_loglikelihood" in summary_df:
        gap = summary_df["best_inadmissible_loglikelihood"] - summary_df["loglikelihood"]
        n_gap = int((gap > LOGLIK_HIT_TOL).sum())
        lines.append(
            f"- In {n_gap} of {len(summary_df)} combinations, a converged fit that satisfies the "
            "first-order conditions but fails stationarity or the mixture rule has a higher log "
            "likelihood than the reported estimate (largest difference "
            f"{np.nanmax(gap.to_numpy(dtype=float), initial=0.0):.4f}). For these combinations the "
            "likelihood peaks on the stationarity boundary or at a degenerate mixture, and the "
            "reported estimate is the best admissible local maximum. The CSV column "
            "`best_inadmissible_loglikelihood` records the higher value."
        )
    if "params_at_arch_bound" in summary_df:
        at_bound = summary_df["params_at_arch_bound"].fillna("").astype(str)
        names = pd.Series(
            [re.sub(r"\[\d+\]", "", n) for v in at_bound for n in v.split(";") if n]
        ).value_counts()
        detail = ", ".join(f"{n} ({c})" for n, c in names.items()) or "none"
        lines.append(
            f"- In this run, {int((at_bound != '').sum())} of {len(summary_df)} reported "
            f"estimates have at least one parameter on an `arch` bound; parameters on a bound, "
            f"with counts: {detail}. Standard errors of estimates on a bound or an active "
            "constraint (for example $\\beta_k=0$, or a GJR $\\gamma_i=0$) are not "
            "asymptotically normal and should be read with caution."
        )
    return "\n".join(lines)


def build_markdown_report(
    summary_df: pd.DataFrame,
    param_df: pd.DataFrame,
    failed_df: pd.DataFrame | None = None,
    n_starts: int = DEFAULT_N_STARTS,
    retry_failed_starts: int = DEFAULT_RETRY_FAILED_STARTS,
    sample_note: str = "",
) -> str:
    lines: list[str] = []
    lines.append("```{raw:typst}")
    lines.append("#set page(margin: auto)")
    lines.append("```")
    lines.append("")
    if "Monthly.pkl" in sample_note:
        lines.append('```{raw:typst}\n#set text(fill: rgb("#808080"))\n#show text: set text(fill: rgb("#808080"))\n```\n\n')
    lines.append("# Model Selection Report")
    lines.append("")
    if sample_note:
        lines.append(sample_note)
        lines.append("")
    lines.append(build_parameter_range_section(summary_df, param_df, n_starts))
    lines.append("")
    lines.append("## Model Selection")
    lines.append("")
    lines.append(
        f"For each model combination, the reported row uses the highest log-likelihood "
        f"estimate that passes all admissibility checks (convergence, first-order conditions, "
        f"volatility and mean stationarity, and the mixture variance ratio) across {n_starts} "
        "starts. If no start passes all checks, the reported row uses the best converged start "
        "that satisfies the first-order conditions and is flagged in the CSV `selection_status` "
        "column."
    )
    lines.append("")
    lines.append(build_family_panel_table(summary_df, "aic"))
    lines.append("")
    lines.append(build_family_panel_table(summary_df, "bic"))
    lines.append("")
    lines.append(build_all_model_results_table(summary_df))
    lines.append("")

    lines.append("## Best Models by Criterion and Volatility Family")
    lines.append("")
    lines.append(
        "For each criterion and each volatility family, the selected model is the best "
        "across mean-process choices, orders, and distributions using stable & successful fits."
    )
    lines.append(
        f"The main result tables contain {len(summary_df)} reported model combinations."
    )
    lines.append("")

    for criterion in ["aic", "bic"]:
        pool = _candidate_pool(summary_df, criterion)
        lines.append(f"### {criterion.upper()}")
        lines.append("")
        for family in VOL_FAMILIES:
            sub = pool[pool["vol_family"] == family]
            best = _best_row(sub, criterion)
            if best is None:
                lines.append(f"#### {family}")
                lines.append("No stable converged estimate was available after the restart search.")
                lines.append("")
                continue
            lines.append(f"#### {family}")
            lines.append(_model_detail_markdown(best, param_df))
            lines.append("")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 11) Main estimation routine
# ============================================================
@dataclass(frozen=True)
class ModelEntry:
    index: int
    model_id: str
    dist: str
    mean_spec: MeanSpec
    vol_spec: VolSpec


def enumerate_models() -> list[ModelEntry]:
    """All combinations in a fixed order; the index seeds each model's RNG."""
    entries: list[ModelEntry] = []
    for dist in DISTRIBUTIONS:
        for mean_spec in build_mean_specs():
            for vol_spec in build_vol_specs():
                k = len(entries) + 1
                entries.append(ModelEntry(k, f"M{k:03d}", dist, mean_spec, vol_spec))
    return entries


def failed_row(entry: ModelEntry, attempts: list[dict[str, Any]], message: str) -> dict[str, Any]:
    return {
        "model_id": entry.model_id,
        "distribution": entry.dist,
        "distribution_label": DISTRIBUTION_LABEL[entry.dist],
        "mean_spec": entry.mean_spec.name,
        "mean_label": MEAN_LABEL[entry.mean_spec.name],
        "vol_family": entry.vol_spec.family,
        "vol_spec": entry.vol_spec.label,
        "vol_order": f"({entry.vol_spec.p},{entry.vol_spec.q})",
        "attempts": len(attempts),
        "optimizer_success_attempts": attempt_count(attempts, "optimizer_success"),
        "stable_attempts": attempt_count(attempts, "stable_by_proxy"),
        "best_attempt_loglikelihood": best_attempt_loglikelihood(attempts),
        "best_optimizer_success_loglikelihood": best_attempt_loglikelihood(
            attempts, optimizer_success_only=True
        ),
        "last_error_message": message,
    }


def estimate_model(
    df: pd.DataFrame,
    entry: ModelEntry,
    n_starts: int,
    retry_failed_starts: int,
    maxiter: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate one combination and return its output rows."""
    dist, mean_spec, vol_spec = entry.dist, entry.mean_spec, entry.vol_spec
    prefix = f"{entry.model_id} dist={dist:<10} mean={mean_spec.name:<16} vol={vol_spec.label:<15}"
    record: dict[str, Any] = {
        "model_id": entry.model_id,
        "settings": {
            "n_starts": n_starts,
            "retry_failed_starts": retry_failed_starts,
            "maxiter": maxiter,
            "seed": seed,
        },
        "summary": None,
        "params": [],
        "attempts": [],
        "failed": None,
        "text": "",
    }
    y, x = prepare_mean_inputs(df, mean_spec)
    try:
        if dist in ("normal", "studentst"):
            fit_func, model = make_arch_fit_func(
                y=y,
                x=x,
                mean=mean_spec.mean,
                lags=mean_spec.lags,
                vol_spec=vol_spec,
                dist=dist,
                hold_back=COMMON_HOLD_BACK,
                maxiter=maxiter,
            )
            init_sv = None
        else:
            fit_func, init_sv, model = make_sparch_fit_func(
                y=y,
                x=x,
                mean=mean_spec.mean,
                lags=mean_spec.lags,
                vol_spec=vol_spec,
                hold_back=COMMON_HOLD_BACK,
                maxiter=maxiter,
            )
        feasible = arch_feasible_set(model)
        if init_sv is not None and not feasible.contains(init_sv):
            init_sv = None  # infeasible benchmark-based start: use arch's default instead
        drawer = StartDrawer(feasible, vol_spec, dist, y, x)

        result, attempts, selection_status = fit_with_restarts(
            fit_func=fit_func,
            model=model,
            drawer=drawer,
            vol_spec=vol_spec,
            rng=model_phase_rng(seed, entry.index, 0),
            n_starts=n_starts,
            initial_starting_values=init_sv,
            phase="primary",
        )
        if selection_status != "passed_checks" and retry_failed_starts > 0:
            retry_result, retry_attempts, retry_status = fit_with_restarts(
                fit_func=fit_func,
                model=model,
                drawer=drawer,
                vol_spec=vol_spec,
                rng=model_phase_rng(seed, entry.index, 1),
                n_starts=retry_failed_starts,
                initial_starting_values=init_sv,
                phase="retry",
                attempt_offset=len(attempts),
            )
            attempts.extend(retry_attempts)
            if retry_status == "passed_checks" or (
                retry_result is not None
                and (result is None or float(retry_result.loglikelihood) > float(result.loglikelihood))
            ):
                result, selection_status = retry_result, retry_status

        record["attempts"] = attempt_detail_rows(
            model_id=entry.model_id,
            distribution=dist,
            mean_spec=mean_spec,
            vol_spec=vol_spec,
            attempts=attempts,
        )
        if result is None:
            message = attempts[-1]["error_message"] if attempts else "no attempts"
            record["failed"] = failed_row(entry, attempts, message)
            record["text"] = f"{prefix} FAILED_STABLE_CONVERGED attempts={len(attempts)}"
            return record

        checks = model_checks(result, vol_spec)
        srow = summary_row(
            model_id=entry.model_id,
            distribution=dist,
            mean_spec=mean_spec,
            vol_spec=vol_spec,
            result=result,
            checks=checks,
            attempts=attempts,
            selection_status=selection_status,
            feasible=feasible,
            drawer=drawer,
            n_starts=n_starts,
            kkt_violation=first_order_violation(model, feasible, np.asarray(result.params, dtype=float)),
        )
        record["summary"] = srow
        record["params"] = parameter_rows(entry.model_id, result, vol_spec)
        record["text"] = (
            f"{prefix} LogLik={result.loglikelihood:.6f} AIC={result.aic:.6f} BIC={result.bic:.6f} "
            f"valid_starts={srow['valid_restart_attempts']}/{srow['restart_attempts']} "
            f"hits={srow['best_loglik_hits']} selection={selection_status}"
        )
    except Exception as exc:
        record["failed"] = failed_row(entry, [], str(exc))
        record["text"] = f"{prefix} FAILED: {exc}"
    return record


def sample_note(df: pd.DataFrame, data_path: Path) -> str:
    note = (
        f"Effective sample: **{df.index[0]}--{df.index[-1]}** (**{len(df)} observations**, "
        f"`{data_path.name}`"
    )
    vintage_file = data_path.parent / "data_vintage.json"
    if vintage_file.exists():
        note += f", data vintage {json.loads(vintage_file.read_text())['vintage']}"
    return note + ")."


def write_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    note: str,
) -> None:
    settings = {json.dumps(r["settings"], sort_keys=True) for r in records}
    if len(settings) != 1:
        raise ValueError(f"Records were estimated with different settings: {sorted(settings)}")
    run = records[0]["settings"]

    summary_rows = [r["summary"] for r in records if r["summary"] is not None]
    param_rows = [row for r in records for row in r["params"]]
    attempt_rows = [row for r in records for row in r["attempts"]]
    failed_rows = [r["failed"] for r in records if r["failed"] is not None]
    text_lines = [r["text"] for r in records]

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=MIN_SUMMARY_COLUMNS)
    else:
        summary_df = summary_df.sort_values(SUMMARY_SORT_COLUMNS)
    param_df = pd.DataFrame(param_rows).sort_values(["model_id", "parameter"]) if param_rows else pd.DataFrame()
    attempt_df = (
        pd.DataFrame(attempt_rows).sort_values(["model_id", "attempt"])
        if attempt_rows
        else pd.DataFrame()
    )
    failed_df = (
        pd.DataFrame(failed_rows)
        if failed_rows
        else pd.DataFrame(columns=FAILED_SUMMARY_COLUMNS)
    )

    summary_csv = output_dir / "garch_family_estimation_summary.csv"
    params_csv = output_dir / "garch_family_estimation_parameters.csv"
    attempts_csv = output_dir / "garch_family_restart_attempts.csv.gz"
    failed_csv = output_dir / "garch_family_failed_models.csv"
    summary_txt = output_dir / "garch_family_estimation_summary.txt"
    report_md = output_dir / "garch_family_best_models.md"

    summary_df.to_csv(summary_csv, index=False)
    param_df.to_csv(params_csv, index=False)
    attempt_df.to_csv(attempts_csv, index=False)
    failed_df.to_csv(failed_csv, index=False)

    txt_header = [
        "GARCH-family estimation summary (reported models)",
        "",
        f"Settings: {json.dumps(run, sort_keys=True)}",
        "Columns in line printout: model_id, distribution, mean, vol, LogLik, AIC, BIC, "
        "valid/attempted starts, starts within "
        f"{LOGLIK_HIT_TOL:g} of the reported LogLik, selection",
        "",
    ]
    summary_txt.write_text("\n".join(txt_header + text_lines) + "\n", encoding="utf-8")

    report_text = build_markdown_report(
        summary_df,
        param_df,
        failed_df,
        n_starts=run["n_starts"],
        retry_failed_starts=run["retry_failed_starts"],
        sample_note=note,
    )
    report_md.write_text(report_text + "\n", encoding="utf-8")

    print("\nSaved:")
    for path in [summary_csv, params_csv, attempts_csv, failed_csv, summary_txt, report_md]:
        print(f"  - {path}")
    print("\n" + "\n".join(text_lines))


def run_estimation(
    data_path: Path,
    output_dir: Path,
    n_starts: int,
    retry_failed_starts: int,
    maxiter: int,
    seed: int,
) -> None:
    df = load_quarterly_data(data_path)
    records = [
        estimate_model(df, entry, n_starts, retry_failed_starts, maxiter, seed)
        for entry in enumerate_models()
    ]
    write_outputs(records, output_dir, sample_note(df, data_path))


def run_model_jobs(
    data_path: Path,
    raw_dir: Path,
    model_ids: Sequence[str],
    n_starts: int,
    retry_failed_starts: int,
    maxiter: int,
    seed: int,
) -> None:
    """Grid job: estimate the requested combinations and pickle each record."""
    entries = {e.model_id: e for e in enumerate_models()}
    unknown = sorted(set(model_ids) - set(entries))
    if unknown:
        raise SystemExit(f"Unknown model ids: {unknown}")
    df = load_quarterly_data(data_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        record = estimate_model(df, entries[model_id], n_starts, retry_failed_starts, maxiter, seed)
        tmp = raw_dir / f".{model_id}.pkl.tmp"
        pd.to_pickle(record, tmp)
        tmp.replace(raw_dir / f"{model_id}.pkl")  # atomic: the collector never sees a partial file
        print(record["text"], flush=True)


def collect_model_jobs(
    data_path: Path, raw_dir: Path, output_dir: Path, allow_missing: bool
) -> None:
    """Merge the per-combination records written by run_model_jobs."""
    records, missing = [], []
    for entry in enumerate_models():
        path = raw_dir / f"{entry.model_id}.pkl"
        if path.exists():
            records.append(pd.read_pickle(path))
        else:
            missing.append(entry.model_id)
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} model records missing in {raw_dir}: {missing}")
    if missing:
        print(f"WARNING: collecting without {len(missing)} missing models: {missing}")
    df = load_quarterly_data(data_path)
    write_outputs(records, output_dir, sample_note(df, data_path))


# ============================================================
# 12) CLI entrypoint
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate GARCH/GJR-GARCH/EGARCH with four mean processes and "
            "normal, studentst, and mix_normal distributions."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the effective-sample data pickle (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=DEFAULT_N_STARTS,
        help=(
            "Optimizer starts per model combination, including the default ARCH start "
            f"(default: {DEFAULT_N_STARTS})."
        ),
    )
    parser.add_argument(
        "--retry-failed-starts",
        type=int,
        default=DEFAULT_RETRY_FAILED_STARTS,
        help=(
            "Additional optimizer starts for combinations with no stable converged fit after "
            f"the primary search (default: {DEFAULT_RETRY_FAILED_STARTS})."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help=f"Maximum optimizer iterations per start (default: {DEFAULT_MAXITER}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for restart starting values.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model id, distribution, mean process, and volatility spec, then exit.",
    )
    parser.add_argument(
        "--model-id",
        nargs="+",
        default=None,
        help="Grid job mode: estimate only these combinations and pickle them to --raw-dir.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Merge the per-combination pickles in --raw-dir and write all outputs.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Per-combination pickle directory.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="With --collect, write outputs even if some combinations are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        for e in enumerate_models():
            print(f"{e.model_id}\t{e.dist}\t{e.mean_spec.name}\t{e.vol_spec.label}")
        return
    if (args.model_id or args.collect) and args.raw_dir is None:
        raise SystemExit("--model-id and --collect require --raw-dir")
    if args.model_id:
        run_model_jobs(
            data_path=args.data_path,
            raw_dir=args.raw_dir,
            model_ids=args.model_id,
            n_starts=args.n_starts,
            retry_failed_starts=args.retry_failed_starts,
            maxiter=args.maxiter,
            seed=args.seed,
        )
    elif args.collect:
        collect_model_jobs(args.data_path, args.raw_dir, args.output_dir, args.allow_missing)
    else:
        run_estimation(
            data_path=args.data_path,
            output_dir=args.output_dir,
            n_starts=args.n_starts,
            retry_failed_starts=args.retry_failed_starts,
            maxiter=args.maxiter,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
