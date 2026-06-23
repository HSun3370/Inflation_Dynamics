#!/usr/bin/env python3
"""Estimate GARCH-family models under multiple mean processes and distributions.

Estimated combinations
- Mean processes: Constant anchor, ARX(1,1), ARX(2,1), ARX(2,2)
- Volatility families: GARCH, GJR-GARCH, EGARCH
- Orders: (1,1), (1,2), (2,1), (2,2)
- Distributions: normal, studentst, mix_normal

Outputs
- CSV summary for all models
- CSV parameters for all models
- CSV diagnostics for combinations with no stable converged fit
- TXT line-by-line printout (AIC/BIC/LogLik)
- Markdown report with best model by AIC and by BIC
"""

from __future__ import annotations

# ============================================================
# 1) Imports and runtime environment
# ============================================================
import argparse
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
DEFAULT_N_STARTS = 100
DEFAULT_RETRY_FAILED_STARTS = 400
DEFAULT_MAXITER = 8000
SUMMARY_SORT_COLUMNS = ["distribution", "mean_spec", "vol_family", "p", "q"]
MIN_SUMMARY_COLUMNS = [
    *SUMMARY_SORT_COLUMNS,
    "optimizer_success",
    "stable_by_proxy",
    "loglikelihood",
    "aic",
    "bic",
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
        return float(np.trapezoid((x**n) * pdf, x))


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=StartingValueWarning)
            return am.fit(**kwargs)

    return _fit


def make_sparch_fit_func(
    y: pd.Series,
    x: pd.DataFrame | None,
    mean: str,
    lags: int | list[int],
    vol_spec: VolSpec,
    hold_back: int,
    maxiter: int,
):
    benchmark = arch_model(
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
    ).fit(
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
        kwargs["starting_values"] = initial if starting_values is None else starting_values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=StartingValueWarning)
            return mix_model.fit(**kwargs)

    return _fit, initial


# ============================================================
# 8) Default ARCH initialization + restart search
# ============================================================
def optimizer_converged(result: Any) -> bool:
    return bool(result.optimization_result.success) and int(result.convergence_flag) == 0


def stable_converged(result: Any, vol_spec: VolSpec) -> tuple[bool, float, float, str]:
    unc_var, persistence, unc_status = implied_initial_variance(result.params, vol_spec)
    stable = bool(np.isfinite(persistence) and persistence < 1.0 - STABILITY_MARGIN)
    return optimizer_converged(result) and stable, float(unc_var), float(persistence), unc_status


def _random_mix_params(rng: np.random.Generator) -> tuple[float, float, float]:
    for _ in range(100):
        p1 = float(rng.uniform(0.15, 0.85))
        p2 = 1.0 - p1
        mu1 = float(rng.uniform(-0.8, 0.8))
        mu2 = -p1 * mu1 / p2
        max_sigma1 = (1.0 - p1 * mu1 * mu1 - p2 * mu2 * mu2) / p1
        if max_sigma1 > 0.08:
            sigma1_sq = float(rng.uniform(0.05, min(2.0, 0.9 * max_sigma1)))
            return p1, mu1, sigma1_sq
    return 0.5, 0.0, 0.7


def randomized_starting_values(
    base_params: pd.Series,
    vol_spec: VolSpec,
    dist: str,
    y: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    params = base_params.copy().astype(float)
    y_var = float(np.var(np.asarray(y, dtype=float), ddof=1))
    scale = max(y_var, 1e-4)

    for name in params.index:
        if name in {"Const", "Inflation_lag_1", "Inflation_lag_2", "SPF", "SPF_lag_1"}:
            base = float(params[name])
            params[name] = base + float(rng.normal(0.0, 0.15 * max(1.0, abs(base))))

    if vol_spec.family == "EGARCH":
        beta_weights = rng.dirichlet(np.ones(vol_spec.q))
        beta_total = float(rng.uniform(0.05, 0.95))
        params["omega"] = float(rng.normal(np.log(scale) * (1.0 - beta_total), 0.25))
        for i in range(1, vol_spec.p + 1):
            params[f"alpha[{i}]"] = float(rng.uniform(0.02, 0.7))
        for i in range(1, vol_spec.o + 1):
            params[f"gamma[{i}]"] = float(rng.uniform(-0.35, 0.35))
        for i, weight in enumerate(beta_weights, start=1):
            params[f"beta[{i}]"] = float(beta_total * weight)
    elif vol_spec.family == "GJR-GARCH":
        shock_weights = rng.dirichlet(np.ones(2 * vol_spec.p + vol_spec.q))
        persistence = float(rng.uniform(0.15, 0.96))
        shock_mass = persistence * shock_weights[: 2 * vol_spec.p]
        beta_mass = persistence * shock_weights[2 * vol_spec.p :]
        omega = scale * (1.0 - persistence) * float(rng.uniform(0.5, 1.5))
        params["omega"] = max(omega, 1e-8)
        for i in range(1, vol_spec.p + 1):
            pos_coef = float(2.0 * shock_mass[2 * (i - 1)])
            neg_coef = float(2.0 * shock_mass[2 * (i - 1) + 1])
            params[f"alpha[{i}]"] = max(pos_coef, 1e-8)
            params[f"gamma[{i}]"] = neg_coef - pos_coef
        for i, mass in enumerate(beta_mass, start=1):
            params[f"beta[{i}]"] = max(float(mass), 1e-8)
    else:
        weights = rng.dirichlet(np.ones(vol_spec.p + vol_spec.q))
        persistence = float(rng.uniform(0.15, 0.96))
        omega = scale * (1.0 - persistence) * float(rng.uniform(0.5, 1.5))
        params["omega"] = max(omega, 1e-8)
        for i in range(1, vol_spec.p + 1):
            params[f"alpha[{i}]"] = max(float(persistence * weights[i - 1]), 1e-8)
        for i in range(1, vol_spec.q + 1):
            params[f"beta[{i}]"] = max(float(persistence * weights[vol_spec.p + i - 1]), 1e-8)

    if dist == "studentst" and "nu" in params.index:
        params["nu"] = float(rng.uniform(3.2, 35.0))
    elif dist == "mix_normal":
        p1, mu1, sigma1_sq = _random_mix_params(rng)
        if "p_1" in params.index:
            params["p_1"] = p1
        if "mu_1" in params.index:
            params["mu_1"] = mu1
        if "sigma_1_sq" in params.index:
            params["sigma_1_sq"] = sigma1_sq

    return np.asarray(params, dtype=float)


def fit_with_restarts(
    fit_func: Callable[[np.ndarray | None], Any],
    vol_spec: VolSpec,
    dist: str,
    y: pd.Series,
    rng: np.random.Generator,
    n_starts: int,
    initial_starting_values: np.ndarray | None = None,
    phase: str = "primary",
    attempt_offset: int = 0,
) -> tuple[Any | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    valid_results: list[Any] = []
    base_params: pd.Series | None = None

    start_values: list[np.ndarray | None] = [initial_starting_values]

    for attempt_idx in range(max(1, n_starts)):
        sv = start_values[attempt_idx] if attempt_idx < len(start_values) else None
        attempt_number = attempt_offset + attempt_idx + 1
        if sv is None:
            label = "default"
        elif attempt_idx == 0 and initial_starting_values is not None:
            label = "initial"
        else:
            label = f"random_{attempt_number}"
        try:
            result = fit_func(sv)
            usable, unc_var, persistence, unc_status = stable_converged(result, vol_spec)
            if base_params is None:
                base_params = result.params.copy()
            attempts.append(
                {
                    "phase": phase,
                    "attempt": attempt_number,
                    "start_type": label,
                    "loglikelihood": float(result.loglikelihood),
                    "optimizer_success": bool(result.optimization_result.success),
                    "convergence_flag": int(result.convergence_flag),
                    "stable_by_proxy": bool(
                        np.isfinite(persistence) and persistence < 1.0 - STABILITY_MARGIN
                    ),
                    "persistence_proxy": persistence,
                    "implied_initial_variance": unc_var,
                    "implied_initial_variance_status": unc_status,
                    "error_message": "",
                }
            )
            if usable:
                valid_results.append(result)
            if attempt_idx == 0 and base_params is not None:
                for _ in range(max(0, n_starts - 1)):
                    start_values.append(
                        randomized_starting_values(base_params, vol_spec, dist, y, rng)
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
                    "implied_initial_variance": np.nan,
                    "implied_initial_variance_status": "fit_failed",
                    "error_message": str(exc),
                }
            )
            if attempt_idx == 0 and base_params is None:
                break

    if not valid_results:
        return None, attempts
    best = max(valid_results, key=lambda res: float(res.loglikelihood))
    return best, attempts


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
    unc_var: float,
    persistence: float,
    unc_status: str,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    diag = residual_diagnostics(result)
    stable_flag = bool(np.isfinite(persistence) and persistence < 1.0 - STABILITY_MARGIN)
    valid_attempts = [
        a for a in attempts if bool(a.get("optimizer_success")) and bool(a.get("stable_by_proxy"))
    ]
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
        "loglikelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "persistence_proxy": persistence,
        "stable_by_proxy": stable_flag,
        "implied_initial_variance": unc_var,
        "implied_initial_variance_status": unc_status,
        "initialization_method": "arch_default",
        "restart_attempts": len(attempts),
        "valid_restart_attempts": len(valid_attempts),
        "optimizer_success": bool(result.optimization_result.success),
        "convergence_flag": int(result.convergence_flag),
        "covariance_type": "robust",
        **diag,
    }


def parameter_rows(model_id: str, result: Any) -> list[dict[str, Any]]:
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
    return f"{x:.{digits}f}"


def _mean_label(row: pd.Series) -> str:
    mean_spec = str(row.get("mean_spec", ""))
    if mean_spec in MEAN_LABEL:
        return MEAN_LABEL[mean_spec]
    return str(row.get("mean_label", mean_spec))


def _candidate_pool(summary_df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    crit_values = pd.to_numeric(summary_df[criterion], errors="coerce")
    finite_criterion = np.isfinite(crit_values.to_numpy(dtype=float))
    pool = summary_df[
        (summary_df["optimizer_success"] == True)
        & finite_criterion
        & (summary_df["stable_by_proxy"] == True)
    ].copy()
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
        lines.append("")
        lines.append(
            "[^garch-aic-sample-note]: These numbers differ from earlier GARCH tables because "
            "earlier code used inconsistent effective sample sizes across model settings due "
            "to a sample-trimming error. The correction is discussed in the "
            "[Data Summary effective-sample section](../../DataSummary/README.md#effective-sample). "
            "The current estimates use the common **1969Q2--2022Q4** sample with **215 observations**."
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

    m = re.fullmatch(r"(alpha|beta|gamma)\[(\d+)\]", raw_name)
    if m:
        group, idx = m.groups()
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

    m = re.fullmatch(r"(alpha|gamma|beta)\[(\d+)\]", raw_name)
    if m:
        group, idx_s = m.groups()
        idx = int(idx_s)
        if group == "beta":
            return (1, 100 + idx, raw_name)
        group_order = {"alpha": 0, "gamma": 1}
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
        return f"{v:+.4f}"

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
                parts.append(f"{a:+.4f}\\,u_{{t-{i}}}^2")
        for k in range(1, q + 1):
            b = gp("beta", k)
            if np.isfinite(b):
                parts.append(f"{b:+.4f}\\,\\sigma_{{t-{k}}}^2")
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
                parts.append(f"{a:+.4f}\\,(u_{{t-{i}}}^+)^2")
            if np.isfinite(a) and np.isfinite(g):
                parts.append(f"{(a + g):+.4f}\\,(u_{{t-{i}}}^-)^2")
        for k in range(1, q + 1):
            b = gp("beta", k)
            if np.isfinite(b):
                parts.append(f"{b:+.4f}\\,\\sigma_{{t-{k}}}^2")
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
            parts.append(f"{a:+.4f}\\,(|z_{{t-{i}}}|-\\sqrt{{2/\\pi}})")
    for j in range(1, o + 1):
        g = gp("gamma", j)
        if np.isfinite(g):
            parts.append(f"{g:+.4f}\\,z_{{t-{j}}}")
    for k in range(1, q + 1):
        b = gp("beta", k)
        if np.isfinite(b):
            parts.append(f"{b:+.4f}\\,\\ln\\sigma_{{t-{k}}}^2")
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
            "For GJR-GARCH, $\\gamma_i-\\alpha_i$ is the raw `arch` indicator coefficient; "
            "the equation above reports the negative-shock coefficient $\\gamma_i$."
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


def build_unit_order_comparison_section(summary_df: pd.DataFrame, param_df: pd.DataFrame) -> str:
    loglik_values = pd.to_numeric(summary_df["loglikelihood"], errors="coerce")
    finite_loglik = np.isfinite(loglik_values.to_numpy(dtype=float))
    pool = summary_df[
        (summary_df["optimizer_success"] == True)
        & finite_loglik
        & (summary_df["vol_family"].isin(["GARCH", "GJR-GARCH"]))
        & (summary_df["p"] == 1)
        & (summary_df["q"] == 1)
    ].copy()

    lines: list[str] = []
    lines.append("## GARCH(1,1) and GJR-GARCH(1,1) Comparisons")
    lines.append("")
    lines.append(
        "These estimates are reported by mean process and innovation distribution because "
        "the GARCH(1,1) and GJR-GARCH(1,1) recursions are useful benchmarks for the BEGE specifications."
    )
    lines.append("")

    for mean_spec in MEAN_LABEL:
        mean_rows = pool[pool["mean_spec"] == mean_spec]
        if mean_rows.empty:
            continue
        lines.append(f"### {_mean_label(mean_rows.iloc[0])}")
        lines.append("")
        for dist in DISTRIBUTION_ORDER:
            dist_rows = mean_rows[mean_rows["distribution"] == dist]
            if dist_rows.empty:
                continue
            lines.append(f"#### {DISTRIBUTION_LABEL[dist]}")
            lines.append("")
            for family in ["GARCH", "GJR-GARCH"]:
                sub = dist_rows[dist_rows["vol_family"] == family]
                if sub.empty:
                    lines.append(f"##### {family}(1,1)")
                    lines.append("")
                    lines.append("No successful model.")
                    lines.append("")
                    continue
                row = sub.iloc[0]
                stable_note = ""
                if bool(row.get("stable_by_proxy", True)) is not True:
                    stable_note = " (persistence proxy not satisfied)"
                lines.append(f"##### {family}(1,1){stable_note}")
                lines.append("")
                lines.append(_model_detail_markdown(row, param_df))
                lines.append("")

    return "\n".join(lines).rstrip()


def build_excluded_models_section(failed_df: pd.DataFrame | None) -> str:
    if failed_df is None or failed_df.empty:
        return "\n".join(
            [
                "## Excluded Model Combinations",
                "",
                "No model combinations were excluded after the restart search.",
            ]
        )

    lines: list[str] = []
    lines.append("## Excluded Model Combinations")
    lines.append("")
    lines.append(
        "The combinations below did not produce an optimizer-converged fit satisfying "
        "the stationarity proxy with the `1e-6` margin after the restart search. "
        "They are omitted from the model-selection tables and retained in "
        "`garch_family_failed_models.csv` for audit."
    )
    lines.append("")
    lines.append(
        "| Model ID | Distribution | Mean process | Volatility process | Attempts | "
        "Optimizer-success attempts | Stable attempts | Best attempted LogLik |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for _, row in failed_df.iterrows():
        opt_attempts = int(row.get("optimizer_success_attempts", 0))
        stable_attempts = int(row.get("stable_attempts", 0))
        lines.append(
            "| "
            + f"`{row['model_id']}` | {_distribution_label(row)} | {_mean_label(row)} | "
            + f"{row['vol_spec']} | {int(row['attempts'])} | "
            + f"{opt_attempts} | {stable_attempts} | "
            + f"{_format_value(float(row['best_attempt_loglikelihood']))} |"
        )
    return "\n".join(lines)


def build_markdown_report(
    summary_df: pd.DataFrame,
    param_df: pd.DataFrame,
    failed_df: pd.DataFrame | None = None,
    n_starts: int = DEFAULT_N_STARTS,
    retry_failed_starts: int = DEFAULT_RETRY_FAILED_STARTS,
) -> str:
    lines: list[str] = []
    lines.append("```{raw:typst}")
    lines.append("#set page(margin: auto)")
    lines.append("```")
    lines.append("")
    lines.append("# Model Selection Report")
    lines.append("")
    lines.append(build_family_panel_table(summary_df, "aic"))
    lines.append("")
    lines.append(build_family_panel_table(summary_df, "bic"))
    lines.append("")

    lines.append("## Best Models by Criterion and Volatility Family")
    lines.append("")
    lines.append(
        "For each criterion and each volatility family, the selected model is the best "
        "across mean-process choices, orders, and distributions using stable & successful fits."
    )
    lines.append(
        f"The main result tables contain {len(summary_df)} accepted model combinations."
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
                lines.append("No successful model.")
                lines.append("")
                continue
            lines.append(f"#### {family}")
            lines.append(_model_detail_markdown(best, param_df))
            lines.append("")

    lines.append(build_unit_order_comparison_section(summary_df, param_df))
    lines.append("")
    lines.append("")

    lines.append(build_excluded_models_section(failed_df))
    lines.append("")
    lines.append("")

    lines.append("## EGARCH and Initialization Notes")
    lines.append("")
    lines.append("- Effective sample is fixed at 215 observations for all models (`hold_back=0` with explicit lag regressors in the mean equation).")
    lines.append("- In `arch`, EGARCH uses the package's centered term `|z|-sqrt(2/pi)` in the recursion for all distributions.")
    lines.append("- Under non-Gaussian errors (e.g., Student's t or Gaussian mixture), this is an intercept reparameterization; fitted dynamics and likelihood are still valid.")
    lines.append("- Conditional-variance recursion starts use the default initialization implemented by the `arch` package; this report no longer iterates the `backcast` to the parameter-implied unconditional variance.")
    lines.append(
        f"- Each model combination is estimated from the package default start plus "
        f"{max(0, n_starts - 1)} randomized feasible starts. If no stable "
        f"converged fit is found, the script tries {retry_failed_starts} additional "
        "retry starts using an independent model-specific retry seed."
    )
    lines.append("- Attempt-level diagnostics are saved in `garch_family_restart_attempts.csv`.")
    lines.append("- Only optimizer-converged fits that pass the stationarity proxy with a `1e-6` margin below one are collected in the main result tables.")
    lines.append("- Stability is monitored using a persistence proxy (`< 1 - 1e-6`):")
    lines.append("  GARCH uses `sum(alpha)+sum(beta)`, GJR uses `sum(alpha)+0.5*sum(gamma)+sum(beta)`, EGARCH uses `sum(beta)`.")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 11) Main estimation routine
# ============================================================
def run_estimation(
    data_path: Path,
    output_dir: Path,
    n_starts: int,
    retry_failed_starts: int,
    maxiter: int,
    seed: int,
) -> None:
    df = load_quarterly_data(data_path)
    mean_specs = build_mean_specs()
    vol_specs = build_vol_specs()

    summary_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    text_lines: list[str] = []

    model_counter = 0

    for dist in DISTRIBUTIONS:
        for mean_spec in mean_specs:
            y, x = prepare_mean_inputs(df, mean_spec)

            for vol_spec in vol_specs:
                model_counter += 1
                model_id = f"M{model_counter:03d}"

                try:
                    primary_rng = model_phase_rng(seed, model_counter, 0)
                    if dist in ("normal", "studentst"):
                        fit_func = make_arch_fit_func(
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
                        fit_func, init_sv = make_sparch_fit_func(
                            y=y,
                            x=x,
                            mean=mean_spec.mean,
                            lags=mean_spec.lags,
                            vol_spec=vol_spec,
                            hold_back=COMMON_HOLD_BACK,
                            maxiter=maxiter,
                        )

                    result, attempts = (
                        fit_with_restarts(
                            fit_func=fit_func,
                            vol_spec=vol_spec,
                            dist=dist,
                            y=y,
                            rng=primary_rng,
                            n_starts=n_starts,
                            initial_starting_values=init_sv,
                            phase="primary",
                        )
                    )

                    if result is None and retry_failed_starts > 0:
                        retry_rng = model_phase_rng(seed, model_counter, 1)
                        retry_result, retry_attempts = fit_with_restarts(
                            fit_func=fit_func,
                            vol_spec=vol_spec,
                            dist=dist,
                            y=y,
                            rng=retry_rng,
                            n_starts=retry_failed_starts,
                            initial_starting_values=init_sv,
                            phase="retry",
                            attempt_offset=len(attempts),
                        )
                        attempts.extend(retry_attempts)
                        result = retry_result

                    attempt_rows.extend(
                        attempt_detail_rows(
                            model_id=model_id,
                            distribution=dist,
                            mean_spec=mean_spec,
                            vol_spec=vol_spec,
                            attempts=attempts,
                        )
                    )

                    if result is None:
                        failed_rows.append(
                            {
                                "model_id": model_id,
                                "distribution": dist,
                                "distribution_label": DISTRIBUTION_LABEL[dist],
                                "mean_spec": mean_spec.name,
                                "mean_label": MEAN_LABEL[mean_spec.name],
                                "vol_family": vol_spec.family,
                                "vol_spec": vol_spec.label,
                                "vol_order": f"({vol_spec.p},{vol_spec.q})",
                                "attempts": len(attempts),
                                "optimizer_success_attempts": attempt_count(
                                    attempts, "optimizer_success"
                                ),
                                "stable_attempts": attempt_count(attempts, "stable_by_proxy"),
                                "best_attempt_loglikelihood": best_attempt_loglikelihood(
                                    attempts
                                ),
                                "best_optimizer_success_loglikelihood": best_attempt_loglikelihood(
                                    attempts, optimizer_success_only=True
                                ),
                                "last_error_message": attempts[-1]["error_message"]
                                if attempts
                                else "no attempts",
                            }
                        )
                        text_lines.append(
                            f"{model_id} dist={dist:<10} mean={mean_spec.name:<16} vol={vol_spec.label:<15} "
                            f"FAILED_STABLE_CONVERGED attempts={len(attempts)}"
                        )
                        continue

                    unc_var, persistence, unc_status = implied_initial_variance(
                        result.params, vol_spec
                    )

                    srow = summary_row(
                        model_id=model_id,
                        distribution=dist,
                        mean_spec=mean_spec,
                        vol_spec=vol_spec,
                        result=result,
                        unc_var=unc_var,
                        persistence=persistence,
                        unc_status=unc_status,
                        attempts=attempts,
                    )
                    summary_rows.append(srow)
                    param_rows.extend(parameter_rows(model_id, result))

                    text_lines.append(
                        f"{model_id} dist={dist:<10} mean={mean_spec.name:<16} vol={vol_spec.label:<15} "
                        f"LogLik={result.loglikelihood:.6f} AIC={result.aic:.6f} BIC={result.bic:.6f} "
                        f"valid_starts={srow['valid_restart_attempts']}/{srow['restart_attempts']}"
                    )

                except Exception as exc:
                    failed_rows.append(
                        {
                            "model_id": model_id,
                            "distribution": dist,
                            "distribution_label": DISTRIBUTION_LABEL[dist],
                            "mean_spec": mean_spec.name,
                            "mean_label": MEAN_LABEL[mean_spec.name],
                            "vol_family": vol_spec.family,
                            "vol_spec": vol_spec.label,
                            "vol_order": f"({vol_spec.p},{vol_spec.q})",
                            "attempts": 0,
                            "optimizer_success_attempts": 0,
                            "stable_attempts": 0,
                            "best_attempt_loglikelihood": np.nan,
                            "best_optimizer_success_loglikelihood": np.nan,
                            "last_error_message": str(exc),
                        }
                    )
                    text_lines.append(
                        f"{model_id} dist={dist:<10} mean={mean_spec.name:<16} vol={vol_spec.label:<15} FAILED: {exc}"
                    )

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

    summary_csv = output_dir / "garch_family_estimation_summary.csv"
    params_csv = output_dir / "garch_family_estimation_parameters.csv"
    attempts_csv = output_dir / "garch_family_restart_attempts.csv"
    failed_csv = output_dir / "garch_family_failed_models.csv"
    summary_txt = output_dir / "garch_family_estimation_summary.txt"
    report_md = output_dir / "garch_family_best_models.md"

    summary_df.to_csv(summary_csv, index=False)
    param_df.to_csv(params_csv, index=False)
    attempt_df.to_csv(attempts_csv, index=False)
    pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)

    txt_header = [
        "GARCH-family estimation summary (accepted stable and converged models)",
        "",
        "Columns in line printout: model_id, distribution, mean, vol, LogLik, AIC, BIC",
        "",
    ]
    summary_txt.write_text("\n".join(txt_header + text_lines) + "\n", encoding="utf-8")

    failed_df = pd.DataFrame(failed_rows)
    report_text = build_markdown_report(
        summary_df,
        param_df,
        failed_df,
        n_starts=n_starts,
        retry_failed_starts=retry_failed_starts,
    )
    report_md.write_text(report_text + "\n", encoding="utf-8")

    print("\nSaved:")
    print(f"  - {summary_csv}")
    print(f"  - {params_csv}")
    print(f"  - {attempts_csv}")
    print(f"  - {failed_csv}")
    print(f"  - {summary_txt}")
    print(f"  - {report_md}")
    print("\n" + "\n".join(text_lines))


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
        help=f"Path to quarterly data pickle (default: {DEFAULT_DATA_PATH})",
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
            "Primary optimizer starts per model combination, including the default ARCH start "
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
        default=20260622,
        help="Random seed used for restart starting values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
