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

    def _fit(backcast: float, starting_values: np.ndarray | None):
        kwargs: dict[str, Any] = {
            "disp": "off",
            "update_freq": 0,
            "cov_type": "robust",
            "show_warning": False,
            "options": {"maxiter": 3000},
            "backcast": float(backcast),
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
        options={"maxiter": 3000},
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

    def _fit(backcast: float, starting_values: np.ndarray | None):
        kwargs: dict[str, Any] = {
            "disp": "off",
            "update_freq": 0,
            "cov_type": "robust",
            "show_warning": False,
            "options": {"maxiter": 4000},
            "backcast": float(backcast),
        }
        kwargs["starting_values"] = initial if starting_values is None else starting_values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=StartingValueWarning)
            return mix_model.fit(**kwargs)

    return _fit, initial


# ============================================================
# 8) Iterative initialization + estimation
# ============================================================
def fit_with_iterative_initialization(
    fit_func: Callable[[float, np.ndarray | None], Any],
    vol_spec: VolSpec,
    y: pd.Series,
    initial_starting_values: np.ndarray | None = None,
    max_iter: int = 10,
    tol: float = 1e-8,
):
    backcast = float(np.var(np.asarray(y), ddof=1))
    sv = None if initial_starting_values is None else np.asarray(initial_starting_values)

    iterations = 0
    converged_backcast = False
    result = None
    unc_var = np.nan
    persistence = np.nan
    unc_status = "not_computed"

    for i in range(max_iter):
        iterations = i + 1
        result = fit_func(backcast, sv)
        unc_var, persistence, unc_status = implied_initial_variance(result.params, vol_spec)
        sv = np.asarray(result.params, dtype=float)

        if not np.isfinite(unc_var) or unc_var <= 0.0:
            break

        rel_diff = abs(unc_var - backcast) / max(1.0, abs(backcast))
        backcast = float(unc_var)
        if rel_diff <= tol:
            converged_backcast = True
            break

    if result is None:
        raise RuntimeError("Model fit did not return a result.")

    result = fit_func(backcast, sv)
    unc_var, persistence, unc_status = implied_initial_variance(result.params, vol_spec)
    return result, float(backcast), float(unc_var), float(persistence), unc_status, iterations, converged_backcast


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
    backcast: float,
    unc_var: float,
    persistence: float,
    unc_status: str,
    backcast_iterations: int,
    backcast_converged: bool,
) -> dict[str, Any]:
    diag = residual_diagnostics(result)
    stable_flag = bool(np.isfinite(persistence) and persistence < 1.0)
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
        "initial_variance_backcast_used": backcast,
        "backcast_iterations": backcast_iterations,
        "backcast_converged": bool(backcast_converged),
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


# ============================================================
# 10) Markdown best-model reporting
# ============================================================
def _format_value(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _candidate_pool(summary_df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    pool = summary_df[
        (summary_df["optimizer_success"] == True)
        & np.isfinite(summary_df[criterion])
        & (summary_df["stable_by_proxy"] == True)
    ].copy()
    if pool.empty:
        pool = summary_df[
            (summary_df["optimizer_success"] == True) & np.isfinite(summary_df[criterion])
        ].copy()
    if pool.empty:
        pool = summary_df[np.isfinite(summary_df[criterion])].copy()
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
        title = "**Table 2: Model selection by AIC: GARCH vs GJR vs EGARCH**"
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

        gar_mean = gar["mean_label"] if gar is not None else "NA"
        gar_vol = gar["vol_order"] if gar is not None else "NA"
        gjr_mean = gjr["mean_label"] if gjr is not None else "NA"
        gjr_vol = gjr["vol_order"] if gjr is not None else "NA"
        ega_mean = ega["mean_label"] if ega is not None else "NA"
        ega_vol = ega["vol_order"] if ega is not None else "NA"

        lines.append(
            "| "
            + f"{DISTRIBUTION_LABEL[dist]} | {gar_mean} | {gar_vol} | {crit_fmt(gar)} | "
            + f"{gjr_mean} | {gjr_vol} | {crit_fmt(gjr)} | "
            + f"{ega_mean} | {ega_vol} | {crit_fmt(ega)} |"
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
        if family == "GJR-GARCH":
            group_order = {"alpha": 0, "gamma": 1, "beta": 2}
        else:
            group_order = {"alpha": 0, "gamma": 1, "beta": 2}
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

    c, sc = _param_lookup(psub, "Const")
    i1, si1 = _param_lookup(psub, "Inflation_lag_1")
    spf, sspf = _param_lookup(psub, "SPF")

    if m == "ARX_1_1":
        eq = (
            "$$\n"
            f"\\hat{{\\pi}}_{{t+1}} = {c:.4f} {s(i1)}\\,\\pi_t {s(spf)}\\,SPF_t + \\mu_{{t+1}}\n"
            "$$"
        )
        se = f"Robust SE: $c$ ({sc:.4f}), $\\rho_1$ ({si1:.4f}), $\\phi_1$ ({sspf:.4f})."
        return eq + "\n\n" + se

    i2, si2 = _param_lookup(psub, "Inflation_lag_2")
    if m == "ARX_2_1":
        eq = (
            "$$\n"
            f"\\hat{{\\pi}}_{{t+1}} = {c:.4f} {s(i1)}\\,\\pi_t {s(i2)}\\,\\pi_{{t-1}} {s(spf)}\\,SPF_t + \\mu_{{t+1}}\n"
            "$$"
        )
        se = (
            f"Robust SE: $c$ ({sc:.4f}), $\\rho_1$ ({si1:.4f}), "
            f"$\\rho_2$ ({si2:.4f}), $\\phi_1$ ({sspf:.4f})."
        )
        return eq + "\n\n" + se

    spf1, sspf1 = _param_lookup(psub, "SPF_lag_1")
    eq = (
        "$$\n"
        f"\\hat{{\\pi}}_{{t+1}} = {c:.4f} {s(i1)}\\,\\pi_t {s(i2)}\\,\\pi_{{t-1}} "
        f"{s(spf)}\\,SPF_t {s(spf1)}\\,SPF_{{t-1}} + \\mu_{{t+1}}\n"
        "$$"
    )
    se = (
        f"Robust SE: $c$ ({sc:.4f}), $\\rho_1$ ({si1:.4f}), "
        f"$\\rho_2$ ({si2:.4f}), $\\phi_1$ ({sspf:.4f}), $\\phi_2$ ({sspf1:.4f})."
    )
    return eq + "\n\n" + se


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
    mid = best_row["model_id"]
    family = str(best_row["vol_family"])
    psub = param_df[param_df["model_id"] == mid].copy()
    if not psub.empty:
        psub["_sort_key"] = psub["parameter"].map(lambda x: _parameter_sort_key(str(x), family))
        psub = psub.sort_values("_sort_key")

    lines: list[str] = []
    lines.append(f"- Model ID: `{mid}`")
    lines.append(f"- Distribution: {_distribution_label(best_row)}")
    lines.append(f"- Mean process: {best_row['mean_label']} (`{best_row['mean_spec']}`)")
    lines.append(f"- Volatility process: {best_row['vol_spec']}")
    lines.append("")
    lines.append(_mean_equation_markdown(best_row, psub))
    lines.append("")
    lines.append(_volatility_equation_markdown(best_row, psub))
    lines.append("")
    lines.append(f"- Number of observations: **{int(best_row['nobs'])}**")
    lines.append(f"- Log-likelihood: **{best_row['loglikelihood']:.6f}**")
    lines.append(f"- AIC: **{best_row['aic']:.6f}**, BIC: **{best_row['bic']:.6f}**")
    lines.append(f"- Optimizer success: **{bool(best_row['optimizer_success'])}**")
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
            f"| {label} | {r['coef']:.6f} | {r['std_err']:.6f} | {r['t_value']:.6f} | {r['p_value']:.6f} |"
        )
    return "\n".join(lines)


def build_markdown_report(summary_df: pd.DataFrame, param_df: pd.DataFrame) -> str:
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

    lines.append("## EGARCH and Initialization Notes")
    lines.append("")
    lines.append("- Effective sample is fixed at 215 observations for all models (`hold_back=0` with explicit lag regressors in the mean equation).")
    lines.append("- In `arch`, EGARCH uses the package's centered term `|z|-sqrt(2/pi)` in the recursion for all distributions.")
    lines.append("- Under non-Gaussian errors (e.g., Student's t or Gaussian mixture), this is an intercept reparameterization; fitted dynamics and likelihood are still valid.")
    lines.append("- The recursion start (`backcast`) is updated iteratively during estimation in this script: fit -> implied initial variance from current parameters -> refit.")
    lines.append("- Stability is monitored using a persistence proxy (`<1`):")
    lines.append("  GARCH uses `sum(alpha)+sum(beta)`, GJR uses `sum(alpha)+0.5*sum(gamma)+sum(beta)`, EGARCH uses `sum(beta)`.")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 11) Main estimation routine
# ============================================================
def run_estimation(data_path: Path, output_dir: Path) -> None:
    df = load_quarterly_data(data_path)
    mean_specs = build_mean_specs()
    vol_specs = build_vol_specs()

    summary_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    text_lines: list[str] = []

    model_counter = 0

    for dist in DISTRIBUTIONS:
        for mean_spec in mean_specs:
            y, x = prepare_mean_inputs(df, mean_spec)

            for vol_spec in vol_specs:
                model_counter += 1
                model_id = f"M{model_counter:03d}"

                try:
                    if dist in ("normal", "studentst"):
                        fit_func = make_arch_fit_func(
                            y=y,
                            x=x,
                            mean=mean_spec.mean,
                            lags=mean_spec.lags,
                            vol_spec=vol_spec,
                            dist=dist,
                            hold_back=COMMON_HOLD_BACK,
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
                        )

                    result, backcast, unc_var, persistence, unc_status, n_iter, bc_conv = (
                        fit_with_iterative_initialization(
                            fit_func=fit_func,
                            vol_spec=vol_spec,
                            y=y,
                            initial_starting_values=init_sv,
                        )
                    )

                    srow = summary_row(
                        model_id=model_id,
                        distribution=dist,
                        mean_spec=mean_spec,
                        vol_spec=vol_spec,
                        result=result,
                        backcast=backcast,
                        unc_var=unc_var,
                        persistence=persistence,
                        unc_status=unc_status,
                        backcast_iterations=n_iter,
                        backcast_converged=bc_conv,
                    )
                    summary_rows.append(srow)
                    param_rows.extend(parameter_rows(model_id, result))

                    text_lines.append(
                        f"{model_id} dist={dist:<10} mean={mean_spec.name:<16} vol={vol_spec.label:<15} "
                        f"LogLik={result.loglikelihood:.6f} AIC={result.aic:.6f} BIC={result.bic:.6f}"
                    )

                except Exception as exc:
                    summary_rows.append(
                        {
                            "model_id": model_id,
                            "distribution": dist,
                            "mean_spec": mean_spec.name,
                            "vol_family": vol_spec.family,
                            "vol_spec": vol_spec.label,
                            "p": vol_spec.p,
                            "o": vol_spec.o,
                            "q": vol_spec.q,
                            "nobs": np.nan,
                            "loglikelihood": np.nan,
                            "aic": np.nan,
                            "bic": np.nan,
                            "persistence_proxy": np.nan,
                            "implied_initial_variance": np.nan,
                            "implied_initial_variance_status": "fit_failed",
                            "initial_variance_backcast_used": np.nan,
                            "backcast_iterations": np.nan,
                            "backcast_converged": False,
                            "optimizer_success": False,
                            "convergence_flag": np.nan,
                            "error_message": str(exc),
                        }
                    )
                    text_lines.append(
                        f"{model_id} dist={dist:<10} mean={mean_spec.name:<16} vol={vol_spec.label:<15} FAILED: {exc}"
                    )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["distribution", "mean_spec", "vol_family", "p", "q"])
    param_df = pd.DataFrame(param_rows).sort_values(["model_id", "parameter"]) if param_rows else pd.DataFrame()

    summary_csv = output_dir / "garch_family_estimation_summary.csv"
    params_csv = output_dir / "garch_family_estimation_parameters.csv"
    summary_txt = output_dir / "garch_family_estimation_summary.txt"
    report_md = output_dir / "garch_family_best_models.md"

    summary_df.to_csv(summary_csv, index=False)
    param_df.to_csv(params_csv, index=False)

    txt_header = [
        "GARCH-family estimation summary (all models)",
        "",
        "Columns in line printout: model_id, distribution, mean, vol, LogLik, AIC, BIC",
        "",
    ]
    summary_txt.write_text("\n".join(txt_header + text_lines) + "\n", encoding="utf-8")

    report_text = build_markdown_report(summary_df, param_df)
    report_md.write_text(report_text + "\n", encoding="utf-8")

    print("\nSaved:")
    print(f"  - {summary_csv}")
    print(f"  - {params_csv}")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_estimation(data_path=args.data_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
