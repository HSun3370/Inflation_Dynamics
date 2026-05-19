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
COMMON_HOLD_BACK = 2


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
# 4) Mixture-of-Normals distribution for ARCH
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
        self._name = "Mixture of two Normal distributions"
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
            mean="ARX",
            lags=1,
            x_cols=["SPF"],
        ),
        MeanSpec(
            name="ARX_2_1",
            y_col="Inflation",
            y_transform="identity",
            mean="ARX",
            lags=[1, 2],
            x_cols=["SPF"],
        ),
        MeanSpec(
            name="ARX_2_2",
            y_col="Inflation",
            y_transform="identity",
            mean="ARX",
            lags=[1, 2],
            x_cols=["SPF", "SPF_lag_1"],
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
    return {
        "model_id": model_id,
        "distribution": distribution,
        "mean_spec": mean_spec.name,
        "vol_family": vol_spec.family,
        "vol_spec": vol_spec.label,
        "p": vol_spec.p,
        "o": vol_spec.o,
        "q": vol_spec.q,
        "nobs": int(result.nobs),
        "loglikelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "persistence_proxy": persistence,
        "implied_initial_variance": unc_var,
        "implied_initial_variance_status": unc_status,
        "initial_variance_backcast_used": backcast,
        "backcast_iterations": backcast_iterations,
        "backcast_converged": bool(backcast_converged),
        "optimizer_success": bool(result.optimization_result.success),
        "convergence_flag": int(result.convergence_flag),
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
def _model_detail_markdown(best_row: pd.Series, param_df: pd.DataFrame) -> str:
    mid = best_row["model_id"]
    psub = param_df[param_df["model_id"] == mid].copy()
    psub = psub.sort_values("parameter")

    lines: list[str] = []
    lines.append(f"- Model ID: `{mid}`")
    lines.append(f"- Distribution: `{best_row['distribution']}`")
    lines.append(f"- Mean process: `{best_row['mean_spec']}`")
    lines.append(f"- Volatility: `{best_row['vol_spec']}`")
    lines.append(f"- nobs: `{int(best_row['nobs'])}`")
    lines.append(f"- Log-likelihood: `{best_row['loglikelihood']:.6f}`")
    lines.append(f"- AIC: `{best_row['aic']:.6f}`")
    lines.append(f"- BIC: `{best_row['bic']:.6f}`")
    lines.append(f"- Persistence proxy: `{best_row['persistence_proxy']}`")
    lines.append(f"- Implied initial variance: `{best_row['implied_initial_variance']}`")
    lines.append(f"- Variance status: `{best_row['implied_initial_variance_status']}`")
    lines.append(f"- Optimizer success: `{best_row['optimizer_success']}`")
    lines.append(f"- Convergence flag: `{int(best_row['convergence_flag'])}`")
    lines.append("")
    lines.append("| Parameter | Coef | Std Err | t-value | p-value |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in psub.iterrows():
        lines.append(
            f"| {r['parameter']} | {r['coef']:.6f} | {r['std_err']:.6f} | {r['t_value']:.6f} | {r['p_value']:.6f} |"
        )
    return "\n".join(lines)


def build_markdown_report(summary_df: pd.DataFrame, param_df: pd.DataFrame) -> str:
    ok = summary_df[summary_df["optimizer_success"] == True].copy()
    if ok.empty:
        ok = summary_df.copy()

    best_aic = ok.loc[ok["aic"].idxmin()]
    best_bic = ok.loc[ok["bic"].idxmin()]

    lines: list[str] = []
    lines.append("# Best Model Report")
    lines.append("")
    lines.append("Selection uses all estimated models. Prefer optimizer-successful models when available.")
    lines.append("")

    lines.append("## Best by AIC")
    lines.append(_model_detail_markdown(best_aic, param_df))
    lines.append("")

    lines.append("## Best by BIC")
    lines.append(_model_detail_markdown(best_bic, param_df))
    lines.append("")

    lines.append("## Notes on Initialization")
    lines.append("- `GARCH`: uses exact unconditional variance `omega / (1 - sum(alpha) - sum(beta))` when stationary.")
    lines.append("- `GJR-GARCH`: uses `omega / (1 - sum(alpha) - 0.5*sum(gamma) - sum(beta))` under symmetric innovations.")
    lines.append("- `EGARCH`: uses proxy `exp(omega / (1 - sum(beta)))` from log-variance recursion.")
    lines.append("- In `arch`, `backcast` sets recursion start; it is not automatically replaced by model-implied unconditional variance during optimization.")
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
