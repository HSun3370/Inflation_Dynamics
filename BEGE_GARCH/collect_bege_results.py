"""Collect local BEGE random-draw summaries into tracked result tables.

The raw optimizer outputs live in ignored RandomDraw_* folders. This script
loads their summary_draws.csv files, applies the documented BEGE parameter
checks, and writes compact result files that are appropriate to keep in Git.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BEGE_DIR = SCRIPT_PATH.parent
ROOT = BEGE_DIR.parent
RESULTS_DIR = BEGE_DIR / "results"
DATA_PATH = ROOT / "Aggregate_CPI_inflation.pkl"

CANONICAL_SAMPLE_N = 215
CANONICAL_SAMPLE_START = "1969Q2"
CANONICAL_SAMPLE_END = "2022Q4"
VARIANCE_BOUND = 0.75
SHAPE_CAP = 200.0

REPORT_PREAMBLE = "```{raw:typst}\n#set page(margin: auto)\n```\n\n"


SOURCE_CONFIG: dict[str, dict[str, str]] = {
    "RandomDraw_Constant": {
        "family": "Constant BEGE",
        "scripts": "BEGE_constant.py",
        "notes": "Invariant shape parameters.",
    },
    "RandomDraw_Symmetric_Oct": {
        "family": "Shared-GJR BEGE",
        "scripts": "BEGE_symmetric1.py; BEGE_symmetric2.py",
        "notes": "Separate p0, n0, sigma_p, and sigma_n with shared GJR loadings.",
    },
    "RandomDraw_Symmetric_Sep": {
        "family": "Symmetric BEGE archive",
        "scripts": "BEGE_symmetric*.py",
        "notes": "Earlier shared-shape run retained for audit.",
    },
    "RandomDraw_PerfectSymmetric": {
        "family": "Perfect-symmetric BEGE archive",
        "scripts": "BEGE_symmetric*.py",
        "notes": "Earlier shared shape and shared scale run retained for audit.",
    },
    "RandomDraw_BG_GARCH": {
        "family": "Bad/Good BEGE-GARCH",
        "scripts": "BG_GJR1.py; BG_GJR2.py",
        "notes": "Good and bad shape processes respond symmetrically to shock sign.",
    },
    "RandomDraw_ID": {
        "family": "Inflation/Deflation BEGE-GARCH",
        "scripts": "ID_GJR1.py; ID_GJR2.py",
        "notes": "Positive shocks load on p_t and negative shocks load on n_t.",
    },
    "RandomDraw_GJR": {
        "family": "Full BEGE-GJR archive",
        "scripts": "BEGE_GJR1.py; BEGE_GJR2.py",
        "notes": "Earlier full-GJR local run retained for audit.",
    },
    "RandomDraw_GJR_Oct": {
        "family": "Full BEGE-GJR",
        "scripts": "BEGE_GJR1.py; BEGE_GJR2.py",
        "notes": "Full p_t and n_t GJR recursions.",
    },
    "RandomDraw_Full_Dec": {
        "family": "Full BEGE-GJR near-start archive",
        "scripts": "BEGE_GJR1.py; BEGE_GJR2.py",
        "notes": "Later full-GJR local run retained for audit.",
    },
}

MEAN_LABELS = {
    "constant": "Constant",
    "ARX11": "ARX(1,1)",
    "ARX21": "ARX(2,1)",
    "ARX22": "ARX(2,2)",
}

MEAN_PARAM_COUNT = {
    "constant": 0,
    "ARX11": 3,
    "ARX21": 4,
    "ARX22": 5,
}


@dataclass(frozen=True)
class SampleInfo:
    nobs: int
    start: str
    end: str

    @property
    def caveat(self) -> str:
        if (
            self.nobs == CANONICAL_SAMPLE_N
            and self.start == CANONICAL_SAMPLE_START
            and self.end == CANONICAL_SAMPLE_END
        ):
            return "matches canonical project sample"
        return (
            "differs from canonical project sample "
            f"({CANONICAL_SAMPLE_START}-{CANONICAL_SAMPLE_END}, "
            f"{CANONICAL_SAMPLE_N} observations)"
        )


def quarter_label(row: pd.Series) -> str:
    return f"{int(row['Year'])}Q{int(row['Quarter'])}"


def load_data() -> tuple[pd.DataFrame, SampleInfo]:
    data = pd.read_pickle(DATA_PATH)
    start = quarter_label(data.iloc[0])
    end = quarter_label(data.iloc[-1])
    return data, SampleInfo(nobs=int(len(data)), start=start, end=end)


def clean_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out


def named_value(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row.index:
            return clean_float(row[name])
    return np.nan


def param_value(row: pd.Series, index: int) -> float:
    return named_value(row, f"param_{index}")


def mean_params_from_row(row: pd.Series, mean_folder: str) -> dict[str, float]:
    params: dict[str, float] = {
        "mean_const": np.nan,
        "mean_infl_lag1": np.nan,
        "mean_infl_lag2": np.nan,
        "mean_spf": np.nan,
        "mean_spf_lag1": np.nan,
    }
    if mean_folder == "constant":
        return params

    if "const" in row.index:
        params["mean_const"] = named_value(row, "const")
        params["mean_infl_lag1"] = named_value(row, "Infl(1)")
        params["mean_spf"] = named_value(row, "SPF")
        if mean_folder in {"ARX21", "ARX22"}:
            params["mean_infl_lag2"] = named_value(row, "Infl(2)")
        if mean_folder == "ARX22":
            params["mean_spf_lag1"] = named_value(row, "SPF.lag(1)")
        return params

    if mean_folder == "ARX11":
        params["mean_const"] = param_value(row, 1)
        params["mean_infl_lag1"] = param_value(row, 2)
        params["mean_spf"] = param_value(row, 3)
    elif mean_folder == "ARX21":
        params["mean_const"] = param_value(row, 1)
        params["mean_infl_lag1"] = param_value(row, 2)
        params["mean_infl_lag2"] = param_value(row, 3)
        params["mean_spf"] = param_value(row, 4)
    elif mean_folder == "ARX22":
        params["mean_const"] = param_value(row, 1)
        params["mean_infl_lag1"] = param_value(row, 2)
        params["mean_infl_lag2"] = param_value(row, 3)
        params["mean_spf"] = param_value(row, 4)
        params["mean_spf_lag1"] = param_value(row, 5)
    return params


def residuals_for_row(data: pd.DataFrame, row: pd.Series, mean_folder: str) -> np.ndarray:
    if mean_folder == "constant":
        return data["Inflation shock"].to_numpy(float)

    params = mean_params_from_row(row, mean_folder)
    y = data["Inflation"].to_numpy(float)
    x = data["Forecasted inflation"].to_numpy(float)
    yhat = np.zeros_like(y)

    if mean_folder == "ARX11":
        beta0 = params["mean_const"]
        phi1 = params["mean_infl_lag1"]
        theta1 = params["mean_spf"]
        yhat[0] = beta0 + theta1 * x[0]
        yhat[1:] = beta0 + phi1 * y[:-1] + theta1 * x[1:]
    elif mean_folder == "ARX21":
        beta0 = params["mean_const"]
        phi1 = params["mean_infl_lag1"]
        phi2 = params["mean_infl_lag2"]
        theta1 = params["mean_spf"]
        yhat[0] = beta0 + theta1 * x[0]
        yhat[1] = beta0 + phi1 * y[0] + theta1 * x[1]
        yhat[2:] = beta0 + phi1 * y[1:-1] + phi2 * y[:-2] + theta1 * x[2:]
    elif mean_folder == "ARX22":
        beta0 = params["mean_const"]
        phi1 = params["mean_infl_lag1"]
        phi2 = params["mean_infl_lag2"]
        theta1 = params["mean_spf"]
        theta2 = params["mean_spf_lag1"]
        yhat[0] = beta0 + theta1 * x[0]
        yhat[1] = beta0 + phi1 * y[0] + theta1 * x[1] + theta2 * x[0]
        yhat[2:] = (
            beta0
            + phi1 * y[1:-1]
            + phi2 * y[:-2]
            + theta1 * x[2:]
            + theta2 * x[1:-1]
        )
    else:
        raise ValueError(f"Unknown mean folder: {mean_folder}")

    return y - yhat


def gjr_recursion(
    residuals: np.ndarray,
    constant: float,
    rho: float,
    phi_plus: float,
    phi_minus: float,
    sigma: float,
) -> np.ndarray:
    out = np.empty(len(residuals), dtype=float)
    denom = 1.0 - rho - 0.5 * (phi_plus + phi_minus)
    backcast = constant / denom if denom > 1e-12 else 1e-4
    out[0] = max(backcast, 1e-4)

    scale = 2.0 * sigma * sigma
    for t in range(1, len(residuals)):
        phi = phi_plus if residuals[t - 1] > 0 else phi_minus
        value = constant + rho * out[t - 1] + phi * residuals[t - 1] ** 2 / scale
        out[t] = max(value, 1e-4)
    return out


def blank_params(row: pd.Series, mean_folder: str) -> dict[str, float]:
    params = mean_params_from_row(row, mean_folder)
    params.update(
        {
            "p_bar": np.nan,
            "n_bar": np.nan,
            "p0": np.nan,
            "n0": np.nan,
            "sym_cont": np.nan,
            "rho": np.nan,
            "rho_p": np.nan,
            "rho_n": np.nan,
            "phi_plus": np.nan,
            "phi_minus": np.nan,
            "phi_p": np.nan,
            "phi_n": np.nan,
            "phi_p_plus": np.nan,
            "phi_p_minus": np.nan,
            "phi_n_plus": np.nan,
            "phi_n_minus": np.nan,
            "sigma_p": np.nan,
            "sigma_n": np.nan,
            "shared_sigma": np.nan,
        }
    )
    return params


def standardized_params(source_folder: str, mean_folder: str, row: pd.Series) -> dict[str, float]:
    params = blank_params(row, mean_folder)
    offset = MEAN_PARAM_COUNT[mean_folder]

    if source_folder == "RandomDraw_Constant":
        params["p_bar"] = param_value(row, offset + 1)
        params["n_bar"] = param_value(row, offset + 2)
        params["sigma_p"] = param_value(row, offset + 3)
        params["sigma_n"] = param_value(row, offset + 4)
    elif source_folder in {"RandomDraw_PerfectSymmetric", "RandomDraw_Symmetric_Sep"}:
        params["sym_cont"] = param_value(row, offset + 1)
        params["rho"] = param_value(row, offset + 2)
        params["phi_plus"] = param_value(row, offset + 3)
        params["phi_minus"] = param_value(row, offset + 4)
        params["shared_sigma"] = param_value(row, offset + 5)
    elif source_folder == "RandomDraw_Symmetric_Oct":
        params["p0"] = named_value(row, "p0")
        params["n0"] = named_value(row, "n0")
        params["rho"] = named_value(row, "rho")
        params["phi_plus"] = named_value(row, "phi+", "phi⁺")
        params["phi_minus"] = named_value(row, "phi-", "phi⁻")
        params["sigma_p"] = named_value(row, "sigma_p", "sigma+", "σ₊")
        params["sigma_n"] = named_value(row, "sigma_n", "sigma-", "σ₋")
    elif source_folder == "RandomDraw_BG_GARCH":
        params["p0"] = named_value(row, "p0")
        params["n0"] = named_value(row, "n0")
        params["rho_p"] = named_value(row, "rho_p")
        params["rho_n"] = named_value(row, "rho_n")
        params["phi_p"] = named_value(row, "phi_p")
        params["phi_n"] = named_value(row, "phi_n")
        params["sigma_p"] = named_value(row, "sigma_p", "sigma+", "σ₊")
        params["sigma_n"] = named_value(row, "sigma_n", "sigma-", "σ₋")
    elif source_folder == "RandomDraw_ID":
        params["p0"] = named_value(row, "p0")
        params["n0"] = named_value(row, "n0")
        params["rho_p"] = named_value(row, "rho_p")
        params["rho_n"] = named_value(row, "rho_n")
        params["phi_p_plus"] = named_value(row, "phi_p+", "phi_p_plus")
        params["phi_n_minus"] = named_value(row, "phi_n-", "phi_n_minus")
        params["sigma_p"] = named_value(row, "sigma_p", "sigma+", "σ₊")
        params["sigma_n"] = named_value(row, "sigma_n", "sigma-", "σ₋")
    elif source_folder in {"RandomDraw_GJR", "RandomDraw_GJR_Oct", "RandomDraw_Full_Dec"}:
        params["p0"] = named_value(row, "p0")
        params["n0"] = named_value(row, "n0")
        params["rho_p"] = named_value(row, "rho_p")
        params["rho_n"] = named_value(row, "rho_n")
        params["phi_p_plus"] = named_value(row, "phi_p+", "phi_p⁺", "phi_p_plus")
        params["phi_p_minus"] = named_value(row, "phi_p-", "phi_p⁻", "phi_p_minus")
        params["phi_n_plus"] = named_value(row, "phi_n+", "phi_n⁺", "phi_n_plus")
        params["phi_n_minus"] = named_value(row, "phi_n-", "phi_n⁻", "phi_n_minus")
        params["sigma_p"] = named_value(row, "sigma_p", "sigma+", "σ₊")
        params["sigma_n"] = named_value(row, "sigma_n", "sigma-", "σ₋")
    return params


def finite_values(*values: float) -> bool:
    return all(np.isfinite(value) for value in values)


def evaluate_checks(
    data: pd.DataFrame,
    source_folder: str,
    mean_folder: str,
    row: pd.Series,
) -> tuple[bool, str, float, float, float]:
    params = standardized_params(source_folder, mean_folder, row)
    residuals = residuals_for_row(data, row, mean_folder)
    checks: list[tuple[str, bool]] = []
    max_p = np.nan
    max_n = np.nan
    variance_ref = np.nan

    if source_folder == "RandomDraw_Constant":
        p_bar = params["p_bar"]
        n_bar = params["n_bar"]
        sigma_p = params["sigma_p"]
        sigma_n = params["sigma_n"]
        variance_ref = sigma_p * sigma_p * p_bar + sigma_n * sigma_n * n_bar
        max_p = p_bar
        max_n = n_bar
        checks.extend(
            [
                ("p_bar_bounds", 0.1 < p_bar < 10.0),
                ("n_bar_bounds", 0.1 < n_bar < 10.0),
                ("sigma_p_bounds", 0.05 < sigma_p < 2.0),
                ("sigma_n_bounds", 0.05 < sigma_n < 2.0),
                ("variance_bound", variance_ref < VARIANCE_BOUND),
                ("shape_cap", max(max_p, max_n) < SHAPE_CAP),
            ]
        )
    elif source_folder in {"RandomDraw_PerfectSymmetric", "RandomDraw_Symmetric_Sep"}:
        cont = params["sym_cont"]
        rho = params["rho"]
        phi_plus = params["phi_plus"]
        phi_minus = params["phi_minus"]
        sigma = params["shared_sigma"]
        shapes = gjr_recursion(residuals, cont, rho, phi_plus, phi_minus, sigma)
        max_p = float(np.max(shapes))
        max_n = max_p
        variance_ref = 2.0 * sigma * sigma * cont
        checks.extend(
            [
                ("stability", rho + 0.5 * (phi_plus + phi_minus) < 1.0),
                ("variance_bound", variance_ref < VARIANCE_BOUND),
                ("shape_cap", max(max_p, max_n) < SHAPE_CAP),
            ]
        )
    elif source_folder == "RandomDraw_Symmetric_Oct":
        p0 = params["p0"]
        n0 = params["n0"]
        rho = params["rho"]
        phi_plus = params["phi_plus"]
        phi_minus = params["phi_minus"]
        sigma_p = params["sigma_p"]
        sigma_n = params["sigma_n"]
        p_series = gjr_recursion(residuals, p0, rho, phi_plus, phi_minus, sigma_p)
        n_series = gjr_recursion(residuals, n0, rho, phi_plus, phi_minus, sigma_n)
        max_p = float(np.max(p_series))
        max_n = float(np.max(n_series))
        variance_ref = sigma_p * sigma_p * p0 + sigma_n * sigma_n * n0
        checks.extend(
            [
                ("stability", rho + 0.5 * (phi_plus + phi_minus) < 1.0),
                ("variance_bound", variance_ref < VARIANCE_BOUND),
                ("shape_cap", max(max_p, max_n) < SHAPE_CAP),
            ]
        )
    elif source_folder == "RandomDraw_BG_GARCH":
        p0 = params["p0"]
        n0 = params["n0"]
        rho_p = params["rho_p"]
        rho_n = params["rho_n"]
        phi_p = params["phi_p"]
        phi_n = params["phi_n"]
        sigma_p = params["sigma_p"]
        sigma_n = params["sigma_n"]
        p_series = gjr_recursion(residuals, p0, rho_p, phi_p, phi_p, sigma_p)
        n_series = gjr_recursion(residuals, n0, rho_n, phi_n, phi_n, sigma_n)
        max_p = float(np.max(p_series))
        max_n = float(np.max(n_series))
        variance_ref = sigma_p * sigma_p * p0 + sigma_n * sigma_n * n0
        checks.extend(
            [
                ("p_stability", rho_p + phi_p < 1.0),
                ("n_stability", rho_n + phi_n < 1.0),
                ("variance_bound", variance_ref < VARIANCE_BOUND),
                ("shape_cap", max(max_p, max_n) < SHAPE_CAP),
            ]
        )
    elif source_folder == "RandomDraw_ID":
        p0 = params["p0"]
        n0 = params["n0"]
        rho_p = params["rho_p"]
        rho_n = params["rho_n"]
        phi_p_plus = params["phi_p_plus"]
        phi_n_minus = params["phi_n_minus"]
        sigma_p = params["sigma_p"]
        sigma_n = params["sigma_n"]
        p_series = gjr_recursion(residuals, p0, rho_p, phi_p_plus, 0.0, sigma_p)
        n_series = gjr_recursion(residuals, n0, rho_n, 0.0, phi_n_minus, sigma_n)
        max_p = float(np.max(p_series))
        max_n = float(np.max(n_series))
        variance_ref = sigma_p * sigma_p * p0 + sigma_n * sigma_n * n0
        checks.extend(
            [
                ("p_stability", rho_p + 0.5 * phi_p_plus < 1.0),
                ("n_stability", rho_n + 0.5 * phi_n_minus < 1.0),
                ("variance_bound", variance_ref < VARIANCE_BOUND),
                ("shape_cap", max(max_p, max_n) < SHAPE_CAP),
            ]
        )
    elif source_folder in {"RandomDraw_GJR", "RandomDraw_GJR_Oct", "RandomDraw_Full_Dec"}:
        p0 = params["p0"]
        n0 = params["n0"]
        rho_p = params["rho_p"]
        rho_n = params["rho_n"]
        phi_p_plus = params["phi_p_plus"]
        phi_p_minus = params["phi_p_minus"]
        phi_n_plus = params["phi_n_plus"]
        phi_n_minus = params["phi_n_minus"]
        sigma_p = params["sigma_p"]
        sigma_n = params["sigma_n"]
        p_series = gjr_recursion(residuals, p0, rho_p, phi_p_plus, phi_p_minus, sigma_p)
        n_series = gjr_recursion(residuals, n0, rho_n, phi_n_plus, phi_n_minus, sigma_n)
        max_p = float(np.max(p_series))
        max_n = float(np.max(n_series))
        variance_ref = sigma_p * sigma_p * p0 + sigma_n * sigma_n * n0
        checks.extend(
            [
                ("p_stability", rho_p + 0.5 * (phi_p_plus + phi_p_minus) < 1.0),
                ("n_stability", rho_n + 0.5 * (phi_n_plus + phi_n_minus) < 1.0),
                ("variance_bound", variance_ref < VARIANCE_BOUND),
                ("shape_cap", max(max_p, max_n) < SHAPE_CAP),
            ]
        )
    else:
        checks.append(("known_source_folder", False))

    checks.append(("finite_key_values", finite_values(max_p, max_n, variance_ref)))
    failed = [name for name, passed in checks if not passed]
    return (len(failed) == 0, "; ".join(failed), max_p, max_n, variance_ref)


def best_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    clean = df[np.isfinite(df[metric])]
    if clean.empty:
        return None
    return clean.loc[clean[metric].idxmin()]


def metric_bundle(row: pd.Series | None, prefix: str) -> dict[str, float]:
    if row is None:
        return {
            f"{prefix}_aic": np.nan,
            f"{prefix}_bic": np.nan,
            f"{prefix}_loglik": np.nan,
            f"{prefix}_draw_id": np.nan,
            f"{prefix}_rep_idx": np.nan,
        }
    return {
        f"{prefix}_aic": clean_float(row["AIC"]),
        f"{prefix}_bic": clean_float(row["BIC"]),
        f"{prefix}_loglik": clean_float(row["loglik"]),
        f"{prefix}_draw_id": clean_float(row["draw_id"]),
        f"{prefix}_rep_idx": clean_float(row["rep_idx"]),
    }


def row_identity(path: Path) -> dict[str, Any]:
    source_folder = path.parent.parent.name
    mean_folder = path.parent.name
    config = SOURCE_CONFIG.get(
        source_folder,
        {"family": source_folder, "scripts": "", "notes": "Unmapped local result source."},
    )
    return {
        "source_folder": source_folder,
        "source_summary": str(path.relative_to(ROOT)),
        "model_family": config["family"],
        "mean_model": MEAN_LABELS.get(mean_folder, mean_folder),
        "mean_folder": mean_folder,
        "estimation_scripts": config["scripts"],
        "notes": config["notes"],
    }


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame, SampleInfo]:
    data, sample = load_data()
    comparison_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    for path in sorted(ROOT.glob("RandomDraw_*/*/summary_draws.csv")):
        identity = row_identity(path)
        source_folder = identity["source_folder"]
        mean_folder = identity["mean_folder"]
        if mean_folder not in MEAN_PARAM_COUNT:
            continue

        df = pd.read_csv(path)
        for col in ["AIC", "BIC", "loglik", "draw_id", "rep_idx"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[np.isfinite(df["AIC"])].copy()
        if df.empty:
            continue

        validations = [
            evaluate_checks(data, source_folder, mean_folder, row)
            for _, row in df.iterrows()
        ]
        df["passes_documented_checks"] = [item[0] for item in validations]
        df["failed_checks"] = [item[1] for item in validations]
        df["max_p_shape"] = [item[2] for item in validations]
        df["max_n_shape"] = [item[3] for item in validations]
        df["variance_reference"] = [item[4] for item in validations]

        valid_df = df[df["passes_documented_checks"]].copy()
        all_aic = best_row(df, "AIC")
        all_bic = best_row(df, "BIC")
        valid_aic = best_row(valid_df, "AIC") if not valid_df.empty else None
        valid_bic = best_row(valid_df, "BIC") if not valid_df.empty else None

        comparison_row: dict[str, Any] = {
            **identity,
            "sample_start": sample.start,
            "sample_end": sample.end,
            "sample_nobs": sample.nobs,
            "sample_caveat": sample.caveat,
            "n_estimates": int(len(df)),
            "n_passing_checks": int(len(valid_df)),
        }
        comparison_row.update(metric_bundle(all_aic, "best_all_by_aic"))
        comparison_row.update(metric_bundle(all_bic, "best_all_by_bic"))
        comparison_row.update(metric_bundle(valid_aic, "best_valid_by_aic"))
        comparison_row.update(metric_bundle(valid_bic, "best_valid_by_bic"))
        comparison_rows.append(comparison_row)

        if valid_aic is not None:
            parameter_row = {
                **identity,
                "selection": "best_valid_by_aic",
                "sample_start": sample.start,
                "sample_end": sample.end,
                "sample_nobs": sample.nobs,
                "AIC": clean_float(valid_aic["AIC"]),
                "BIC": clean_float(valid_aic["BIC"]),
                "loglik": clean_float(valid_aic["loglik"]),
                "draw_id": clean_float(valid_aic["draw_id"]),
                "rep_idx": clean_float(valid_aic["rep_idx"]),
                "max_p_shape": clean_float(valid_aic["max_p_shape"]),
                "max_n_shape": clean_float(valid_aic["max_n_shape"]),
                "variance_reference": clean_float(valid_aic["variance_reference"]),
            }
            parameter_row.update(standardized_params(source_folder, mean_folder, valid_aic))
            parameter_rows.append(parameter_row)

    comparison = pd.DataFrame(comparison_rows)
    parameters = pd.DataFrame(parameter_rows)

    if not comparison.empty:
        comparison = comparison.sort_values(
            ["best_valid_by_aic_aic", "model_family", "mean_model"],
            na_position="last",
        )
    if not parameters.empty:
        parameters = parameters.sort_values(["AIC", "model_family", "mean_model"])
    return comparison, parameters, sample


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return ""
    return f"{value_float:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join([header, separator, *body])


def write_markdown_report(comparison: pd.DataFrame, sample: SampleInfo) -> str:
    top = comparison.copy()
    top = top[np.isfinite(top["best_valid_by_aic_aic"])].head(30)

    rows = []
    for _, row in top.iterrows():
        rows.append(
            {
                "family": row["model_family"],
                "mean": row["mean_model"],
                "source": row["source_folder"],
                "valid": f"{int(row['n_passing_checks'])}/{int(row['n_estimates'])}",
                "loglik": fmt(row["best_valid_by_aic_loglik"]),
                "aic": fmt(row["best_valid_by_aic_aic"]),
                "bic": fmt(row["best_valid_by_aic_bic"]),
                "draw": fmt(row["best_valid_by_aic_draw_id"], digits=0),
                "rep": fmt(row["best_valid_by_aic_rep_idx"], digits=0),
            }
        )

    all_invalid = comparison[
        comparison["best_all_by_aic_aic"] != comparison["best_valid_by_aic_aic"]
    ].copy()
    all_invalid = all_invalid[np.isfinite(all_invalid["best_all_by_aic_aic"])].head(12)
    invalid_rows = []
    for _, row in all_invalid.iterrows():
        invalid_rows.append(
            {
                "family": row["model_family"],
                "mean": row["mean_model"],
                "source": row["source_folder"],
                "all_aic": fmt(row["best_all_by_aic_aic"]),
                "valid_aic": fmt(row["best_valid_by_aic_aic"]),
            }
        )

    parts = [
        REPORT_PREAMBLE,
        "# BEGE Result Summary\n\n",
        "This file is generated by `BEGE_GARCH/collect_bege_results.py` from the local "
        "`RandomDraw_*/*/summary_draws.csv` files. The raw random-draw folders remain "
        "ignored by Git; the tables here are the compact tracked result artifacts.\n\n",
        "## Sample Note\n\n",
        f"The loaded BEGE summaries use `{DATA_PATH.name}` with {sample.nobs} observations "
        f"from {sample.start} to {sample.end}. This {sample.caveat}.\n\n",
        "## Best Estimates Passing Documented Checks\n\n",
        markdown_table(
            rows,
            [
                ("Family", "family"),
                ("Mean", "mean"),
                ("Source", "source"),
                ("Passing/Total", "valid"),
                ("LogLik", "loglik"),
                ("AIC", "aic"),
                ("BIC", "bic"),
                ("Draw", "draw"),
                ("Rep", "rep"),
            ],
        ),
        "\n\n",
        "## Diagnostic Note\n\n",
        "Some local searches found lower raw AIC values that fail at least one documented "
        "BEGE check. Those rows are retained in `bege_model_comparison.csv` for audit, "
        "but the table above selects only rows passing the persistence, variance, and "
        "shape-cap checks applicable to each model family.\n\n",
    ]

    if invalid_rows:
        parts.extend(
            [
                markdown_table(
                    invalid_rows,
                    [
                        ("Family", "family"),
                        ("Mean", "mean"),
                        ("Source", "source"),
                        ("Raw Best AIC", "all_aic"),
                        ("Checked Best AIC", "valid_aic"),
                    ],
                ),
                "\n",
            ]
        )
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where compact BEGE result summaries are written.",
    )
    args = parser.parse_args()

    comparison, parameters, sample = collect_results()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = args.results_dir / "bege_model_comparison.csv"
    parameters_path = args.results_dir / "bege_best_parameters.csv"
    report_path = args.results_dir / "bege_best_models.md"

    comparison.to_csv(comparison_path, index=False)
    parameters.to_csv(parameters_path, index=False)
    report_path.write_text(write_markdown_report(comparison, sample), encoding="utf-8")

    print(f"Wrote {comparison_path.relative_to(ROOT)}")
    print(f"Wrote {parameters_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
