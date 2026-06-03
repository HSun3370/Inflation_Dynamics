from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd


MEAN_TYPES = ["constant", "ARX(1,1)", "ARX(2,1)", "ARX(2,2)"]
MEAN_PARAM_NAMES = {
    "constant": [],
    "ARX(1,1)": ["c", "rho_1", "phi_1"],
    "ARX(2,1)": ["c", "rho_1", "rho_2", "phi_1"],
    "ARX(2,2)": ["c", "rho_1", "rho_2", "phi_1", "phi_2"],
}
REPORT_DROP_COLUMNS = {"message"}
DRAW_FILE_RE = re.compile(r"draw_(\d+)\.csv$")
DEFAULT_HIGH_SHAPE_REFERENCE = 200.0
IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD = -150.0
BEST_MODELS_PER_REPORT = 1
TOP_MODELS_PER_MEAN = BEST_MODELS_PER_REPORT
MEAN_FILE_STEMS = {
    "constant": "constant",
    "ARX(1,1)": "ARX_1_1",
    "ARX(2,1)": "ARX_2_1",
    "ARX(2,2)": "ARX_2_2",
}


def model_param_names_for_family(model_family: str) -> list[str]:
    if model_family == "badgood":
        return ["p0", "n0", "rho_p", "rho_n", "phi_p", "phi_n", "sigma_p", "sigma_n"]
    if model_family == "id":
        return ["p0", "n0", "rho_p", "rho_n", "phi_p_plus", "phi_n_minus", "sigma_p", "sigma_n"]
    if model_family == "full":
        return [
            "p0",
            "n0",
            "rho_p",
            "rho_n",
            "phi_p_plus",
            "phi_p_minus",
            "phi_n_plus",
            "phi_n_minus",
            "sigma_p",
            "sigma_n",
        ]
    if model_family == "symmetric":
        return ["p0", "n0", "rho", "phi_plus", "phi_minus", "sigma_p", "sigma_n"]
    raise ValueError(f"Unknown model_family {model_family!r}.")


def variance_bound_for_family(model_family: str) -> float:
    if model_family == "full":
        return 0.75
    if model_family in {"badgood", "id", "symmetric"}:
        return 0.87
    raise ValueError(f"Unknown model_family {model_family!r}.")


def _resolve_column(df: pd.DataFrame, preferred: str, aliases: tuple[str, ...]) -> str:
    candidates = (preferred, *aliases)
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing required column. Tried: {candidates}")


def load_effective_sample(project_root: Path) -> pd.DataFrame:
    data_path = project_root / "DataSummary" / "Aggregate_CPI_inflation_Quarterly.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing effective-sample file: {data_path}")

    df = pd.read_pickle(data_path)
    inflation_col = _resolve_column(df, "Inflation", ("inflation",))
    spf_col = _resolve_column(df, "SPF", ("Forecasted inflation", "forecast", "SPF_t"))

    if "SPF_shock" in df.columns:
        shock = df["SPF_shock"].to_numpy(dtype=float)
    elif "Inflation shock" in df.columns:
        shock = df["Inflation shock"].to_numpy(dtype=float)
    else:
        shock = (df[inflation_col] - df[spf_col]).to_numpy(dtype=float)

    canonical = pd.DataFrame(
        {
            "Inflation": df[inflation_col].to_numpy(dtype=float),
            "SPF": df[spf_col].to_numpy(dtype=float),
            "SPF_shock": shock,
        },
        index=df.index,
    )

    lag_aliases = {
        "Inflation_lag_1": ("Inflation_lag_1", "Inflation.Lag(1)", "Inflation lag 1"),
        "Inflation_lag_2": ("Inflation_lag_2", "Inflation.Lag(2)", "Inflation lag 2"),
        "SPF_lag_1": ("SPF_lag_1", "SPF.lag(1)", "Forecasted inflation lag 1"),
    }
    for canon, aliases in lag_aliases.items():
        for col in aliases:
            if col in df.columns:
                canonical[canon] = df[col].to_numpy(dtype=float)
                break

    return canonical


def build_model_specs(df: pd.DataFrame, include_arx22: bool) -> list[dict]:
    if "Inflation_lag_1" in df.columns:
        x_arx11 = {
            "SPF": df["SPF"].to_numpy(dtype=float),
            "Inflation_lag_1": df["Inflation_lag_1"].to_numpy(dtype=float),
        }
    else:
        x_arx11 = df["SPF"].to_numpy(dtype=float)

    if {"Inflation_lag_1", "Inflation_lag_2"}.issubset(df.columns):
        x_arx21 = {
            "SPF": df["SPF"].to_numpy(dtype=float),
            "Inflation_lag_1": df["Inflation_lag_1"].to_numpy(dtype=float),
            "Inflation_lag_2": df["Inflation_lag_2"].to_numpy(dtype=float),
        }
    else:
        x_arx21 = df["SPF"].to_numpy(dtype=float)

    specs = [
        {"mean_type": "constant", "Y": df["SPF_shock"].to_numpy(dtype=float), "X": None},
        {"mean_type": "ARX(1,1)", "Y": df["Inflation"].to_numpy(dtype=float), "X": x_arx11},
        {"mean_type": "ARX(2,1)", "Y": df["Inflation"].to_numpy(dtype=float), "X": x_arx21},
    ]

    if include_arx22:
        if {"Inflation_lag_1", "Inflation_lag_2", "SPF_lag_1"}.issubset(df.columns):
            x_arx22 = {
                "SPF": df["SPF"].to_numpy(dtype=float),
                "Inflation_lag_1": df["Inflation_lag_1"].to_numpy(dtype=float),
                "Inflation_lag_2": df["Inflation_lag_2"].to_numpy(dtype=float),
                "SPF_lag_1": df["SPF_lag_1"].to_numpy(dtype=float),
            }
        else:
            x_arx22 = df["SPF"].to_numpy(dtype=float)
        specs.append({"mean_type": "ARX(2,2)", "Y": df["Inflation"].to_numpy(dtype=float), "X": x_arx22})

    return specs


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} requires columns {missing} in the effective sample file.")


def result_to_row(
    result: dict,
    mean_type: str,
    draw: int,
    seed: int,
    random_state: int,
    model_param_names: list[str],
) -> dict:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "draw": draw,
        "random_state": random_state,
        "mean_type": mean_type,
        "loglik": float(result["loglik"]),
        "AIC": float(result["AIC"]),
        "BIC": float(result["BIC"]),
    }

    opt = result.get("opt")
    row["success"] = bool(getattr(opt, "success", False))
    row["status"] = int(getattr(opt, "status", -1))
    row["message"] = str(getattr(opt, "message", ""))

    params = np.asarray(result.get("params", []), dtype=float)
    ses = np.asarray(result.get("se", []), dtype=float)
    names = MEAN_PARAM_NAMES[mean_type] + model_param_names

    for name, value in zip(names, params):
        row[f"param_{name}"] = float(value)
    for name, value in zip(names, ses):
        row[f"se_{name}"] = float(value)

    return row


def save_seed_csv(rows: list[dict], path: Path) -> None:
    out = pd.DataFrame(rows).sort_values(["mean_type", "draw"])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def run_seed_estimation(
    *,
    estimator: Callable,
    model_label: str,
    script_dir: Path,
    project_root: Path,
    model_param_names: list[str],
    seed: int,
    n_draws: int,
    n_starts: int,
    maxiter: int,
    tol: float,
    include_arx22: bool,
    print_summary: bool,
    density_hyperu_method: str,
    cap_pn,
    compute_se: bool,
) -> None:
    output_dir = script_dir / "output"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = load_effective_sample(project_root)
    if len(df) != 215:
        print(f"[WARN] Effective sample length is {len(df)} (expected 215).")

    require_columns(df, ["Inflation", "SPF", "SPF_shock"], model_label)
    specs = build_model_specs(df, include_arx22=include_arx22)

    total_starts = len(specs) * n_draws * n_starts
    print(
        f"Seed {seed}: estimating {model_label} for {len(specs)} mean process(es), "
        f"{n_draws} draws each, {n_starts} starts per draw "
        f"({total_starts:,} optimizer starts)."
    )

    all_rows: list[dict] = []
    out_file = raw_dir / f"draw_{seed:03d}.csv"
    for spec in specs:
        mean_type = spec["mean_type"]
        print(f"Estimating mean_type={mean_type} with {n_draws} draws...")

        for draw in range(1, n_draws + 1):
            rs = draw + seed * 10000
            try:
                result = estimator(
                    Y=spec["Y"],
                    X=spec["X"],
                    mean_type=mean_type,
                    n_starts=n_starts,
                    maxiter=maxiter,
                    tol=tol,
                    random_state=rs,
                    cap_pn=cap_pn,
                    compute_se=compute_se,
                    density_hyperu_method=density_hyperu_method,
                    print_summary=print_summary,
                )
                row = result_to_row(
                    result,
                    mean_type=mean_type,
                    draw=draw,
                    seed=seed,
                    random_state=rs,
                    model_param_names=model_param_names,
                )
            except Exception as exc:
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "seed": seed,
                    "draw": draw,
                    "random_state": rs,
                    "mean_type": mean_type,
                    "success": False,
                    "status": -999,
                    "message": f"{type(exc).__name__}: {exc}",
                    "loglik": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                }

            all_rows.append(row)
            save_seed_csv(all_rows, out_file)

        print(f"Finished mean_type={mean_type}.")

    save_seed_csv(all_rows, out_file)
    print(f"Saved completed seed results to: {out_file}")


def format_value(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return f"{val:.6f}"


def format_int(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return str(int(val))


def success_mask(df: pd.DataFrame) -> pd.Series:
    if "success" not in df.columns:
        return pd.Series(False, index=df.index)

    values = df["success"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def optimizer_success_mask(df: pd.DataFrame) -> pd.Series:
    if "optimizer_success" not in df.columns:
        return pd.Series(True, index=df.index)

    values = df["optimizer_success"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def strict_success_mask(df: pd.DataFrame) -> pd.Series:
    return success_mask(df) & optimizer_success_mask(df)


def selection_eligible_mask(df: pd.DataFrame) -> pd.Series:
    if "selection_eligible" not in df.columns:
        return pd.Series(False, index=df.index)

    values = df["selection_eligible"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def _finite_metric_mask(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in ("loglik", "AIC", "BIC") if col in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= np.isfinite(pd.to_numeric(df[col], errors="coerce"))
    return mask


def _row_float(row: pd.Series, name: str) -> float:
    value = row.get(name, np.nan)
    return float(value)


def _mean_residuals_from_row(row: pd.Series, specs_by_mean: dict[str, dict]) -> np.ndarray:
    from BEGE_GARCH.BEGE_GARCH import _make_residual_function

    mean_type = row.get("mean_type")
    if mean_type not in specs_by_mean:
        raise KeyError(f"Unknown mean_type {mean_type!r}.")

    names = MEAN_PARAM_NAMES.get(mean_type, [])
    params = np.asarray([_row_float(row, f"param_{name}") for name in names], dtype=float)
    if not np.all(np.isfinite(params)):
        raise ValueError(f"Missing finite mean parameters for {mean_type}.")

    spec = specs_by_mean[mean_type]
    return _make_residual_function(spec["Y"], spec["X"], mean_type)(params)


def _selection_metrics_for_row(
    row: pd.Series,
    *,
    model_family: str,
    specs_by_mean: dict[str, dict],
) -> dict[str, float]:
    from BEGE_GARCH.BEGE_GARCH import gjr_recursion
    from BEGE_GARCH.BEGE_GARCH import bege_variance_bounds_ok
    from BEGE_GARCH.BEGE_Density.BEGE_density import BEGE_log_density

    residuals = _mean_residuals_from_row(row, specs_by_mean)

    if model_family == "badgood":
        p0 = _row_float(row, "param_p0")
        n0 = _row_float(row, "param_n0")
        rho_p = _row_float(row, "param_rho_p")
        rho_n = _row_float(row, "param_rho_n")
        phi_p = _row_float(row, "param_phi_p")
        phi_n = _row_float(row, "param_phi_n")
        sigma_p = _row_float(row, "param_sigma_p")
        sigma_n = _row_float(row, "param_sigma_n")
        pseries = gjr_recursion(residuals, (p0, rho_p, phi_p, phi_p), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho_n, phi_n, phi_n), sigma_n)
        persistence_p = rho_p + phi_p
        persistence_n = rho_n + phi_n

    elif model_family == "id":
        p0 = _row_float(row, "param_p0")
        n0 = _row_float(row, "param_n0")
        rho_p = _row_float(row, "param_rho_p")
        rho_n = _row_float(row, "param_rho_n")
        phi_p_plus = _row_float(row, "param_phi_p_plus")
        phi_n_minus = _row_float(row, "param_phi_n_minus")
        sigma_p = _row_float(row, "param_sigma_p")
        sigma_n = _row_float(row, "param_sigma_n")
        pseries = gjr_recursion(residuals, (p0, rho_p, phi_p_plus, 0.0), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho_n, 0.0, phi_n_minus), sigma_n)
        persistence_p = rho_p + 0.5 * phi_p_plus
        persistence_n = rho_n + 0.5 * phi_n_minus

    elif model_family == "full":
        p0 = _row_float(row, "param_p0")
        n0 = _row_float(row, "param_n0")
        rho_p = _row_float(row, "param_rho_p")
        rho_n = _row_float(row, "param_rho_n")
        phi_p_plus = _row_float(row, "param_phi_p_plus")
        phi_p_minus = _row_float(row, "param_phi_p_minus")
        phi_n_plus = _row_float(row, "param_phi_n_plus")
        phi_n_minus = _row_float(row, "param_phi_n_minus")
        sigma_p = _row_float(row, "param_sigma_p")
        sigma_n = _row_float(row, "param_sigma_n")
        pseries = gjr_recursion(residuals, (p0, rho_p, phi_p_plus, phi_p_minus), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho_n, phi_n_plus, phi_n_minus), sigma_n)
        persistence_p = rho_p + 0.5 * (phi_p_plus + phi_p_minus)
        persistence_n = rho_n + 0.5 * (phi_n_plus + phi_n_minus)

    elif model_family == "symmetric":
        p0 = _row_float(row, "param_p0")
        n0 = _row_float(row, "param_n0")
        rho = _row_float(row, "param_rho")
        phi_plus = _row_float(row, "param_phi_plus")
        phi_minus = _row_float(row, "param_phi_minus")
        sigma_p = _row_float(row, "param_sigma_p")
        sigma_n = _row_float(row, "param_sigma_n")
        pseries = gjr_recursion(residuals, (p0, rho, phi_plus, phi_minus), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho, phi_plus, phi_minus), sigma_n)
        persistence_p = rho + 0.5 * (phi_plus + phi_minus)
        persistence_n = persistence_p

    else:
        raise ValueError(f"Unknown model_family {model_family!r}.")

    variance_details = bege_variance_bounds_ok(
        residuals,
        pseries,
        nseries,
        sigma_p,
        sigma_n,
        return_details=True,
    )
    cond_var = variance_details["cond_var"]
    lower = variance_details["lower"]
    upper = variance_details["upper"]
    ll_vec = BEGE_log_density(residuals, pseries, nseries, sigma_p, sigma_n)
    if not np.all(np.isfinite(ll_vec)):
        corrected_loglik = np.nan
    else:
        corrected_loglik = float(np.sum(ll_vec))

    mean_type = row.get("mean_type")
    k_params = len(MEAN_PARAM_NAMES.get(mean_type, [])) + len(model_param_names_for_family(model_family))
    n_obs = int(residuals.shape[0])
    corrected_aic = 2.0 * k_params - 2.0 * corrected_loglik
    corrected_bic = np.log(n_obs) * k_params - 2.0 * corrected_loglik

    return {
        "corrected_loglik": float(corrected_loglik),
        "corrected_AIC": float(corrected_aic),
        "corrected_BIC": float(corrected_bic),
        "selection_persistence_p": float(persistence_p),
        "selection_persistence_n": float(persistence_n),
        "selection_sigma_min": float(min(sigma_p, sigma_n)),
        "selection_uncond_var_ref": float(sigma_p * sigma_p * p0 + sigma_n * sigma_n * n0),
        "selection_max_p_t": float(np.max(pseries)),
        "selection_max_n_t": float(np.max(nseries)),
        "selection_shape_max": float(max(np.max(pseries), np.max(nseries))),
        "selection_cond_var_min": float(np.min(cond_var)),
        "selection_cond_var_median": float(np.median(cond_var)),
        "selection_cond_var_max": float(np.max(cond_var)),
        "selection_implied_variance_bounds_ok": bool(variance_details["ok"]),
        "selection_cond_var_lower_min": float(np.min(lower)),
        "selection_cond_var_lower_max": float(np.max(lower)),
        "selection_cond_var_upper_min": float(np.min(upper)),
        "selection_cond_var_upper_max": float(np.max(upper)),
    }


def _parameter_names_for_row(row: pd.Series, model_family: str) -> list[str]:
    mean_type = row.get("mean_type")
    return MEAN_PARAM_NAMES[mean_type] + model_param_names_for_family(model_family)


def _parameter_vector_from_row(row: pd.Series, model_family: str) -> tuple[list[str], np.ndarray]:
    names = _parameter_names_for_row(row, model_family)
    params = np.asarray([_row_float(row, f"param_{name}") for name in names], dtype=float)
    if not np.all(np.isfinite(params)):
        raise ValueError(f"Missing finite parameter values for {row.get('mean_type')}.")
    return names, params


def _bounds_for_row(spec: dict, mean_type: str, model_family: str) -> list[tuple[float | None, float | None]]:
    y = np.asarray(spec["Y"], dtype=float)
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    if mean_type == "constant":
        bounds_mean: list[tuple[float | None, float | None]] = []
    elif mean_type == "ARX(1,1)":
        bounds_mean = [(ymin, ymax), (-0.999, 0.999), (-10.0, 10.0)]
    elif mean_type == "ARX(2,1)":
        bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10.0, 10.0)]
    elif mean_type == "ARX(2,2)":
        bounds_mean = [
            (ymin, ymax),
            (-1.999, 1.999),
            (-0.999, 0.999),
            (-10.0, 10.0),
            (-10.0, 10.0),
        ]
    else:
        raise ValueError(f"Unknown mean_type {mean_type!r}.")

    p0_bounds = (0.0, 10.0)
    rho_bounds = (0.0, 1.0)
    phi_bounds = (0.0, 2.0)
    sigma_bounds = (1e-5, 2.0)

    if model_family in {"badgood", "id"}:
        bounds_vol = [
            p0_bounds,
            p0_bounds,
            rho_bounds,
            rho_bounds,
            phi_bounds,
            phi_bounds,
            sigma_bounds,
            sigma_bounds,
        ]
    elif model_family == "full":
        bounds_vol = [
            p0_bounds,
            p0_bounds,
            rho_bounds,
            rho_bounds,
            phi_bounds,
            phi_bounds,
            phi_bounds,
            phi_bounds,
            sigma_bounds,
            sigma_bounds,
        ]
    elif model_family == "symmetric":
        bounds_vol = [p0_bounds, p0_bounds, rho_bounds, phi_bounds, phi_bounds, sigma_bounds, sigma_bounds]
    else:
        raise ValueError(f"Unknown model_family {model_family!r}.")

    return bounds_mean + bounds_vol


def _project_to_bounds(theta: np.ndarray, bounds: list[tuple[float | None, float | None]]) -> np.ndarray:
    out = np.asarray(theta, dtype=float).copy()
    tiny = 1e-10
    for idx, (lo, hi) in enumerate(bounds):
        if lo is not None:
            out[idx] = max(out[idx], lo + tiny)
        if hi is not None:
            out[idx] = min(out[idx], hi - tiny)
    return out


def _sym_matrix(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _safe_inv_with_ridge(mat: np.ndarray, ridge0: float = 1e-8, max_tries: int = 6):
    mat = _sym_matrix(np.asarray(mat, dtype=float))
    eye = np.eye(mat.shape[0])
    ridge = float(ridge0)
    for _ in range(max_tries):
        try:
            return np.linalg.inv(mat + ridge * eye), ridge, False
        except np.linalg.LinAlgError:
            ridge *= 10.0
    return np.linalg.pinv(mat), ridge, True


def _vol_paths_from_theta(
    theta: np.ndarray,
    *,
    mean_type: str,
    model_family: str,
    residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, float]]:
    from BEGE_GARCH.BEGE_GARCH import gjr_recursion

    num_m = len(MEAN_PARAM_NAMES[mean_type])
    vol = theta[num_m:]

    if model_family == "badgood":
        p0, n0, rho_p, rho_n, phi_p, phi_n, sigma_p, sigma_n = vol
        pseries = gjr_recursion(residuals, (p0, rho_p, phi_p, phi_p), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho_n, phi_n, phi_n), sigma_n)
        persistence_p = rho_p + phi_p
        persistence_n = rho_n + phi_n

    elif model_family == "id":
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigma_p, sigma_n = vol
        pseries = gjr_recursion(residuals, (p0, rho_p, phi_p_plus, 0.0), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho_n, 0.0, phi_n_minus), sigma_n)
        persistence_p = rho_p + 0.5 * phi_p_plus
        persistence_n = rho_n + 0.5 * phi_n_minus

    elif model_family == "full":
        (
            p0,
            n0,
            rho_p,
            rho_n,
            phi_p_plus,
            phi_p_minus,
            phi_n_plus,
            phi_n_minus,
            sigma_p,
            sigma_n,
        ) = vol
        pseries = gjr_recursion(residuals, (p0, rho_p, phi_p_plus, phi_p_minus), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho_n, phi_n_plus, phi_n_minus), sigma_n)
        persistence_p = rho_p + 0.5 * (phi_p_plus + phi_p_minus)
        persistence_n = rho_n + 0.5 * (phi_n_plus + phi_n_minus)

    elif model_family == "symmetric":
        p0, n0, rho, phi_plus, phi_minus, sigma_p, sigma_n = vol
        pseries = gjr_recursion(residuals, (p0, rho, phi_plus, phi_minus), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho, phi_plus, phi_minus), sigma_n)
        persistence_p = rho + 0.5 * (phi_plus + phi_minus)
        persistence_n = persistence_p

    else:
        raise ValueError(f"Unknown model_family {model_family!r}.")

    diagnostics = {
        "p0": float(p0),
        "n0": float(n0),
        "persistence_p": float(persistence_p),
        "persistence_n": float(persistence_n),
    }
    return pseries, nseries, float(sigma_p), float(sigma_n), diagnostics


def _constraints_ok_for_theta(theta: np.ndarray, *, mean_type: str, model_family: str) -> bool:
    num_m = len(MEAN_PARAM_NAMES[mean_type])
    vol = np.asarray(theta[num_m:], dtype=float)
    if not np.all(np.isfinite(vol)):
        return False

    variance_bound = variance_bound_for_family(model_family)

    if model_family == "badgood":
        p0, n0, rho_p, rho_n, phi_p, phi_n, sigma_p, sigma_n = vol
        stable = (rho_p + phi_p < 1.0 - 1e-6) and (rho_n + phi_n < 1.0 - 1e-6)
    elif model_family == "id":
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigma_p, sigma_n = vol
        stable = (rho_p + 0.5 * phi_p_plus < 1.0 - 1e-6) and (
            rho_n + 0.5 * phi_n_minus < 1.0 - 1e-6
        )
    elif model_family == "full":
        (
            p0,
            n0,
            rho_p,
            rho_n,
            phi_p_plus,
            phi_p_minus,
            phi_n_plus,
            phi_n_minus,
            sigma_p,
            sigma_n,
        ) = vol
        stable = (rho_p + 0.5 * (phi_p_plus + phi_p_minus) < 1.0 - 1e-6) and (
            rho_n + 0.5 * (phi_n_plus + phi_n_minus) < 1.0 - 1e-6
        )
    elif model_family == "symmetric":
        p0, n0, rho, phi_plus, phi_minus, sigma_p, sigma_n = vol
        stable = rho + 0.5 * (phi_plus + phi_minus) < 1.0 - 1e-6
    else:
        raise ValueError(f"Unknown model_family {model_family!r}.")

    if not stable:
        return False
    if sigma_p * sigma_p * p0 + sigma_n * sigma_n * n0 > variance_bound:
        return False
    return True


def _mean_stationarity_ok(theta: np.ndarray, mean_type: str) -> bool:
    theta = np.asarray(theta, dtype=float)
    if mean_type == "constant":
        return True
    if mean_type == "ARX(1,1)":
        if theta.size < 2 or not np.isfinite(theta[1]):
            return False
        return bool(abs(theta[1]) < 1.0)
    if mean_type in {"ARX(2,1)", "ARX(2,2)"}:
        if theta.size < 3 or not np.all(np.isfinite(theta[1:3])):
            return False
        rho_1, rho_2 = theta[1], theta[2]
        companion = np.array([[rho_1, rho_2], [1.0, 0.0]], dtype=float)
        roots = np.linalg.eigvals(companion)
        return bool(np.all(np.abs(roots) < 1.0))
    raise ValueError(f"Unknown mean_type {mean_type!r}.")


def _row_likelihood_functions(
    *,
    spec: dict,
    mean_type: str,
    model_family: str,
    big_penalty: float = 1e12,
    big_vec_penalty: float = 1e6,
):
    from BEGE_GARCH.BEGE_GARCH import _make_residual_function, bege_variance_bounds_ok
    from BEGE_GARCH.BEGE_Density.BEGE_density import BEGE_log_density

    residual_function = _make_residual_function(spec["Y"], spec["X"], mean_type)
    n_obs = int(np.asarray(spec["Y"], dtype=float).shape[0])
    num_m = len(MEAN_PARAM_NAMES[mean_type])

    def _ind_negloglik(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if not _constraints_ok_for_theta(theta, mean_type=mean_type, model_family=model_family):
            return np.full(n_obs, float(big_vec_penalty), dtype=float)

        residuals = residual_function(theta[:num_m])
        pseries, nseries, sigma_p, sigma_n, _ = _vol_paths_from_theta(
            theta,
            mean_type=mean_type,
            model_family=model_family,
            residuals=residuals,
        )
        if (
            not np.all(np.isfinite(pseries))
            or not np.all(np.isfinite(nseries))
            or np.any(pseries <= 0.0)
            or np.any(nseries <= 0.0)
        ):
            return np.full(n_obs, float(big_vec_penalty), dtype=float)
        if not bege_variance_bounds_ok(residuals, pseries, nseries, sigma_p, sigma_n):
            return np.full(n_obs, float(big_vec_penalty), dtype=float)

        values = -BEGE_log_density(residuals, pseries, nseries, sigma_p, sigma_n)
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != n_obs:
            values = np.full(n_obs, float(values.ravel()[0]))
        if not np.all(np.isfinite(values)):
            values = np.full(n_obs, float(big_vec_penalty), dtype=float)
        return values

    def _negloglik(theta: np.ndarray) -> float:
        values = _ind_negloglik(theta)
        val = float(np.sum(values))
        if not np.isfinite(val) or val >= big_vec_penalty * n_obs:
            return float(big_penalty)
        return val

    return _negloglik, _ind_negloglik


def _central_diff_scores(
    theta: np.ndarray,
    f_per_obs: Callable[[np.ndarray], np.ndarray],
    bounds: list[tuple[float | None, float | None]],
    n_obs: int,
    rel: float = 1e-4,
    absmin: float = 1e-6,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    k_params = theta.size
    f0 = np.asarray(f_per_obs(theta), dtype=float).reshape(-1)
    scores = np.empty((n_obs, k_params), dtype=float)
    steps = np.maximum(absmin, rel * np.maximum(1.0, np.abs(theta)))

    for idx in range(k_params):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[idx] += steps[idx]
        theta_minus[idx] -= steps[idx]
        theta_plus = _project_to_bounds(theta_plus, bounds)
        theta_minus = _project_to_bounds(theta_minus, bounds)

        f_plus = np.asarray(f_per_obs(theta_plus), dtype=float).reshape(-1)
        f_minus = np.asarray(f_per_obs(theta_minus), dtype=float).reshape(-1)
        if f_plus.shape[0] != n_obs:
            f_plus = np.full(n_obs, float(f_plus.ravel()[0]))
        if f_minus.shape[0] != n_obs:
            f_minus = np.full(n_obs, float(f_minus.ravel()[0]))

        denom = float(theta_plus[idx] - theta_minus[idx])
        if denom == 0.0:
            theta_plus = theta.copy()
            theta_plus[idx] += steps[idx]
            theta_plus = _project_to_bounds(theta_plus, bounds)
            f_plus = np.asarray(f_per_obs(theta_plus), dtype=float).reshape(-1)
            if f_plus.shape[0] != n_obs:
                f_plus = np.full(n_obs, float(f_plus.ravel()[0]))
            f_minus = f0
            denom = float(theta_plus[idx] - theta[idx])
            if denom == 0.0:
                scores[:, idx] = 0.0
                continue

        scores[:, idx] = (f_plus - f_minus) / denom

    return scores


def compute_standard_errors_for_row(
    row: pd.Series,
    *,
    spec: dict,
    model_family: str,
) -> dict:
    from statsmodels.tools.numdiff import approx_hess

    mean_type = row["mean_type"]
    names, theta = _parameter_vector_from_row(row, model_family)
    bounds = _bounds_for_row(spec, mean_type, model_family)
    theta_eval = _project_to_bounds(theta, bounds)
    n_obs = int(np.asarray(spec["Y"], dtype=float).shape[0])
    negloglik, ind_negloglik = _row_likelihood_functions(
        spec=spec,
        mean_type=mean_type,
        model_family=model_family,
    )

    obj_value = negloglik(theta_eval)
    if not np.isfinite(obj_value) or obj_value >= 1e12:
        raise ValueError("Corrected likelihood is not finite at the supplied estimate.")

    hessian = approx_hess(theta_eval, negloglik, epsilon=1e-5)
    hessian = _sym_matrix(hessian)
    hessian_inv, used_ridge, used_pseudo = _safe_inv_with_ridge(hessian)

    scores = _central_diff_scores(theta_eval, ind_negloglik, bounds, n_obs)
    opg = scores.T @ scores
    opg_scale = np.linalg.norm(opg) / max(1, opg.size)
    used_opg_fallback = (not np.isfinite(opg_scale)) or opg_scale < 1e-8
    if used_opg_fallback:
        covariance = hessian_inv.copy()
    else:
        covariance = hessian_inv @ _sym_matrix(opg) @ hessian_inv

    covariance = _sym_matrix(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    standard_errors = np.sqrt(np.diag(covariance))

    result = {
        "se_message": "computed",
        "se_hessian_ridge": float(used_ridge),
        "se_used_pseudoinverse": bool(used_pseudo),
        "se_used_opg_fallback": bool(used_opg_fallback),
    }
    for name, se in zip(names, standard_errors):
        result[f"se_{name}"] = float(se)
    return result


def add_standard_errors_for_rows(
    rows: list[pd.Series],
    *,
    project_root: Path,
    model_family: str,
) -> list[dict]:
    specs_by_mean = {
        spec["mean_type"]: spec
        for spec in build_model_specs(load_effective_sample(project_root), include_arx22=True)
    }

    enriched_rows: list[dict] = []
    for row in rows:
        enriched = row.to_dict()
        try:
            se_values = compute_standard_errors_for_row(
                row,
                spec=specs_by_mean[row["mean_type"]],
                model_family=model_family,
            )
            enriched.update(se_values)
        except Exception as exc:
            enriched["se_message"] = f"{type(exc).__name__}: {exc}"
        enriched_rows.append(enriched)
    return enriched_rows


def add_selection_diagnostics(
    df: pd.DataFrame,
    *,
    project_root: Path,
    model_family: str,
    shape_reference: float = DEFAULT_HIGH_SHAPE_REFERENCE,
) -> pd.DataFrame:
    out = df.copy()
    diagnostics = []
    specs_by_mean = {
        spec["mean_type"]: spec
        for spec in build_model_specs(load_effective_sample(project_root), include_arx22=True)
    }

    converged = strict_success_mask(out)

    for idx, row in out.iterrows():
        diag = {
            "stored_loglik": _row_float(row, "stored_loglik") if "stored_loglik" in row else _row_float(row, "loglik"),
            "stored_AIC": _row_float(row, "stored_AIC") if "stored_AIC" in row else _row_float(row, "AIC"),
            "stored_BIC": _row_float(row, "stored_BIC") if "stored_BIC" in row else _row_float(row, "BIC"),
            "corrected_loglik": np.nan,
            "corrected_AIC": np.nan,
            "corrected_BIC": np.nan,
            "corrected_loglik_delta": np.nan,
            "selection_shape_reference": float(shape_reference),
            "selection_high_shape_density": False,
            "selection_loglik_upper_threshold": float(IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD),
            "selection_loglik_plausible": False,
            "selection_variance_bound": variance_bound_for_family(model_family),
            "selection_bounds_ok": False,
            "selection_constraints_ok": False,
            "selection_mean_stationary": False,
            "selection_implied_variance_bounds_ok": False,
            "selection_eligible": False,
            "selection_reason": "",
            "selection_persistence_p": np.nan,
            "selection_persistence_n": np.nan,
            "selection_sigma_min": np.nan,
            "selection_uncond_var_ref": np.nan,
            "selection_max_p_t": np.nan,
            "selection_max_n_t": np.nan,
            "selection_shape_max": np.nan,
            "selection_cond_var_min": np.nan,
            "selection_cond_var_median": np.nan,
            "selection_cond_var_max": np.nan,
            "selection_cond_var_lower_min": np.nan,
            "selection_cond_var_lower_max": np.nan,
            "selection_cond_var_upper_min": np.nan,
            "selection_cond_var_upper_max": np.nan,
        }

        reasons = []
        if not bool(converged.loc[idx]):
            reasons.append("optimizer did not converge")

        try:
            metrics = _selection_metrics_for_row(
                row,
                model_family=model_family,
                specs_by_mean=specs_by_mean,
            )
            diag.update(metrics)
            diag["corrected_loglik_delta"] = diag["corrected_loglik"] - diag["stored_loglik"]
            corrected_finite = all(
                np.isfinite(diag[col])
                for col in ("corrected_loglik", "corrected_AIC", "corrected_BIC")
            )
            if not corrected_finite:
                reasons.append("nonfinite corrected information criterion")
            else:
                diag["selection_loglik_plausible"] = bool(
                    diag["corrected_loglik"] <= IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD
                )
            if not np.isfinite(diag["selection_shape_max"]):
                reasons.append("nonfinite shape path")
            elif diag["selection_shape_max"] >= shape_reference:
                diag["selection_high_shape_density"] = True
            if not np.isfinite(diag["selection_cond_var_min"]) or diag["selection_cond_var_min"] <= 0.0:
                reasons.append("nonpositive conditional variance path")
            if not bool(diag.get("selection_implied_variance_bounds_ok", False)):
                reasons.append("implied variance outside EWMA bounds")

            names, theta = _parameter_vector_from_row(row, model_family)
            bounds = _bounds_for_row(specs_by_mean[row["mean_type"]], row["mean_type"], model_family)
            bounds_ok = all(
                (lo is None or value >= lo - 1e-8) and (hi is None or value <= hi + 1e-8)
                for value, (lo, hi) in zip(theta, bounds)
            )
            constraints_ok = _constraints_ok_for_theta(
                theta,
                mean_type=row["mean_type"],
                model_family=model_family,
            )
            mean_stationary = _mean_stationarity_ok(theta, row["mean_type"])
            diag["selection_bounds_ok"] = bool(bounds_ok)
            diag["selection_constraints_ok"] = bool(constraints_ok)
            diag["selection_mean_stationary"] = bool(mean_stationary)
            if not bounds_ok:
                reasons.append("parameter outside documented bounds")
            if not constraints_ok:
                reasons.append("violates documented stability/variance constraints")
            if not mean_stationary:
                reasons.append("mean process is not stationary")
        except Exception as exc:
            reasons.append(f"diagnostics failed: {type(exc).__name__}: {exc}")

        diag["selection_eligible"] = len(reasons) == 0
        diag["selection_reason"] = "eligible" if diag["selection_eligible"] else "; ".join(reasons)
        diagnostics.append(diag)

    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(diagnostics)], axis=1)
    for metric in ("loglik", "AIC", "BIC"):
        corrected_col = f"corrected_{metric}"
        out[metric] = pd.to_numeric(out[corrected_col], errors="coerce")
    return out


def analysis_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=[metric]).copy()
    if valid.empty:
        return valid

    if "selection_eligible" in valid.columns:
        return valid.loc[valid["selection_eligible"].fillna(False)].copy()

    if "success" in valid.columns:
        successful = valid.loc[strict_success_mask(valid)].copy()
        if not successful.empty:
            valid = successful

    return valid


def best_by_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    valid = analysis_rows(df, metric)
    if valid.empty:
        return pd.Series(dtype=object)
    ascending = metric != "loglik"
    return valid.sort_values(metric, ascending=ascending).iloc[0]


def best_by_mean(df: pd.DataFrame, metric: str = "AIC") -> list[pd.Series]:
    valid = analysis_rows(df, metric)
    if valid.empty or "mean_type" not in valid.columns:
        return []

    rows = []
    for mean_type in MEAN_TYPES:
        group = valid.loc[valid["mean_type"] == mean_type]
        if group.empty:
            continue
        ascending = metric != "loglik"
        rows.append(group.sort_values(metric, ascending=ascending).iloc[0])
    return rows


def mean_file_stem(mean_type: str) -> str:
    return MEAN_FILE_STEMS.get(mean_type, re.sub(r"[^A-Za-z0-9]+", "_", str(mean_type)).strip("_"))


def write_mean_split_csvs(df: pd.DataFrame, results_dir: Path) -> list[Path]:
    split_dir = results_dir / "by_mean"
    split_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mean_type in MEAN_TYPES:
        path = split_dir / f"{mean_file_stem(mean_type)}.csv"
        df.loc[df.get("mean_type") == mean_type].to_csv(path, index=False)
        written.append(path)
    return written


def eligible_result_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[optimizer_success_mask(df) & selection_eligible_mask(df)].copy()


def top_n_by_mean(df: pd.DataFrame, metric: str = "loglik", n: int = TOP_MODELS_PER_MEAN) -> list[pd.Series]:
    valid = analysis_rows(df, metric)
    if valid.empty or "mean_type" not in valid.columns:
        return []

    ascending = metric != "loglik"
    rows: list[pd.Series] = []
    for mean_type in MEAN_TYPES:
        group = valid.loc[valid["mean_type"] == mean_type].sort_values(metric, ascending=ascending)
        for rank, (_, row) in enumerate(group.head(n).iterrows(), start=1):
            ranked = row.copy()
            ranked["rank"] = rank
            rows.append(ranked)
    return rows


def best_overall(df: pd.DataFrame, metric: str = "loglik") -> pd.Series:
    valid = analysis_rows(df, metric)
    if valid.empty:
        return pd.Series(dtype=object)
    ascending = metric != "loglik"
    row = valid.sort_values(metric, ascending=ascending).iloc[0].copy()
    row["rank"] = 1
    return row


def _math_param(row: dict, name: str) -> str:
    estimate = row.get(f"param_{name}", np.nan)
    se = row.get(f"se_{name}", np.nan)
    estimate_text = format_value(estimate)
    if pd.isna(se):
        return estimate_text
    return rf"\underset{{({format_value(se)})}}{{{estimate_text}}}"


def _math_scalar(row: dict, name: str) -> str:
    estimate = row.get(name, np.nan)
    return format_value(estimate)


def _mean_equation(row: dict) -> list[str]:
    mean_type = row["mean_type"]
    if mean_type == "constant":
        return [
            "$$",
            r"\pi_{t+1} = SPF_t + u_{t+1}",
            "$$",
            "",
            "No estimated mean-process coefficients.",
        ]
    if mean_type == "ARX(1,1)":
        return [
            "$$",
            rf"\pi_{{t+1}} = {_math_param(row, 'c')} + {_math_param(row, 'rho_1')}\,\pi_t + {_math_param(row, 'phi_1')}\,SPF_t + u_{{t+1}}",
            "$$",
        ]
    if mean_type == "ARX(2,1)":
        return [
            "$$",
            rf"\pi_{{t+1}} = {_math_param(row, 'c')} + {_math_param(row, 'rho_1')}\,\pi_t + {_math_param(row, 'rho_2')}\,\pi_{{t-1}} + {_math_param(row, 'phi_1')}\,SPF_t + u_{{t+1}}",
            "$$",
        ]
    if mean_type == "ARX(2,2)":
        return [
            "$$",
            rf"\pi_{{t+1}} = {_math_param(row, 'c')} + {_math_param(row, 'rho_1')}\,\pi_t + {_math_param(row, 'rho_2')}\,\pi_{{t-1}} + {_math_param(row, 'phi_1')}\,SPF_t + {_math_param(row, 'phi_2')}\,SPF_{{t-1}} + u_{{t+1}}",
            "$$",
        ]
    raise ValueError(f"Unknown mean_type {mean_type!r}.")


def _volatility_equation(row: dict, model_family: str) -> list[str]:
    sp = _math_param(row, "sigma_p")
    sn = _math_param(row, "sigma_n")
    lines = [
        "$$",
        r"\begin{aligned}",
        rf"u_t &= {sp}\,\omega_{{p,t}} - {sn}\,\omega_{{n,t}},\\",
        r"\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(n_t,1).",
        r"\end{aligned}",
        "$$",
        "",
        "$$",
        r"\begin{aligned}",
    ]

    if model_family == "badgood":
        lines.extend(
            [
                rf"p_t &= {_math_param(row, 'p0')} + {_math_param(row, 'rho_p')}\,p_{{t-1}} + \frac{{{_math_param(row, 'phi_p')}}}{{2({sp})^2}}\,u_{{t-1}}^2,\\",
                rf"n_t &= {_math_param(row, 'n0')} + {_math_param(row, 'rho_n')}\,n_{{t-1}} + \frac{{{_math_param(row, 'phi_n')}}}{{2({sn})^2}}\,u_{{t-1}}^2",
            ]
        )
    elif model_family == "id":
        lines.extend(
            [
                rf"p_t &= {_math_param(row, 'p0')} + {_math_param(row, 'rho_p')}\,p_{{t-1}} + \frac{{{_math_param(row, 'phi_p_plus')}}}{{2({sp})^2}}\,(u_{{t-1}}^+)^2,\\",
                rf"n_t &= {_math_param(row, 'n0')} + {_math_param(row, 'rho_n')}\,n_{{t-1}} + \frac{{{_math_param(row, 'phi_n_minus')}}}{{2({sn})^2}}\,(u_{{t-1}}^-)^2",
            ]
        )
    elif model_family == "full":
        lines.extend(
            [
                rf"p_t &= {_math_param(row, 'p0')} + {_math_param(row, 'rho_p')}\,p_{{t-1}} + \frac{{{_math_param(row, 'phi_p_plus')}}}{{2({sp})^2}}\,(u_{{t-1}}^+)^2 + \frac{{{_math_param(row, 'phi_p_minus')}}}{{2({sp})^2}}\,(u_{{t-1}}^-)^2,\\",
                rf"n_t &= {_math_param(row, 'n0')} + {_math_param(row, 'rho_n')}\,n_{{t-1}} + \frac{{{_math_param(row, 'phi_n_plus')}}}{{2({sn})^2}}\,(u_{{t-1}}^+)^2 + \frac{{{_math_param(row, 'phi_n_minus')}}}{{2({sn})^2}}\,(u_{{t-1}}^-)^2",
            ]
        )
    elif model_family == "symmetric":
        lines.extend(
            [
                rf"p_t &= {_math_param(row, 'p0')} + {_math_param(row, 'rho')}\,p_{{t-1}} + \frac{{{_math_param(row, 'phi_plus')}}}{{2({sp})^2}}\,(u_{{t-1}}^+)^2 + \frac{{{_math_param(row, 'phi_minus')}}}{{2({sp})^2}}\,(u_{{t-1}}^-)^2,\\",
                rf"n_t &= {_math_param(row, 'n0')} + {_math_param(row, 'rho')}\,n_{{t-1}} + \frac{{{_math_param(row, 'phi_plus')}}}{{2({sn})^2}}\,(u_{{t-1}}^+)^2 + \frac{{{_math_param(row, 'phi_minus')}}}{{2({sn})^2}}\,(u_{{t-1}}^-)^2",
            ]
        )
    else:
        raise ValueError(f"Unknown model_family {model_family!r}.")

    lines.extend([r"\end{aligned}", "$$"])
    return lines


def _append_parameter_table(lines: list[str], row: dict, names: list[str]) -> None:
    lines.extend(
        [
            "| Parameter | Estimate | Std. Error |",
            "|---|---:|---:|",
        ]
    )
    for name in names:
        lines.append(
            f"| {name} | {format_value(row.get(f'param_{name}'))} | "
            f"{format_value(row.get(f'se_{name}'))} |"
        )
    lines.append("")


def _append_top20_section(
    lines: list[str],
    mean_type: str,
    rows: list[dict],
    *,
    model_family: str,
    model_param_names: list[str],
) -> None:
    lines.extend([f"## {mean_type}", ""])
    if not rows:
        lines.extend(["No eligible estimates found for this mean process.", ""])
        return

    lines.extend(
        [
            f"Top {len(rows)} admissible estimates ranked by corrected log likelihood.",
            "",
            "| Rank | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        high_flag = "yes" if not bool(row.get("selection_loglik_plausible", True)) else "no"
        lines.append(
            f"| {format_int(row.get('rank'))} | {format_int(row.get('seed'))} | {format_int(row.get('draw'))} | "
            f"{format_value(row.get('loglik'))} | {format_value(row.get('AIC'))} | {format_value(row.get('BIC'))} | "
            f"{format_value(row.get('selection_shape_max'))} | {format_value(row.get('selection_cond_var_max'))} | "
            f"{high_flag} |"
        )
    lines.append("")

    names = MEAN_PARAM_NAMES[mean_type] + model_param_names
    for row in rows:
        lines.extend(
            [
                f"### Rank {format_int(row.get('rank'))}: Seed {format_int(row.get('seed'))}, Draw {format_int(row.get('draw'))}",
                "",
                f"- LogLik: `{format_value(row.get('loglik'))}`; AIC: `{format_value(row.get('AIC'))}`; BIC: `{format_value(row.get('BIC'))}`",
                f"- Max shape path: `{format_value(row.get('selection_shape_max'))}`; max implied variance: `{format_value(row.get('selection_cond_var_max'))}`",
                f"- Selection diagnostics: `{row.get('selection_reason', 'NA')}`",
                "",
                "Mean process:",
                "",
            ]
        )
        lines.extend(_mean_equation(row))
        lines.extend(["", "BEGE volatility process:", ""])
        lines.extend(_volatility_equation(row, model_family))
        lines.extend(["", "Parameter table:", ""])
        _append_parameter_table(lines, row, names)


def _bool_text(value) -> str:
    if pd.isna(value):
        return "NA"
    return "yes" if bool(value) else "no"


def _append_best_model_section(
    lines: list[str],
    row: dict | None,
    *,
    model_family: str,
    model_param_names: list[str],
) -> None:
    lines.extend(["## Selected Best Model", ""])
    if not row:
        lines.extend(["No eligible estimates found for best-model selection.", ""])
        return

    mean_type = row["mean_type"]
    high_flag = "yes" if not bool(row.get("selection_loglik_plausible", True)) else "no"
    lines.extend(
        [
            "Best admissible estimate ranked by corrected log likelihood.",
            "",
            "| Mean | Seed | Draw | LogLik | AIC | BIC | Max Shape | Max Implied Var | Above -150 Diagnostic |",
            "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
            f"| {mean_type} | {format_int(row.get('seed'))} | {format_int(row.get('draw'))} | "
            f"{format_value(row.get('loglik'))} | {format_value(row.get('AIC'))} | {format_value(row.get('BIC'))} | "
            f"{format_value(row.get('selection_shape_max'))} | {format_value(row.get('selection_cond_var_max'))} | "
            f"{high_flag} |",
            "",
            "Selection checks:",
            "",
            f"- Optimizer convergence: `{_bool_text(row.get('optimizer_success', row.get('success', np.nan)))}`",
            f"- Parameter bounds: `{_bool_text(row.get('selection_bounds_ok', np.nan))}`",
            f"- BEGE stability and variance restrictions: `{_bool_text(row.get('selection_constraints_ok', np.nan))}`",
            f"- Shape upper-cap diagnostic: `{'flagged' if bool(row.get('selection_high_shape_density', False)) else 'not flagged'}`",
            f"- Implied variance bounds: `{_bool_text(row.get('selection_implied_variance_bounds_ok', np.nan))}`",
            f"- Mean-process stationarity: `{_bool_text(row.get('selection_mean_stationary', np.nan))}`",
            f"- Selection diagnostics: `{row.get('selection_reason', 'NA')}`",
            f"- Standard errors: `{row.get('se_message', 'not computed')}`",
            "",
            "Mean process:",
            "",
        ]
    )
    lines.extend(_mean_equation(row))
    lines.extend(["", "BEGE volatility process:", ""])
    lines.extend(_volatility_equation(row, model_family))
    lines.extend(["", "Parameter table:", ""])
    _append_parameter_table(lines, row, MEAN_PARAM_NAMES[mean_type] + model_param_names)


def write_markdown_summary(
    df: pd.DataFrame,
    summary_path: Path,
    title: str,
    model_param_names: list[str],
    best_loglik_row: dict | None = None,
    model_family: str | None = None,
) -> None:
    model_family = model_family or ""
    observed_means = set(df.get("mean_type", pd.Series(dtype=str)).dropna().unique())
    missing_means = [mean_type for mean_type in MEAN_TYPES if mean_type not in observed_means]
    eligible_count = int(df.get("selection_eligible", pd.Series(False, index=df.index)).fillna(False).sum())
    converged_count = int(strict_success_mask(df).sum())
    high_loglik_count = int(
        (pd.to_numeric(df.get("corrected_loglik", pd.Series(np.nan, index=df.index)), errors="coerce")
         > IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD).sum()
    )

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        f"# {title}",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Total estimations: `{len(df)}`",
        f"Converged estimations: `{converged_count}`",
        f"Eligible estimations for best-model selection: `{eligible_count}`",
        "",
        "Saved likelihoods are recomputed from the stored parameter paths before ranking. "
        "Large recursive shape states are evaluated by the BEGE saddlepoint density backend; "
        "`max(p_t, n_t)` is reported as a diagnostic, not as an exclusion rule.",
        "",
        "Selection screen: finite corrected AIC/BIC/log-likelihood, successful optimizer status, "
        "finite positive shape paths, positive conditional variance paths, EWMA implied-variance "
        "bounds, mean-process stationarity, and documented parameter/stability/unconditional-variance "
        "constraints. Corrected log likelihoods above "
        f"`{IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD:g}` are flagged for review but are not excluded by this threshold.",
        "This report shows only the single likelihood-best admissible estimate. "
        "Standard errors are computed at the reporting stage for this selected model.",
        "",
    ]

    if missing_means:
        lines.extend(
            [
                "```{warning}",
                "Missing expected mean process results: " + ", ".join(missing_means),
                "```",
                "",
            ]
        )

    if high_loglik_count:
        lines.extend(
            [
                "```{note}",
                f"Flagged {high_loglik_count} estimate(s) with corrected log likelihood above "
                f"`{IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD:g}` for manual review. These rows remain eligible if they pass the admissibility checks.",
                "```",
                "",
            ]
        )

    _append_best_model_section(
        lines,
        best_loglik_row,
        model_family=model_family,
        model_param_names=model_param_names,
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def cleaned_csv_view(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in df.columns
        if col in REPORT_DROP_COLUMNS
        or col.startswith("se_")
        or col.startswith("selection_")
        or col.startswith("corrected_")
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def selection_diagnostics_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seed",
        "draw",
        "mean_type",
        "success",
        "optimizer_success",
        "status",
        "loglik",
        "AIC",
        "BIC",
        "stored_loglik",
        "stored_AIC",
        "stored_BIC",
        "corrected_loglik",
        "corrected_AIC",
        "corrected_BIC",
        "corrected_loglik_delta",
        "selection_eligible",
        "selection_reason",
        "selection_shape_reference",
        "selection_high_shape_density",
        "selection_loglik_upper_threshold",
        "selection_loglik_plausible",
        "selection_variance_bound",
        "selection_bounds_ok",
        "selection_constraints_ok",
        "selection_mean_stationary",
        "selection_implied_variance_bounds_ok",
        "selection_shape_max",
        "selection_max_p_t",
        "selection_max_n_t",
        "selection_sigma_min",
        "selection_persistence_p",
        "selection_persistence_n",
        "selection_uncond_var_ref",
        "selection_cond_var_min",
        "selection_cond_var_median",
        "selection_cond_var_max",
        "selection_cond_var_lower_min",
        "selection_cond_var_lower_max",
        "selection_cond_var_upper_min",
        "selection_cond_var_upper_max",
    ]
    return df[[col for col in cols if col in df.columns]]


def seed_range_from_env():
    start_raw = os.environ.get("START_ID")
    end_raw = os.environ.get("END_ID")
    start = int(start_raw) if start_raw else None
    end = int(end_raw) if end_raw else None
    return start, end


def seed_from_path(path: Path):
    match = DRAW_FILE_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def filter_csv_files(csv_files: list[Path], start_seed, end_seed) -> list[Path]:
    if start_seed is None and end_seed is None:
        return csv_files

    selected = []
    for path in csv_files:
        seed = seed_from_path(path)
        if seed is None:
            continue
        if start_seed is not None and seed < start_seed:
            continue
        if end_seed is not None and seed > end_seed:
            continue
        selected.append(path)
    return selected


def collect_results(
    script_dir: Path,
    title: str,
    model_param_names: list[str],
    model_family: str,
) -> None:
    raw_dir = script_dir / "output" / "raw"
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_seed, end_seed = seed_range_from_env()
    csv_files = filter_csv_files(sorted(raw_dir.glob("draw_*.csv")), start_seed, end_seed)
    used_merged_results = False
    if not csv_files:
        range_note = ""
        if start_seed is not None or end_seed is not None:
            range_note = f" for START_ID={start_seed}, END_ID={end_seed}"
            raise FileNotFoundError(f"No per-seed CSV files found in: {raw_dir}{range_note}")
        merged_path = results_dir / "all_estimations.csv"
        if not merged_path.exists():
            raise FileNotFoundError(f"No per-seed CSV files found in: {raw_dir}")
        all_results = pd.read_csv(merged_path)
        used_merged_results = True
    else:
        frames = [pd.read_csv(path) for path in csv_files]
        all_results = pd.concat(frames, ignore_index=True)

    key_cols = ["seed", "draw", "mean_type"]
    if set(key_cols).issubset(all_results.columns):
        all_results = all_results.drop_duplicates(subset=key_cols, keep="last")

    all_results = all_results.sort_values(["mean_type", "seed", "draw"]).reset_index(drop=True)

    all_results_with_diagnostics = add_selection_diagnostics(
        all_results,
        project_root=script_dir.parents[1],
        model_family=model_family,
    )
    selected_best = best_overall(all_results_with_diagnostics, "loglik")
    best_loglik_rows = []
    if not selected_best.empty:
        best_loglik_rows = add_standard_errors_for_rows(
            [selected_best],
            project_root=script_dir.parents[1],
            model_family=model_family,
        )

    split_paths: list[Path] = []
    if not used_merged_results:
        cleaned_results = cleaned_csv_view(all_results_with_diagnostics)
        cleaned_results.to_csv(results_dir / "all_estimations.csv", index=False)
        by_mean_results = cleaned_csv_view(eligible_result_rows(all_results_with_diagnostics))
        split_paths = write_mean_split_csvs(by_mean_results, results_dir)
        selection_diagnostics_view(all_results_with_diagnostics).to_csv(
            results_dir / "selection_diagnostics.csv",
            index=False,
        )
    stale_se_path = results_dir / "best_loglik_top20_with_se.csv"
    if stale_se_path.exists():
        stale_se_path.unlink()
    best_se_path = results_dir / "best_loglik_with_se.csv"
    pd.DataFrame(best_loglik_rows).to_csv(best_se_path, index=False)
    write_markdown_summary(
        all_results_with_diagnostics,
        results_dir / "best_model.md",
        title,
        model_param_names,
        best_loglik_row=best_loglik_rows[0] if best_loglik_rows else None,
        model_family=model_family,
    )

    if start_seed is not None or end_seed is not None:
        print(f"Seed file filter: START_ID={start_seed}, END_ID={end_seed}")
    if used_merged_results:
        print(f"Read existing merged results from {results_dir / 'all_estimations.csv'}")
    else:
        print(f"Read {len(csv_files)} raw file(s).")
        print(f"Wrote {results_dir / 'all_estimations.csv'}")
        print(f"Wrote {len(split_paths)} mean-process CSV file(s) under {results_dir / 'by_mean'}")
        print(f"Wrote {results_dir / 'selection_diagnostics.csv'}")
    print(f"Wrote {best_se_path}")
    print(f"Wrote {results_dir / 'best_model.md'}")
