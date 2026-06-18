from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.BEGE_GARCH import _make_residual_function, bege_variance_bounds_ok, loglikedgam_constant
from BEGE_GARCH.bege_batch import (
    IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD,
    TOP_MODELS_PER_MEAN,
    _append_path_quantile_table,
    _central_diff_scores,
    _empty_path_quantile_metrics,
    _path_quantile_columns,
    _path_quantile_metrics,
    _project_to_bounds,
    _standard_error_result,
    append_csv_links,
    build_model_specs,
    eligible_result_rows,
    load_effective_sample,
    optimizer_success_mask,
    parameter_label,
    path_quantile_diagnostics_view,
    readme_markdown_from_best_model,
    write_mean_split_csvs,
)


MEAN_TYPES = ["constant", "ARX(1,1)", "ARX(2,1)", "ARX(2,2)"]
REPORT_DROP_COLUMNS = {"message"}
SHAPE_SUM_INTEGER_TOL = 1e-8
MEAN_PARAM_NAMES = {
    "constant": [],
    "ARX(1,1)": ["c", "rho_1", "phi_1"],
    "ARX(2,1)": ["c", "rho_1", "rho_2", "phi_1"],
    "ARX(2,2)": ["c", "rho_1", "rho_2", "phi_1", "phi_2"],
}
PARAMETER_NAMES = {
    "constant": ["shape_p", "shape_n", "sigma_p", "sigma_n"],
    "ARX(1,1)": ["c", "rho_1", "phi_1", "shape_p", "shape_n", "sigma_p", "sigma_n"],
    "ARX(2,1)": ["c", "rho_1", "rho_2", "phi_1", "shape_p", "shape_n", "sigma_p", "sigma_n"],
    "ARX(2,2)": [
        "c",
        "rho_1",
        "rho_2",
        "phi_1",
        "phi_2",
        "shape_p",
        "shape_n",
        "sigma_p",
        "sigma_n",
    ],
}


def format_value(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return f"{val:.4f}"


def format_int(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return str(int(val))


def numerical_artifact_mask(df: pd.DataFrame) -> pd.Series:
    required = {"param_shape_p", "param_shape_n"}
    if not required.issubset(df.columns) or "loglik" not in df.columns:
        return pd.Series(False, index=df.index)

    shape_sum = df["param_shape_p"] + df["param_shape_n"]
    nearest_integer = shape_sum.round()
    near_integer = (shape_sum - nearest_integer).abs() <= SHAPE_SUM_INTEGER_TOL
    reported_positive_loglik = df["loglik"] > 0
    return shape_sum.notna() & (nearest_integer >= 1) & near_integer & reported_positive_loglik


def _analysis_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=[metric]).copy()
    if valid.empty:
        return valid

    if "success" in valid.columns:
        success = success_mask(valid)
        successful = valid.loc[success].copy()
        if not successful.empty:
            valid = successful

    if "optimizer_success" in valid.columns:
        optimizer_successful = valid.loc[optimizer_success_mask(valid)].copy()
        if not optimizer_successful.empty:
            valid = optimizer_successful

    artifacts = numerical_artifact_mask(valid)
    if artifacts.any():
        valid = valid.loc[~artifacts].copy()

    if "selection_eligible" in valid.columns:
        valid = valid.loc[valid["selection_eligible"].fillna(False)].copy()

    return valid


def success_mask(df: pd.DataFrame) -> pd.Series:
    if "success" not in df.columns:
        return pd.Series(False, index=df.index)

    values = df["success"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def ensure_optimizer_success(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "optimizer_success" not in out.columns:
        out["optimizer_success"] = success_mask(out)
    return out


def best_by_mean(df: pd.DataFrame) -> list[pd.Series]:
    valid = _analysis_rows(df, "loglik")
    if valid.empty or "mean_type" not in valid.columns:
        return []

    rows = []
    for mean_type in MEAN_TYPES:
        g = valid.loc[valid["mean_type"] == mean_type]
        if g.empty:
            continue
        rows.append(g.loc[g["loglik"].idxmax()])
    return rows


def _mean_stationarity_ok(mean_type: str, params: np.ndarray) -> bool:
    if mean_type == "constant":
        return True
    if mean_type == "ARX(1,1)":
        return bool(params.size >= 2 and np.isfinite(params[1]) and abs(params[1]) < 1.0)
    if mean_type in {"ARX(2,1)", "ARX(2,2)"}:
        if params.size < 3 or not np.all(np.isfinite(params[1:3])):
            return False
        rho_1, rho_2 = params[1], params[2]
        companion = np.array([[rho_1, rho_2], [1.0, 0.0]], dtype=float)
        return bool(np.all(np.abs(np.linalg.eigvals(companion)) < 1.0))
    return False


def _row_float(row: pd.Series, name: str) -> float:
    return float(row.get(name, np.nan))


def _parameter_vector_from_row(row: pd.Series) -> tuple[list[str], np.ndarray]:
    mean_type = row.get("mean_type")
    names = PARAMETER_NAMES[mean_type]
    params = np.asarray([_row_float(row, f"param_{name}") for name in names], dtype=float)
    if not np.all(np.isfinite(params)):
        raise ValueError(f"Missing finite parameter values for {mean_type}.")
    return names, params


def _bounds_for_row(spec: dict, mean_type: str) -> list[tuple[float | None, float | None]]:
    y = np.asarray(spec["Y"], dtype=float)
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    if mean_type == "constant":
        bounds_mean: list[tuple[float | None, float | None]] = []
    elif mean_type == "ARX(1,1)":
        bounds_mean = [(ymin, ymax), (-1.0, 1.0), (-10.0, 10.0)]
    elif mean_type == "ARX(2,1)":
        bounds_mean = [(ymin, ymax), (-2.0, 2.0), (-1.0, 1.0), (-10.0, 10.0)]
    elif mean_type == "ARX(2,2)":
        bounds_mean = [
            (ymin, ymax),
            (-2.0, 2.0),
            (-1.0, 1.0),
            (-10.0, 10.0),
            (-10.0, 10.0),
        ]
    else:
        raise ValueError(f"Unknown mean_type {mean_type!r}.")

    return bounds_mean + [(0.0, 10.0), (0.0, 10.0), (1e-5, 2.0), (1e-5, 2.0)]


def _bounds_ok(theta: np.ndarray, bounds: list[tuple[float | None, float | None]]) -> bool:
    return bool(
        np.all(np.isfinite(theta))
        and all(
            (lo is None or value >= lo - 1e-8) and (hi is None or value <= hi + 1e-8)
            for value, (lo, hi) in zip(theta, bounds)
        )
    )


def _constant_likelihood_functions(
    *,
    spec: dict,
    mean_type: str,
    bounds: list[tuple[float | None, float | None]],
    big_penalty: float = 1e12,
    big_vec_penalty: float = 1e6,
):
    residual_function = _make_residual_function(spec["Y"], spec["X"], mean_type)
    n_obs = int(np.asarray(spec["Y"], dtype=float).shape[0])
    num_m = len(MEAN_PARAM_NAMES[mean_type])

    def _ind_negloglik(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if not _bounds_ok(theta, bounds) or not _mean_stationarity_ok(mean_type, theta[:num_m]):
            return np.full(n_obs, float(big_vec_penalty), dtype=float)

        shape_p, shape_n, sigma_p, sigma_n = theta[num_m:]
        residuals = residual_function(theta[:num_m])
        pseries = np.full_like(residuals, float(shape_p), dtype=float)
        nseries = np.full_like(residuals, float(shape_n), dtype=float)
        if not bege_variance_bounds_ok(residuals, pseries, nseries, sigma_p, sigma_n):
            return np.full(n_obs, float(big_vec_penalty), dtype=float)

        values = -loglikedgam_constant(residuals, shape_p, shape_n, sigma_p, sigma_n)
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


def compute_standard_errors_for_row(row: pd.Series, *, spec: dict) -> dict:
    from statsmodels.tools.numdiff import approx_hess

    mean_type = row["mean_type"]
    names, theta = _parameter_vector_from_row(row)
    bounds = _bounds_for_row(spec, mean_type)
    theta_eval = _project_to_bounds(theta, bounds)
    n_obs = int(np.asarray(spec["Y"], dtype=float).shape[0])
    negloglik, ind_negloglik = _constant_likelihood_functions(
        spec=spec,
        mean_type=mean_type,
        bounds=bounds,
    )

    obj_value = negloglik(theta_eval)
    if not np.isfinite(obj_value) or obj_value >= 1e12:
        raise ValueError("Likelihood is not finite at the supplied estimate.")

    scores = _central_diff_scores(theta_eval, ind_negloglik, bounds, n_obs)
    hessian = approx_hess(theta_eval, negloglik, epsilon=1e-5)
    return _standard_error_result(
        names=names,
        theta=theta_eval,
        bounds=bounds,
        hessian=hessian,
        scores=scores,
    )


def add_standard_errors_for_rows(rows: list[pd.Series], *, project_root: Path) -> list[dict]:
    specs_by_mean = {
        spec["mean_type"]: spec
        for spec in build_model_specs(load_effective_sample(project_root), include_arx22=True)
    }

    enriched_rows: list[dict] = []
    for row in rows:
        enriched = row.to_dict()
        try:
            enriched.update(compute_standard_errors_for_row(row, spec=specs_by_mean[row["mean_type"]]))
        except Exception as exc:
            enriched["se_message"] = f"{type(exc).__name__}: {exc}"
        enriched_rows.append(enriched)
    return enriched_rows


def add_selection_diagnostics(df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    specs_by_mean = {
        spec["mean_type"]: spec
        for spec in build_model_specs(load_effective_sample(project_root), include_arx22=True)
    }

    diagnostics = []
    for _, row in df.iterrows():
        mean_type = row.get("mean_type")
        diag = {
            "selection_eligible": False,
            "selection_reason": "",
            "selection_bounds_ok": False,
            "selection_mean_stationary": False,
            "selection_loglik_upper_threshold": float(IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD),
            "selection_loglik_plausible": False,
            "selection_implied_variance_bounds_ok": False,
            "selection_cond_var_min": np.nan,
            "selection_cond_var_median": np.nan,
            "selection_cond_var_max": np.nan,
            "selection_cond_var_lower_min": np.nan,
            "selection_cond_var_lower_max": np.nan,
            "selection_cond_var_upper_min": np.nan,
            "selection_cond_var_upper_max": np.nan,
        }
        diag.update(_empty_path_quantile_metrics())
        reasons = []
        row_df = pd.DataFrame([row])
        if not bool(success_mask(row_df).iloc[0]) or not bool(optimizer_success_mask(row_df).iloc[0]):
            reasons.append("optimizer did not converge")

        try:
            if mean_type not in specs_by_mean:
                raise KeyError(f"Unknown mean_type {mean_type!r}.")
            mean_names = MEAN_PARAM_NAMES[mean_type]
            mean_params = np.asarray([float(row.get(f"param_{name}", np.nan)) for name in mean_names], dtype=float)
            shape_p = float(row.get("param_shape_p", np.nan))
            shape_n = float(row.get("param_shape_n", np.nan))
            sigma_p = float(row.get("param_sigma_p", np.nan))
            sigma_n = float(row.get("param_sigma_n", np.nan))
            values = np.asarray([shape_p, shape_n, sigma_p, sigma_n], dtype=float)
            if not np.all(np.isfinite(values)) or shape_p <= 0.0 or shape_n <= 0.0 or sigma_p <= 0.0 or sigma_n <= 0.0:
                reasons.append("nonfinite or nonpositive BEGE parameters")
            loglik = float(row.get("loglik", np.nan))
            diag["selection_loglik_plausible"] = bool(
                np.isfinite(loglik) and loglik <= IMPLAUSIBLY_HIGH_LOGLIK_THRESHOLD
            )

            spec = specs_by_mean[mean_type]
            _, theta = _parameter_vector_from_row(row)
            bounds_ok = _bounds_ok(theta, _bounds_for_row(spec, mean_type))
            diag["selection_bounds_ok"] = bool(bounds_ok)
            if not bounds_ok:
                reasons.append("parameter outside documented bounds")

            residuals = _make_residual_function(spec["Y"], spec["X"], mean_type)(mean_params)
            pseries = np.full_like(residuals, shape_p, dtype=float)
            nseries = np.full_like(residuals, shape_n, dtype=float)
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
            cond_skewness = 2.0 * (sigma_p**3 * pseries - sigma_n**3 * nseries)
            cond_excess_kurtosis = 6.0 * (sigma_p**4 * pseries + sigma_n**4 * nseries)
            diag.update(
                {
                    "selection_mean_stationary": _mean_stationarity_ok(mean_type, mean_params),
                    "selection_implied_variance_bounds_ok": bool(variance_details["ok"]),
                    "selection_cond_var_min": float(np.min(cond_var)),
                    "selection_cond_var_median": float(np.median(cond_var)),
                    "selection_cond_var_max": float(np.max(cond_var)),
                    "selection_cond_var_lower_min": float(np.min(lower)),
                    "selection_cond_var_lower_max": float(np.max(lower)),
                    "selection_cond_var_upper_min": float(np.min(upper)),
                    "selection_cond_var_upper_max": float(np.max(upper)),
                }
            )
            diag.update(_path_quantile_metrics("selection_p_t", pseries))
            diag.update(_path_quantile_metrics("selection_n_t", nseries))
            diag.update(_path_quantile_metrics("selection_cond_var", cond_var))
            diag.update(_path_quantile_metrics("selection_cond_skewness", cond_skewness))
            diag.update(_path_quantile_metrics("selection_cond_excess_kurtosis", cond_excess_kurtosis))
            if not diag["selection_mean_stationary"]:
                reasons.append("mean process is not stationary")
            if not diag["selection_implied_variance_bounds_ok"]:
                reasons.append("implied variance outside EWMA bounds")
            if not np.isfinite(diag["selection_cond_var_min"]) or diag["selection_cond_var_min"] <= 0.0:
                reasons.append("nonpositive conditional variance path")
        except Exception as exc:
            reasons.append(f"diagnostics failed: {type(exc).__name__}: {exc}")

        diag["selection_eligible"] = len(reasons) == 0
        diag["selection_reason"] = "eligible" if diag["selection_eligible"] else "; ".join(reasons)
        diagnostics.append(diag)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(diagnostics)], axis=1)


def top_n_by_mean(df: pd.DataFrame, n: int = TOP_MODELS_PER_MEAN) -> list[pd.Series]:
    valid = _analysis_rows(df, "loglik")
    if valid.empty or "mean_type" not in valid.columns:
        return []

    rows: list[pd.Series] = []
    for mean_type in MEAN_TYPES:
        group = valid.loc[valid["mean_type"] == mean_type].sort_values("loglik", ascending=False)
        for rank, (_, row) in enumerate(group.head(n).iterrows(), start=1):
            ranked = row.copy()
            ranked["rank"] = rank
            rows.append(ranked)
    return rows


def best_overall(df: pd.DataFrame) -> pd.Series:
    valid = _analysis_rows(df, "loglik")
    if valid.empty:
        return pd.Series(dtype=object)
    row = valid.sort_values("loglik", ascending=False).iloc[0].copy()
    row["rank"] = 1
    return row


def _math_param(row: dict, name: str) -> str:
    estimate = row.get(f"param_{name}", np.nan)
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


def _constant_volatility_equation(row: dict) -> list[str]:
    sp = _math_param(row, "sigma_p")
    sn = _math_param(row, "sigma_n")
    shape_p = _math_param(row, "shape_p")
    shape_n = _math_param(row, "shape_n")
    return [
        "$$",
        r"\begin{aligned}",
        rf"u_t &= {sp}\,\omega_{{p,t}} - {sn}\,\omega_{{n,t}},\\",
        r"\omega_{p,t} &\sim \tilde{\Gamma}(\bar{p},1),\qquad \omega_{n,t}\sim \tilde{\Gamma}(\bar{n},1),\\",
        rf"\bar{{p}} &= {shape_p},\qquad \bar{{n}} = {shape_n},\\",
        rf"\operatorname{{Var}}_t(u_t) &= ({sp})^2\,{shape_p} + ({sn})^2\,{shape_n}.",
        r"\end{aligned}",
        "$$",
    ]


def _append_parameter_table(lines: list[str], row: dict, names: list[str]) -> None:
    lines.extend(["| Parameter | Estimate | Std. Error |", "|---|---:|---:|"])
    for name in names:
        lines.append(
            f"| {parameter_label(name)} | {format_value(row.get(f'param_{name}'))} | "
            f"{format_value(row.get(f'se_{name}'))} |"
        )
    lines.append("")


def _append_top20_section(lines: list[str], mean_type: str, rows: list[dict]) -> None:
    lines.extend([f"## {mean_type}", ""])
    if not rows:
        lines.extend(["No eligible estimates found for this mean process.", ""])
        return

    lines.extend(
        [
            f"Top {len(rows)} admissible estimates ranked by log likelihood.",
            "",
            "| Rank | Seed | Draw | LogLik | AIC | BIC | Implied Var |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {format_int(row.get('rank'))} | {format_int(row.get('seed'))} | {format_int(row.get('draw'))} | "
            f"{format_value(row.get('loglik'))} | {format_value(row.get('AIC'))} | {format_value(row.get('BIC'))} | "
            f"{format_value(row.get('selection_cond_var_max'))} |"
        )
    lines.append("")

    for row in rows:
        lines.extend(
            [
                f"### Rank {format_int(row.get('rank'))}: Seed {format_int(row.get('seed'))}, Draw {format_int(row.get('draw'))}",
                "",
                f"- LogLik: `{format_value(row.get('loglik'))}`; AIC: `{format_value(row.get('AIC'))}`; BIC: `{format_value(row.get('BIC'))}`",
                f"- Implied variance: `{format_value(row.get('selection_cond_var_max'))}`",
                f"- Selection diagnostics: `{row.get('selection_reason', 'NA')}`",
                "",
                "Mean process:",
                "",
            ]
        )
        lines.extend(_mean_equation(row))
        lines.extend(["", "BEGE volatility process:", ""])
        lines.extend(_constant_volatility_equation(row))
        lines.extend(["", "Parameter table:", ""])
        _append_parameter_table(lines, row, PARAMETER_NAMES[mean_type])


def _bool_text(value) -> str:
    if pd.isna(value):
        return "NA"
    return "yes" if bool(value) else "no"


def _best_initial_shapes_view(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "mean_type": row.get("mean_type"),
                "seed": row.get("seed"),
                "draw": row.get("draw"),
                "loglik": row.get("loglik"),
                "AIC": row.get("AIC"),
                "BIC": row.get("BIC"),
                "p_bar": row.get("param_shape_p"),
                "n_bar": row.get("param_shape_n"),
                "initial_p_0": row.get("param_shape_p"),
                "initial_n_0": row.get("param_shape_n"),
                "se_p_bar": row.get("se_shape_p"),
                "se_n_bar": row.get("se_shape_n"),
                "sigma_p": row.get("param_sigma_p"),
                "sigma_n": row.get("param_sigma_n"),
                "implied_variance": row.get("selection_cond_var_median"),
                "optimizer_success": row.get("optimizer_success", row.get("success", np.nan)),
                "parameter_bounds_ok": row.get("selection_bounds_ok"),
                "implied_variance_bounds_ok": row.get("selection_implied_variance_bounds_ok"),
                "mean_stationary": row.get("selection_mean_stationary"),
                "selection_reason": row.get("selection_reason"),
            }
        )

    columns = [
        "mean_type",
        "seed",
        "draw",
        "loglik",
        "AIC",
        "BIC",
        "p_bar",
        "n_bar",
        "initial_p_0",
        "initial_n_0",
        "se_p_bar",
        "se_n_bar",
        "sigma_p",
        "sigma_n",
        "implied_variance",
        "optimizer_success",
        "parameter_bounds_ok",
        "implied_variance_bounds_ok",
        "mean_stationary",
        "selection_reason",
    ]
    return pd.DataFrame(records, columns=columns)


def _append_best_by_mean_section(
    lines: list[str],
    rows: list[dict],
    *,
    initial_shapes_path: Path | None = None,
) -> None:
    lines.extend(["## Selected Best Model", ""])
    if not rows:
        lines.extend(["No eligible estimates found for best-model selection.", ""])
        return

    overall_best = max(
        rows,
        key=lambda row: float(row.get("loglik", -np.inf))
        if pd.notna(row.get("loglik", np.nan))
        else -np.inf,
    )
    mean_type = overall_best.get("mean_type")

    lines.extend(
        [
            "Best admissible estimate ranked by log likelihood across mean processes.",
            "",
            "| Mean | Seed | Draw | LogLik | AIC | BIC |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {mean_type} | {format_int(overall_best.get('seed'))} | {format_int(overall_best.get('draw'))} | "
            f"{format_value(overall_best.get('loglik'))} | {format_value(overall_best.get('AIC'))} | "
            f"{format_value(overall_best.get('BIC'))} |",
            "",
            "Selection checks:",
            "",
            f"- Optimizer convergence: `{_bool_text(overall_best.get('optimizer_success', overall_best.get('success', np.nan)))}`",
            f"- Parameter bounds: `{_bool_text(overall_best.get('selection_bounds_ok', np.nan))}`",
            f"- Implied variance bounds: `{_bool_text(overall_best.get('selection_implied_variance_bounds_ok', np.nan))}`",
            f"- Mean-process stationarity: `{_bool_text(overall_best.get('selection_mean_stationary', np.nan))}`",
            f"- Standard errors: `{overall_best.get('se_message', 'not computed')}`",
            "",
        ]
    )
    _append_path_quantile_table(lines, overall_best)
    lines.extend(["Mean process:", ""])
    lines.extend(_mean_equation(overall_best))
    lines.extend(["", "BEGE volatility process:", ""])
    lines.extend(_constant_volatility_equation(overall_best))
    lines.extend(["", "Parameter table:", ""])
    _append_parameter_table(lines, overall_best, PARAMETER_NAMES[mean_type])


def write_markdown_summary(
    df: pd.DataFrame,
    summary_path: Path,
    best_loglik_rows: list[dict] | None = None,
    initial_shapes_path: Path | None = None,
) -> None:
    observed_means = set(df.get("mean_type", pd.Series(dtype=str)).dropna().unique())
    missing_means = [mean_type for mean_type in MEAN_TYPES if mean_type not in observed_means]

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        "# Constant BEGE Best Model Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Total estimations: `{len(df)}`",
        f"Successful estimations: `{int(success_mask(df).sum())}`",
        f"Eligible estimations for best-model selection: `{int(df.get('selection_eligible', pd.Series(False, index=df.index)).fillna(False).sum())}`",
        "",
        "Selection screen: successful optimizer status, finite positive BEGE parameters, "
        "documented parameter bounds, EWMA implied-variance bounds, positive conditional variance, "
        "and mean-process stationarity.",
        "This report shows only the single likelihood-best admissible estimate across mean processes. "
        "Standard errors are computed at the reporting stage and reported in the parameter table.",
        "",
    ]
    append_csv_links(lines, summary_path.parent, include_constant_shapes=True)

    if missing_means:
        lines.extend(
            [
                "```{warning}",
                "Missing expected mean process results: " + ", ".join(missing_means),
                "```",
                "",
            ]
        )

    _append_best_by_mean_section(
        lines,
        best_loglik_rows or [],
        initial_shapes_path=initial_shapes_path,
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def cleaned_csv_view(df: pd.DataFrame, *, include_path_quantiles: bool = False) -> pd.DataFrame:
    retained_selection_cols = set(_path_quantile_columns()) if include_path_quantiles else set()
    drop_cols = [
        col
        for col in df.columns
        if col in REPORT_DROP_COLUMNS
        or col.startswith("se_")
        or (col.startswith("selection_") and col not in retained_selection_cols)
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def selection_diagnostics_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seed",
        "draw",
        "mean_type",
        "success",
        "optimizer_success",
        "loglik",
        "AIC",
        "BIC",
        "selection_eligible",
        "selection_reason",
        "selection_bounds_ok",
        "selection_mean_stationary",
        "selection_loglik_upper_threshold",
        "selection_loglik_plausible",
        "selection_implied_variance_bounds_ok",
        "selection_cond_var_min",
        "selection_cond_var_median",
        "selection_cond_var_max",
        *_path_quantile_columns(),
        "selection_cond_var_lower_min",
        "selection_cond_var_lower_max",
        "selection_cond_var_upper_min",
        "selection_cond_var_upper_max",
    ]
    return df[[col for col in dict.fromkeys(cols) if col in df.columns]]


def seed_range_from_env():
    start_raw = os.environ.get("START_ID")
    end_raw = os.environ.get("END_ID")
    start = int(start_raw) if start_raw else None
    end = int(end_raw) if end_raw else None
    return start, end


def seed_from_path(path: Path):
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


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


def main() -> None:
    script_dir = Path(__file__).resolve().parent
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
        frames = [pd.read_csv(p) for p in csv_files]
        all_results = pd.concat(frames, ignore_index=True)

    key_cols = ["seed", "draw", "mean_type"]
    if set(key_cols).issubset(all_results.columns):
        all_results = all_results.drop_duplicates(subset=key_cols, keep="last")

    all_results = all_results.sort_values(["mean_type", "seed", "draw"]).reset_index(drop=True)

    all_results = ensure_optimizer_success(all_results)
    all_results_with_diagnostics = add_selection_diagnostics(all_results, script_dir.parents[1])
    path_quantile_path = results_dir / "path_quantile_diagnostics.csv"
    path_quantile_diagnostics_view(all_results_with_diagnostics).to_csv(
        path_quantile_path,
        index=False,
    )
    selected_by_mean = best_by_mean(all_results_with_diagnostics)
    best_loglik_rows = []
    if selected_by_mean:
        best_loglik_rows = add_standard_errors_for_rows(
            selected_by_mean,
            project_root=script_dir.parents[1],
        )

    out_csv = results_dir / "all_estimations.csv"
    out_diag = results_dir / "selection_diagnostics.csv"
    out_md = results_dir / "best_model.md"

    split_paths: list[Path] = []
    if not used_merged_results:
        cleaned_results = cleaned_csv_view(all_results_with_diagnostics)
        cleaned_results.to_csv(out_csv, index=False)
        by_mean_results = cleaned_csv_view(
            eligible_result_rows(all_results_with_diagnostics),
            include_path_quantiles=True,
        )
        split_paths = write_mean_split_csvs(by_mean_results, results_dir)
        selection_diagnostics_view(all_results_with_diagnostics).to_csv(out_diag, index=False)
    stale_se_path = results_dir / "best_loglik_top20_with_se.csv"
    if stale_se_path.exists():
        stale_se_path.unlink()
    best_se_path = results_dir / "best_loglik_with_se.csv"
    pd.DataFrame(best_loglik_rows).to_csv(best_se_path, index=False)
    initial_shapes_path = results_dir / "constant_bege_initial_shapes_by_mean.csv"
    _best_initial_shapes_view(best_loglik_rows).to_csv(initial_shapes_path, index=False)
    write_markdown_summary(
        all_results_with_diagnostics,
        out_md,
        best_loglik_rows=best_loglik_rows,
        initial_shapes_path=initial_shapes_path,
    )
    readme_path = script_dir / "README.md"
    if results_dir.resolve() == (script_dir / "results").resolve():
        readme_path.write_text(
            readme_markdown_from_best_model(out_md.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

    if start_seed is not None or end_seed is not None:
        print(f"Seed file filter: START_ID={start_seed}, END_ID={end_seed}")
    if used_merged_results:
        print(f"Read existing merged results from: {out_csv}")
    else:
        print(f"Merged {len(csv_files)} seed files into: {out_csv}")
        print(f"Wrote {len(split_paths)} mean-process CSV file(s) under {results_dir / 'by_mean'}")
        print(f"Wrote selection diagnostics: {out_diag}")
    print(f"Wrote path quantile diagnostics: {path_quantile_path}")
    print(f"Wrote selected best models with SEs: {best_se_path}")
    print(f"Wrote selected initial shapes: {initial_shapes_path}")
    print(f"Wrote summary markdown: {out_md}")
    if results_dir.resolve() == (script_dir / "results").resolve():
        print(f"Wrote summary README: {readme_path}")


if __name__ == "__main__":
    main()
