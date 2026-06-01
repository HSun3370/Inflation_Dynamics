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

from BEGE_GARCH.BEGE_GARCH import _make_residual_function, bege_variance_bounds_ok
from BEGE_GARCH.bege_batch import build_model_specs, load_effective_sample


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
    return f"{val:.6f}"


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
            "selection_mean_stationary": False,
            "selection_implied_variance_bounds_ok": False,
            "selection_cond_var_min": np.nan,
            "selection_cond_var_median": np.nan,
            "selection_cond_var_max": np.nan,
            "selection_cond_var_lower_min": np.nan,
            "selection_cond_var_lower_max": np.nan,
            "selection_cond_var_upper_min": np.nan,
            "selection_cond_var_upper_max": np.nan,
        }
        reasons = []
        if not bool(success_mask(pd.DataFrame([row])).iloc[0]):
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

            spec = specs_by_mean[mean_type]
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


def append_best_table(lines: list[str], title: str, rows: list[pd.Series]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Mean Type | Seed | Draw | LogLik | AIC | BIC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not rows:
        lines.append("| No finite successful estimates found | NA | NA | NA | NA | NA |")
    for row in rows:
        lines.append(
            f"| {row['mean_type']} | {format_int(row.get('seed'))} | {format_int(row.get('draw'))} | "
            f"{format_value(row.get('loglik'))} | {format_value(row.get('AIC'))} | "
            f"{format_value(row.get('BIC'))} |"
        )
    lines.append("")


def append_parameter_tables(lines: list[str], rows: list[pd.Series]) -> None:
    lines.extend(["## Parameter Estimates From Best Log-Likelihood Fits", ""])
    if not rows:
        lines.extend(["No finite successful estimates found.", ""])
        return

    for row in rows:
        mean_type = row["mean_type"]
        lines.extend(
            [
                f"### {mean_type}",
                "",
                "| Parameter | Estimate | Std. Error |",
                "|---|---:|---:|",
            ]
        )
        for name in PARAMETER_NAMES.get(mean_type, []):
            param_col = f"param_{name}"
            se_col = f"se_{name}"
            lines.append(
                f"| {name} | {format_value(row.get(param_col))} | {format_value(row.get(se_col))} |"
            )
        lines.append("")


def write_markdown_summary(df: pd.DataFrame, summary_path: Path) -> None:
    best_loglik_by_mean = best_by_mean(df)
    observed_means = set(df.get("mean_type", pd.Series(dtype=str)).dropna().unique())
    missing_means = [mean_type for mean_type in MEAN_TYPES if mean_type not in observed_means]
    selection_artifacts = success_mask(df) & numerical_artifact_mask(df)

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
        "EWMA implied-variance bounds, positive conditional variance, and mean-process stationarity.",
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

    if selection_artifacts.any():
        excluded = df.loc[selection_artifacts].sort_values("loglik", ascending=False)
        top = excluded.iloc[0]
        lines.extend(
            [
                "```{warning}",
                f"Excluded {len(excluded)} successful estimate(s) from best-model selection because "
                "`shape_p + shape_n` is numerically an integer, a known unstable point for the "
                "SciPy hyperu BEGE-density evaluation. Top excluded row: "
                f"`{top['mean_type']}`, seed `{format_int(top.get('seed'))}`, "
                f"draw `{format_int(top.get('draw'))}`, reported LogLik "
                f"`{format_value(top.get('loglik'))}`.",
                "```",
                "",
            ]
        )

    append_best_table(lines, "Best by Mean Type (Log-Likelihood)", best_loglik_by_mean)
    append_parameter_tables(lines, best_loglik_by_mean)

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def cleaned_csv_view(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in df.columns
        if col in REPORT_DROP_COLUMNS or col.startswith("se_") or col.startswith("selection_")
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def selection_diagnostics_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seed",
        "draw",
        "mean_type",
        "success",
        "loglik",
        "AIC",
        "BIC",
        "selection_eligible",
        "selection_reason",
        "selection_mean_stationary",
        "selection_implied_variance_bounds_ok",
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
    if not csv_files:
        range_note = ""
        if start_seed is not None or end_seed is not None:
            range_note = f" for START_ID={start_seed}, END_ID={end_seed}"
        raise FileNotFoundError(f"No per-seed CSV files found in: {raw_dir}{range_note}")

    frames = [pd.read_csv(p) for p in csv_files]
    all_results = pd.concat(frames, ignore_index=True)

    key_cols = ["seed", "draw", "mean_type"]
    if set(key_cols).issubset(all_results.columns):
        all_results = all_results.drop_duplicates(subset=key_cols, keep="last")

    all_results = all_results.sort_values(["mean_type", "seed", "draw"]).reset_index(drop=True)

    all_results_with_diagnostics = add_selection_diagnostics(all_results, script_dir.parents[1])

    out_csv = results_dir / "all_estimations.csv"
    out_diag = results_dir / "selection_diagnostics.csv"
    out_md = results_dir / "best_model.md"

    cleaned_csv_view(all_results_with_diagnostics).to_csv(out_csv, index=False)
    selection_diagnostics_view(all_results_with_diagnostics).to_csv(out_diag, index=False)
    write_markdown_summary(all_results_with_diagnostics, out_md)

    if start_seed is not None or end_seed is not None:
        print(f"Seed file filter: START_ID={start_seed}, END_ID={end_seed}")
    print(f"Merged {len(csv_files)} seed files into: {out_csv}")
    print(f"Wrote selection diagnostics: {out_diag}")
    print(f"Wrote summary markdown: {out_md}")


if __name__ == "__main__":
    main()
