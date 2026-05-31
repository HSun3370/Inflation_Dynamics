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
DEFAULT_SELECTION_SHAPE_CAP = 200.0


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

    cond_var = sigma_p * sigma_p * pseries + sigma_n * sigma_n * nseries
    return {
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
    }


def add_selection_diagnostics(
    df: pd.DataFrame,
    *,
    project_root: Path,
    model_family: str,
    shape_cap: float = DEFAULT_SELECTION_SHAPE_CAP,
) -> pd.DataFrame:
    out = df.copy()
    diagnostics = []
    specs_by_mean = {
        spec["mean_type"]: spec
        for spec in build_model_specs(load_effective_sample(project_root), include_arx22=True)
    }

    finite = _finite_metric_mask(out)
    converged = strict_success_mask(out)

    for idx, row in out.iterrows():
        diag = {
            "selection_shape_cap": float(shape_cap),
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
        }

        reasons = []
        if not bool(finite.loc[idx]):
            reasons.append("nonfinite information criterion")
        if not bool(converged.loc[idx]):
            reasons.append("optimizer did not converge")

        try:
            metrics = _selection_metrics_for_row(
                row,
                model_family=model_family,
                specs_by_mean=specs_by_mean,
            )
            diag.update(metrics)
            if not np.isfinite(diag["selection_shape_max"]):
                reasons.append("nonfinite shape path")
            elif diag["selection_shape_max"] >= shape_cap:
                reasons.append(f"shape cap exceeded ({diag['selection_shape_max']:.6g} >= {shape_cap:g})")
        except Exception as exc:
            reasons.append(f"diagnostics failed: {type(exc).__name__}: {exc}")

        diag["selection_eligible"] = len(reasons) == 0
        diag["selection_reason"] = "eligible" if diag["selection_eligible"] else "; ".join(reasons)
        diagnostics.append(diag)

    return pd.concat([out.reset_index(drop=True), pd.DataFrame(diagnostics)], axis=1)


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


def append_best_table(lines: list[str], title: str, rows: list[pd.Series]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Mean Type | Seed | Draw | AIC | BIC | LogLik | Max Shape | Min Sigma |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if not rows:
        lines.append("| No eligible estimates found | NA | NA | NA | NA | NA | NA | NA |")
    for row in rows:
        lines.append(
            f"| {row['mean_type']} | {format_int(row.get('seed'))} | {format_int(row.get('draw'))} | "
            f"{format_value(row.get('AIC'))} | {format_value(row.get('BIC'))} | "
            f"{format_value(row.get('loglik'))} | {format_value(row.get('selection_shape_max'))} | "
            f"{format_value(row.get('selection_sigma_min'))} |"
        )
    lines.append("")


def append_parameter_tables(lines: list[str], rows: list[pd.Series], model_param_names: list[str]) -> None:
    lines.extend(["## Parameter Estimates From Eligible Best AIC Fits", ""])
    if not rows:
        lines.extend(["No eligible estimates found.", ""])
        return

    for row in rows:
        mean_type = row["mean_type"]
        names = MEAN_PARAM_NAMES[mean_type] + model_param_names
        lines.extend(
            [
                f"### {mean_type}",
                "",
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


def write_markdown_summary(
    df: pd.DataFrame,
    summary_path: Path,
    title: str,
    model_param_names: list[str],
) -> None:
    best_aic = best_by_metric(df, "AIC")
    best_bic = best_by_metric(df, "BIC")
    best_aic_rows = best_by_mean(df, "AIC")
    best_bic_rows = best_by_mean(df, "BIC")
    observed_means = set(df.get("mean_type", pd.Series(dtype=str)).dropna().unique())
    missing_means = [mean_type for mean_type in MEAN_TYPES if mean_type not in observed_means]
    eligible_count = int(df.get("selection_eligible", pd.Series(False, index=df.index)).fillna(False).sum())
    converged_count = int(strict_success_mask(df).sum())

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
        f"Selection screen: finite AIC/BIC/log-likelihood, successful optimizer status, and "
        f"`max(p_t, n_t) < {DEFAULT_SELECTION_SHAPE_CAP:g}`.",
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

    if not best_aic.empty:
        lines.extend(
            [
                "## Global Best by AIC",
                "",
                f"- Mean type: `{best_aic['mean_type']}`",
                f"- Seed / draw: `{format_int(best_aic.get('seed'))}` / `{format_int(best_aic.get('draw'))}`",
                f"- AIC: `{format_value(best_aic.get('AIC'))}`",
                f"- BIC: `{format_value(best_aic.get('BIC'))}`",
                f"- LogLik: `{format_value(best_aic.get('loglik'))}`",
                f"- Max shape: `{format_value(best_aic.get('selection_shape_max'))}`",
                "",
            ]
        )

    if not best_bic.empty:
        lines.extend(
            [
                "## Global Best by BIC",
                "",
                f"- Mean type: `{best_bic['mean_type']}`",
                f"- Seed / draw: `{format_int(best_bic.get('seed'))}` / `{format_int(best_bic.get('draw'))}`",
                f"- AIC: `{format_value(best_bic.get('AIC'))}`",
                f"- BIC: `{format_value(best_bic.get('BIC'))}`",
                f"- LogLik: `{format_value(best_bic.get('loglik'))}`",
                f"- Max shape: `{format_value(best_bic.get('selection_shape_max'))}`",
                "",
            ]
        )

    append_best_table(lines, "Eligible Best by Mean Type (AIC)", best_aic_rows)
    append_best_table(lines, "Eligible Best by Mean Type (BIC)", best_bic_rows)
    append_parameter_tables(lines, best_aic_rows, model_param_names)

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def cleaned_csv_view(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if col in REPORT_DROP_COLUMNS or col.startswith("se_")]
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
        "selection_eligible",
        "selection_reason",
        "selection_shape_cap",
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
    if not csv_files:
        range_note = ""
        if start_seed is not None or end_seed is not None:
            range_note = f" for START_ID={start_seed}, END_ID={end_seed}"
        raise FileNotFoundError(f"No per-seed CSV files found in: {raw_dir}{range_note}")

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

    cleaned_csv_view(all_results).to_csv(results_dir / "all_estimations.csv", index=False)
    selection_diagnostics_view(all_results_with_diagnostics).to_csv(
        results_dir / "selection_diagnostics.csv",
        index=False,
    )
    write_markdown_summary(
        all_results_with_diagnostics,
        results_dir / "best_model.md",
        title,
        model_param_names,
    )

    if start_seed is not None or end_seed is not None:
        print(f"Seed file filter: START_ID={start_seed}, END_ID={end_seed}")
    print(f"Read {len(csv_files)} raw file(s).")
    print(f"Wrote {results_dir / 'all_estimations.csv'}")
    print(f"Wrote {results_dir / 'selection_diagnostics.csv'}")
    print(f"Wrote {results_dir / 'best_model.md'}")
