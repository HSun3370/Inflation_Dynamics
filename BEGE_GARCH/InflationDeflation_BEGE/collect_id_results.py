from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.BEGE_GARCH import ID_GARCH
from BEGE_GARCH.bege_batch import (
    DEFAULT_SELECTION_SHAPE_CAP,
    add_selection_diagnostics,
    selection_diagnostics_view,
    strict_success_mask,
)
from BEGE_GARCH.InflationDeflation_BEGE.ID_GJR1 import (
    MEAN_PARAM_NAMES,
    MEAN_TYPES,
    VOL_PARAM_NAMES,
    build_model_specs,
    load_effective_sample,
)


def format_value(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return f"{float(val):.6f}"


def format_int(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return str(int(val))


def success_mask(df: pd.DataFrame) -> pd.Series:
    return strict_success_mask(df)


def analysis_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()
    valid = df.dropna(subset=[metric]).copy()
    if valid.empty:
        return valid

    if "selection_eligible" in valid.columns:
        return valid.loc[valid["selection_eligible"].fillna(False)].copy()

    successful = valid.loc[success_mask(valid)].copy()
    if not successful.empty:
        valid = successful
    return valid


def best_by_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    valid = analysis_rows(df, metric)
    if valid.empty:
        return pd.Series(dtype=object)
    return valid.loc[valid[metric].idxmin()]


def best_by_mean(df: pd.DataFrame, metric: str) -> list[pd.Series]:
    valid = analysis_rows(df, metric)
    if valid.empty:
        return []
    rows = []
    for mean_type in MEAN_TYPES:
        g = valid.loc[valid["mean_type"] == mean_type]
        if not g.empty:
            rows.append(g.loc[g[metric].idxmin()])
    return rows


def row_params(row: pd.Series) -> np.ndarray:
    mean_type = row["mean_type"]
    names = MEAN_PARAM_NAMES[mean_type] + VOL_PARAM_NAMES
    params = [row.get(f"param_{name}") for name in names]
    params = np.asarray(params, dtype=float)
    if np.any(~np.isfinite(params)):
        raise ValueError(f"Missing finite parameter values for {mean_type}.")
    return params


def add_se_for_best_rows(rows: list[pd.Series], specs_by_mean: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        enriched = row.to_dict()
        mean_type = row["mean_type"]
        try:
            spec = specs_by_mean[mean_type]
            result = ID_GARCH(
                Y=spec["Y"],
                X=spec["X"],
                mean_type=mean_type,
                n_starts=0,
                maxiter=0,
                tol=1e-8,
                initial_params=row_params(row),
                compute_se=True,
                print_summary=False,
            )
            names = MEAN_PARAM_NAMES[mean_type] + VOL_PARAM_NAMES
            for name, se in zip(names, np.asarray(result["se"], dtype=float)):
                enriched[f"se_{name}"] = float(se)
            enriched["se_message"] = "computed"
        except Exception as exc:
            enriched["se_message"] = f"{type(exc).__name__}: {exc}"
        out.append(enriched)
    return out


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


def append_parameter_tables(lines: list[str], rows: list[dict]) -> None:
    lines.extend(["## Parameter Estimates From Eligible Best AIC Fits", ""])
    if not rows:
        lines.extend(["No eligible estimates found.", ""])
        return

    for row in rows:
        mean_type = row["mean_type"]
        lines.extend(
            [
                f"### {mean_type}",
                "",
                f"SE status: `{row.get('se_message', 'NA')}`",
                "",
                "| Parameter | Estimate | Std. Error |",
                "|---|---:|---:|",
            ]
        )
        for name in MEAN_PARAM_NAMES[mean_type] + VOL_PARAM_NAMES:
            lines.append(
                f"| {name} | {format_value(row.get(f'param_{name}'))} | "
                f"{format_value(row.get(f'se_{name}'))} |"
            )
        lines.append("")


def write_markdown_summary(df: pd.DataFrame, best_aic_with_se: list[dict], summary_path: Path) -> None:
    best_aic = best_by_metric(df, "AIC")
    best_bic = best_by_metric(df, "BIC")
    best_aic_by_mean = best_by_mean(df, "AIC")
    best_bic_by_mean = best_by_mean(df, "BIC")
    observed_means = set(df.get("mean_type", pd.Series(dtype=str)).dropna().unique())
    missing_means = [mean_type for mean_type in MEAN_TYPES if mean_type not in observed_means]
    eligible_count = int(df.get("selection_eligible", pd.Series(False, index=df.index)).fillna(False).sum())

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        "# Inflation/Deflation BEGE-GJR Best Model Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Total saved estimations: `{len(df)}`",
        f"Converged estimations: `{int(success_mask(df).sum())}`",
        f"Eligible estimations for best-model selection: `{eligible_count}`",
        "",
        "SEs are skipped during Slurm estimation jobs and computed only for the eligible best AIC fit in each mean process.",
        "",
        "Selection screen: finite AIC/BIC/log-likelihood, successful optimizer status, and "
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

    append_best_table(lines, "Eligible Best by Mean Type (AIC)", best_aic_by_mean)
    append_best_table(lines, "Eligible Best by Mean Type (BIC)", best_bic_by_mean)
    append_parameter_tables(lines, best_aic_with_se)

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw_dir = SCRIPT_DIR / "output" / "raw"
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("draw_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No per-seed CSV files found in: {raw_dir}")

    frames = [pd.read_csv(path) for path in csv_files]
    all_results = pd.concat(frames, ignore_index=True)
    key_cols = ["seed", "draw", "mean_type"]
    if set(key_cols).issubset(all_results.columns):
        all_results = all_results.drop_duplicates(subset=key_cols, keep="last")
    all_results = all_results.sort_values(["mean_type", "seed", "draw"]).reset_index(drop=True)

    all_results_with_diagnostics = add_selection_diagnostics(
        all_results,
        project_root=PROJECT_ROOT,
        model_family="id",
    )

    df = load_effective_sample(PROJECT_ROOT)
    specs_by_mean = {spec["mean_type"]: spec for spec in build_model_specs(df)}
    best_aic_with_se = add_se_for_best_rows(
        best_by_mean(all_results_with_diagnostics, "AIC"),
        specs_by_mean,
    )

    out_csv = results_dir / "all_estimations.csv"
    out_best_csv = results_dir / "best_aic_with_se.csv"
    out_diag_csv = results_dir / "selection_diagnostics.csv"
    out_md = results_dir / "best_model.md"

    all_results.to_csv(out_csv, index=False)
    selection_diagnostics_view(all_results_with_diagnostics).to_csv(out_diag_csv, index=False)
    pd.DataFrame(best_aic_with_se).to_csv(out_best_csv, index=False)
    write_markdown_summary(all_results_with_diagnostics, best_aic_with_se, out_md)

    print(f"Merged {len(csv_files)} seed files into: {out_csv}")
    print(f"Wrote selection diagnostics: {out_diag_csv}")
    print(f"Wrote best rows with SE: {out_best_csv}")
    print(f"Wrote summary markdown: {out_md}")


if __name__ == "__main__":
    main()
