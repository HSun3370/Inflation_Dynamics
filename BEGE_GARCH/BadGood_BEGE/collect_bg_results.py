from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


MEAN_TYPES = ["constant", "ARX(1,1)", "ARX(2,1)", "ARX(2,2)"]
REPORT_DROP_COLUMNS = {"message"}
PARAMETER_NAMES = {
    "constant": ["p0", "n0", "rho_p", "rho_n", "phi_p", "phi_n", "sigma_p", "sigma_n"],
    "ARX(1,1)": [
        "c",
        "rho_1",
        "phi_1",
        "p0",
        "n0",
        "rho_p",
        "rho_n",
        "phi_p",
        "phi_n",
        "sigma_p",
        "sigma_n",
    ],
    "ARX(2,1)": [
        "c",
        "rho_1",
        "rho_2",
        "phi_1",
        "p0",
        "n0",
        "rho_p",
        "rho_n",
        "phi_p",
        "phi_n",
        "sigma_p",
        "sigma_n",
    ],
    "ARX(2,2)": [
        "c",
        "rho_1",
        "rho_2",
        "phi_1",
        "phi_2",
        "p0",
        "n0",
        "rho_p",
        "rho_n",
        "phi_p",
        "phi_n",
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


def success_mask(df: pd.DataFrame) -> pd.Series:
    if "success" not in df.columns:
        return pd.Series(False, index=df.index)

    values = df["success"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


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

    return valid


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
    best_rows = best_by_mean(df)
    observed_means = set(df.get("mean_type", pd.Series(dtype=str)).dropna().unique())
    missing_means = [mean_type for mean_type in MEAN_TYPES if mean_type not in observed_means]

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        "# BadGood BEGE Best Model Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Total estimations: `{len(df)}`",
        f"Successful estimations: `{int(success_mask(df).sum())}`",
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

    append_best_table(lines, "Best by Mean Type (Log-Likelihood)", best_rows)
    append_parameter_tables(lines, best_rows)

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def cleaned_csv_view(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if col in REPORT_DROP_COLUMNS or col.startswith("se_")]
    return df.drop(columns=drop_cols, errors="ignore")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "output" / "raw"
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("draw_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No per-seed CSV files found in: {raw_dir}")

    frames = [pd.read_csv(p) for p in csv_files]
    all_results = pd.concat(frames, ignore_index=True)

    key_cols = ["seed", "draw", "mean_type"]
    if set(key_cols).issubset(all_results.columns):
        all_results = all_results.drop_duplicates(subset=key_cols, keep="last")

    all_results = all_results.sort_values(["mean_type", "seed", "draw"]).reset_index(drop=True)

    out_csv = results_dir / "all_estimations.csv"
    out_md = results_dir / "best_model.md"

    cleaned_csv_view(all_results).to_csv(out_csv, index=False)
    write_markdown_summary(all_results, out_md)

    print(f"Merged {len(csv_files)} seed files into: {out_csv}")
    print(f"Wrote summary markdown: {out_md}")


if __name__ == "__main__":
    main()
