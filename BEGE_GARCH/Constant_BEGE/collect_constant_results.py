from datetime import datetime
from pathlib import Path

import pandas as pd


def format_value(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return f"{val:.6f}"


def pick_best_rows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    valid_aic = df.dropna(subset=["AIC"]) if "AIC" in df.columns else pd.DataFrame()
    valid_bic = df.dropna(subset=["BIC"]) if "BIC" in df.columns else pd.DataFrame()

    best_aic = valid_aic.loc[valid_aic["AIC"].idxmin()] if not valid_aic.empty else pd.Series(dtype=object)
    best_bic = valid_bic.loc[valid_bic["BIC"].idxmin()] if not valid_bic.empty else pd.Series(dtype=object)
    return best_aic, best_bic


def write_markdown_summary(df: pd.DataFrame, summary_path: Path) -> None:
    best_aic, best_bic = pick_best_rows(df)

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        "# Constant BEGE Best Model Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Total estimations: `{len(df)}`",
        f"Successful estimations: `{int(df.get('success', pd.Series(dtype=bool)).fillna(False).sum())}`",
        "",
    ]

    if not best_aic.empty:
        lines.extend(
            [
                "## Global Best by AIC",
                "",
                f"- Mean type: `{best_aic['mean_type']}`",
                f"- Seed / draw: `{int(best_aic['seed'])}` / `{int(best_aic['draw'])}`",
                f"- AIC: `{format_value(best_aic['AIC'])}`",
                f"- BIC: `{format_value(best_aic['BIC'])}`",
                f"- LogLik: `{format_value(best_aic['loglik'])}`",
                "",
            ]
        )

    if not best_bic.empty:
        lines.extend(
            [
                "## Global Best by BIC",
                "",
                f"- Mean type: `{best_bic['mean_type']}`",
                f"- Seed / draw: `{int(best_bic['seed'])}` / `{int(best_bic['draw'])}`",
                f"- AIC: `{format_value(best_bic['AIC'])}`",
                f"- BIC: `{format_value(best_bic['BIC'])}`",
                f"- LogLik: `{format_value(best_bic['loglik'])}`",
                "",
            ]
        )

    by_mean = []
    for mean_type, g in df.groupby("mean_type", sort=True):
        g_ok = g.dropna(subset=["AIC"])
        if g_ok.empty:
            continue
        row = g_ok.loc[g_ok["AIC"].idxmin()]
        by_mean.append(row)

    if by_mean:
        lines.extend(
            [
                "## Best by Mean Type (AIC)",
                "",
                "| Mean Type | Seed | Draw | AIC | BIC | LogLik |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in by_mean:
            lines.append(
                f"| {row['mean_type']} | {int(row['seed'])} | {int(row['draw'])} | "
                f"{format_value(row['AIC'])} | {format_value(row['BIC'])} | {format_value(row['loglik'])} |"
            )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


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

    all_results.to_csv(out_csv, index=False)
    write_markdown_summary(all_results, out_md)

    print(f"Merged {len(csv_files)} seed files into: {out_csv}")
    print(f"Wrote summary markdown: {out_md}")


if __name__ == "__main__":
    main()
