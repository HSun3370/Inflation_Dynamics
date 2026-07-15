"""Summary statistics and figure for the trimmed monthly inflation dataset.

Reads Aggregate_CPI_inflation_Monthly.pkl (built by update_inflation_data.py)
and the updated workbook, then writes:

- inflation_summary_stats_monthly_trimmed.md : summary table for the trimmed
  monthly effective sample (statistics parallel the quarterly table in
  inflation_summary_stats_trimmed.md).
- cpi_inflation_monthly.png : realized vs. forecasted monthly inflation
  (same style as the quarterly figure produced in DataSummary.ipynb).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
WORKBOOK = max(HERE.glob("Aggregate_CPI_inflation_2*.xlsx"))


def summarize(series: pd.Series) -> dict:
    s = series.dropna().astype(float)

    def ac(x, lag):
        return x.autocorr(lag=lag) if lag < len(x) else np.nan

    return {
        "#Observations": len(s),
        "Mean": s.mean(),
        "Median": s.median(),
        "Std": s.std(ddof=1),
        "Min": s.min(),
        "Max": s.max(),
        "P5": s.quantile(0.05),
        "P25": s.quantile(0.25),
        "P75": s.quantile(0.75),
        "P95": s.quantile(0.95),
        "Skewness": stats.skew(s, bias=False),
        "Excess kurtosis": stats.kurtosis(s, fisher=True, bias=False),
        "AC(1)": ac(s, 1),
        "AC(2)": ac(s, 2),
        "AC(4)": ac(s, 4),
        "AC(12)": ac(s, 12),
    }


def fmt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return str(int(x)) if float(x).is_integer() else f"{x:.4f}"
    return str(x)


def to_md(table: pd.DataFrame) -> str:
    n = table.shape[1]
    return table.map(fmt).to_markdown(tablefmt="pipe", colalign=("left",) + ("right",) * n)


def main() -> None:
    state = pd.read_pickle(HERE / "Aggregate_CPI_inflation_Monthly.pkl")

    rows = {}
    for c in ["Inflation", "SPF", "SPF_shock"]:
        sub = state[c].dropna()
        rows[c] = {
            "Date Start": str(sub.index[0]),
            "Date End": str(sub.index[-1]),
            **summarize(sub),
        }
    summary_table = pd.DataFrame(rows).T[
        ["Date Start", "Date End", "#Observations", "Mean", "Median", "Std",
         "Min", "Max", "P5", "P25", "P75", "P95", "Skewness",
         "Excess kurtosis", "AC(1)", "AC(2)", "AC(4)", "AC(12)"]
    ].T

    md_path = HERE / "inflation_summary_stats_monthly_trimmed.md"
    with open(md_path, "w") as f:
        f.write("```{raw:typst}\n#set page(margin: auto)\n```\n\n")
        f.write("# Inflation Summary Statistics (Monthly, Trimmed)\n\n")
        f.write("## Variables: Inflation, SPF, SPF_shock\n\n")
        f.write(to_md(summary_table))
        f.write("\n")
    print(f"Wrote {md_path.name}")
    print(summary_table.to_string())

    # ---- Figure: realized vs. forecasted monthly inflation (full sample) ----
    xls = pd.ExcelFile(WORKBOOK)
    data_month = xls.parse("Monthly inflation", skiprows=2)
    data_month.columns = ["Year", "Month", "Price index", "Inflation",
                          "Forecasted inflation", "Inflation shock"]
    data_month.index = pd.to_datetime(data_month[["Year", "Month"]].assign(day=1))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data_month.index, data_month["Inflation"],
            label="Inflation", color="#1f77b4", linewidth=1.0)
    ax.plot(data_month.index, data_month["Forecasted inflation"],
            label="Forecasted inflation", color="#d62728", linewidth=1.0, linestyle="--")
    ax.fill_between(data_month.index, data_month["Inflation"],
                    data_month["Forecasted inflation"], alpha=0.15, color="gray")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Monthly CPI Inflation: Realized vs. Forecast")
    ax.set_ylabel("Inflation rate")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "cpi_inflation_monthly.png", dpi=300, bbox_inches="tight")
    print(f"Wrote cpi_inflation_monthly.png (source workbook: {WORKBOOK.name})")


if __name__ == "__main__":
    main()
