"""Summary statistics and figures for the current inflation data vintage.

Run after update_inflation_data.py (pinned project environment):

    ~/.conda/envs/inflation_dynamics/bin/python DataSummary/data_summary.py

Reads data_vintage.json, the vintage workbook, and the two trimmed estimation
pickles, then writes:

- inflation_summary_stats_trimmed.md         : quarterly effective sample.
- inflation_summary_stats_monthly_trimmed.md : monthly effective sample.
- cpi_inflation_quarterly.png / cpi_inflation_monthly.png : realized vs.
  forecasted inflation over the full workbook sample.
- README.md : refreshes the blocks between "% BEGIN auto:<name>" and
  "% END auto:<name>" (current vintage facts and all data tables).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
TYPST_PREAMBLE = "```{raw:typst}\n#set page(margin: auto)\n```\n\n"

STAT_ROWS = ["Date Start", "Date End", "#Observations", "Mean", "Median", "Std",
             "Min", "Max", "P5", "P25", "P75", "P95", "Skewness",
             "Excess kurtosis", "AC(1)", "AC(2)", "AC(4)", "AC(12)"]
README_LABELS = {"Inflation": r"Inflation ($\pi$)", "SPF": "$SPF$", "SPF_shock": r"$\pi - SPF$"}


def summarize(series: pd.Series) -> dict:
    # Autocorrelations use the gap-preserving series so that pairs straddling
    # a missing month are dropped rather than treated as adjacent.
    s = series.dropna().astype(float)

    def ac(lag):
        return series.astype(float).autocorr(lag=lag) if lag < len(s) else np.nan

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
        "AC(1)": ac(1),
        "AC(2)": ac(2),
        "AC(4)": ac(4),
        "AC(12)": ac(12),
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


def month_label(period: str) -> str:
    """'1969-02' -> '1969M2' (prose convention); quarters pass through."""
    if "-" not in period:
        return period
    year, month = period.split("-")
    return f"{year}M{int(month)}"


def span(periods: list[str]) -> str:
    """Collapse consecutive period labels into 'A--B' runs."""
    if not periods:
        return "none"
    idx = pd.PeriodIndex([pd.Period(p) for p in periods])
    runs, start = [], idx[0]
    for prev, cur in zip(idx[:-1], idx[1:]):
        if cur != prev + 1:
            runs.append((start, prev))
            start = cur
    runs.append((start, idx[-1]))
    return ", ".join(
        month_label(str(a)) if a == b else f"{month_label(str(a))}--{month_label(str(b))}"
        for a, b in runs
    )


def load_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    cols = ["Price index", "Inflation", "Forecasted inflation", "Inflation shock"]
    q = xls.parse("Quarterly inflation", skiprows=2)
    m = xls.parse("Monthly inflation", skiprows=2)
    q.columns = ["Year", "Quarter"] + cols
    m.columns = ["Year", "Month"] + cols
    q.index = pd.PeriodIndex.from_fields(year=q["Year"], quarter=q["Quarter"], freq="Q")
    m.index = pd.PeriodIndex.from_fields(year=m["Year"], month=m["Month"], freq="M")
    return q, m


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in ["Inflation", "Forecasted inflation"]:
        sub = df[c].dropna()
        rows[c] = {"Date Start": str(sub.index[0]), "Date End": str(sub.index[-1]),
                   "#Observations": len(sub)}
    return pd.DataFrame(rows)


def summary_table(state: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in ["Inflation", "SPF", "SPF_shock"]:
        sub = state[c].dropna()
        rows[c] = {"Date Start": str(sub.index[0]), "Date End": str(sub.index[-1]),
                   **summarize(state[c])}
    return pd.DataFrame(rows).T[STAT_ROWS].T


def md_table_block(title: str, table: pd.DataFrame) -> str:
    return f"```{{table}} {title}\n:align: center\n\n{to_md(table)}\n```"


def decision_note(vintage: dict, freq: str) -> str:
    """Admonition flagging interpolated CPI months that enter this frequency
    (quarterly prices use only the last month of each quarter)."""
    items = vintage.get("interpolated_cpi_months", [])
    if freq == "quarterly":
        items = [i for i in items if pd.Period(i["month"]).month % 3 == 0]
    if not items:
        return ""
    values = ", ".join(f"{month_label(i['month'])} = {i['value']:.4f}" for i in items)
    return (
        "```{important} Data decision: interpolated CPI\n"
        f"BLS did not publish CPIAUCSL for {span([i['month'] for i in items])}. **The missing "
        f"{'level is' if len(items) == 1 else 'levels are'} filled by geometric interpolation between "
        f"the neighbouring published months** ({values}), so the published price change across the "
        f"gap is split evenly. {'It is an estimate' if len(items) == 1 else 'They are estimates'}, "
        "not BLS data; see the data decision in `DataSummary/README.md`.\n"
        "```\n\n"
    )


def write_report(path: Path, title: str, vintage: dict, coverage: pd.DataFrame,
                 summary: pd.DataFrame) -> None:
    with open(path, "w") as f:
        f.write(TYPST_PREAMBLE)
        if title.lower() == "monthly":
            f.write('```{raw:typst}\n#set text(fill: rgb("#808080"))\n#show text: set text(fill: rgb("#808080"))\n```\n\n')
        f.write(f"# Inflation Summary Statistics ({title}, Trimmed)\n\n")
        f.write(f"Data vintage {vintage['vintage']} (`{vintage['workbook']}`).\n\n")
        f.write(decision_note(vintage, title.lower()))
        f.write("## Full workbook sample\n\n")
        f.write(to_md(coverage) + "\n\n")
        f.write("## Variables: Inflation, SPF, SPF_shock\n\n")
        f.write(to_md(summary) + "\n")


def vintage_block(v: dict) -> str:
    q, m = v["quarterly"], v["monthly"]
    ref = v["reference_check"]
    cpi_rev = ref["monthly"]["Price"]["max_abs_diff_since_revision_start"]
    infl_rev = ref["monthly"]["Inflation"]["max_abs_diff_since_revision_start"]
    lines = [
        f"- **Data vintage**: {v['vintage']} (`{v['workbook']}`; source files as downloaded in "
        f"`{v['raw_files']['cpi']}` and `{v['raw_files']['spf']}`).",
        f"- **Source coverage**: CPIAUCSL {month_label(v['cpi_coverage'][0])}--"
        f"{month_label(v['cpi_coverage'][1])}; SPF target quarters "
        f"{v['spf_target_coverage'][0]}--{v['spf_target_coverage'][1]}.",
        f"- **Quarterly effective sample**: **{q['start']}--{q['end']}**, "
        f"**{q['nobs']} observations**"
        + (f"; inflation missing in {span(q['missing_inflation'])}." if q["missing_inflation"]
           else "; no missing values."),
        f"- **Monthly effective sample**: **{month_label(m['start'])}--{month_label(m['end'])}**, "
        f"**{m['nobs']} months**"
        + (f"; inflation is missing in {span(m['missing_inflation'])}, so rows "
           f"{span(m['rows_with_missing_values'])} have a missing value in `Inflation` or its lags."
           if m["missing_inflation"] else "; no missing values."),
        "- **Interpolated CPIAUCSL months** (not published by BLS; filled by geometric "
        "interpolation, see the data decision above): "
        + (", ".join(f"**{month_label(i['month'])} = {i['value']:.4f}**"
                     for i in v.get("interpolated_cpi_months", [])) or "none") + ".",
        f"- **Reference check** against `{v['reference_workbook']}`: SPF expected inflation is "
        f"identical at every overlapping date at both frequencies; CPIAUCSL is identical before "
        f"2019M1 and differs from 2019M1 onward by at most {cpi_rev:.2f} index points "
        f"({infl_rev:.2f} percentage points of monthly inflation).",
    ]
    return "\n".join(lines)


def replace_auto_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(% BEGIN auto:{name}\n).*?(% END auto:{name})", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md is missing the '% BEGIN auto:{name}' / '% END auto:{name}' markers")
    return pattern.sub(lambda mt: mt.group(1) + "\n" + body + "\n\n" + mt.group(2), text)


def plot(df: pd.DataFrame, path: Path, title: str, linewidth: float, index) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(index, df["Inflation"], label="Inflation", color="#1f77b4", linewidth=linewidth)
    ax.plot(index, df["Forecasted inflation"], label="Forecasted inflation",
            color="#d62728", linewidth=linewidth, linestyle="--")
    ax.fill_between(index, df["Inflation"], df["Forecasted inflation"], alpha=0.15, color="gray")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Inflation rate")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    vintage = json.loads((HERE / "data_vintage.json").read_text())
    wq, wm = load_workbook(HERE / vintage["workbook"])
    state_q = pd.read_pickle(HERE / vintage["quarterly"]["pickle"])
    state_m = pd.read_pickle(HERE / vintage["monthly"]["pickle"])

    cov_q, cov_m = coverage_table(wq), coverage_table(wm)
    sum_q, sum_m = summary_table(state_q), summary_table(state_m)

    write_report(HERE / "inflation_summary_stats_trimmed.md", "Quarterly", vintage, cov_q, sum_q)
    write_report(HERE / "inflation_summary_stats_monthly_trimmed.md", "Monthly", vintage, cov_m, sum_m)
    for name, table in [("Quarterly", sum_q), ("Monthly", sum_m)]:
        print(f"== {name} effective sample\n{table.map(fmt).to_string()}\n")

    plot(wq, HERE / "cpi_inflation_quarterly.png", "Quarterly CPI Inflation", 1.5,
         wq.index.to_timestamp(how="end").normalize())
    plot(wm, HERE / "cpi_inflation_monthly.png", "Monthly CPI Inflation: Realized vs. Forecast", 1.0,
         wm.index.to_timestamp())

    text = README.read_text()
    blocks = {
        "vintage": vintage_block(vintage),
        "quarterly-coverage": md_table_block("Quarterly Inflation Statistics", cov_q),
        "quarterly-summary": md_table_block("Inflation statistics", sum_q.rename(columns=README_LABELS)),
        "monthly-coverage": md_table_block("Monthly Inflation Statistics", cov_m),
        "monthly-summary": md_table_block("Monthly inflation statistics", sum_m.rename(columns=README_LABELS)),
    }
    for name, body in blocks.items():
        text = replace_auto_block(text, name, body)
    README.write_text(text)
    print(f"Wrote summary tables, figures, and README.md blocks for vintage {vintage['vintage']}")


if __name__ == "__main__":
    main()
