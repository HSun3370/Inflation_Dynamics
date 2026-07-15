"""Update the aggregate CPI inflation workbook and pickled estimation datasets.

Data sources
------------
1. CPIAUCSL (FRED): Consumer Price Index for All Urban Consumers: All Items
   in U.S. City Average, Index 1982-1984=100, Monthly, Seasonally Adjusted.
   https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL
2. Survey of Professional Forecasters (Philadelphia Fed), median GNP/GDP
   price deflator *level* forecasts (Median_PGDP_Level.xlsx).
   https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters

Construction rules (reverse-engineered exactly from
Aggregate_CPI_inflation_20230513.xls; every historical value matched to
machine precision):

- Quarterly price index  : CPIAUCSL level in the *last month* of the quarter.
- Quarterly inflation    : percent change of the quarterly price index.
- Quarterly SPF expected inflation for quarter t (in %):
      (PGDP3 / PGDP2 - 1) * 100  from the survey taken in quarter t-1,
  i.e. the one-quarter-ahead median deflator forecast over the nowcast.
  First available target quarter: 1969Q1 (from the 1968Q4 survey).
- Monthly inflation      : percent change of monthly CPIAUCSL.
- Monthly SPF expected inflation: the quarterly SPF de-compounded
  geometrically, ((1 + q/100)^(1/3) - 1) * 100, constant within a quarter.
- Inflation shock        : Inflation - SPF expected inflation (both freqs).

Outputs (written to this directory)
-----------------------------------
- Aggregate_CPI_inflation_YYYYMMDD.xlsx : two sheets in the legacy layout.
- Aggregate_CPI_inflation_Monthly.pkl   : trimmed monthly estimation dataset
  (columns Inflation, Inflation_lag_1, Inflation_lag_2, SPF, SPF_lag_1,
  SPF_shock; PeriodIndex 'M'; trimmed to rows where SPF and SPF_lag_1 exist).
- Aggregate_CPI_inflation_Quarterly.pkl : only with --write-quarterly-pkl.
  By default the quarterly pickle is left untouched because the canonical
  quarterly effective sample (1969Q2-2022Q4, 215 obs; see AGENTS.md and
  DataSummary/README.md) underlies all committed quarterly results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
SPF_PGDP_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files/median_pgdp_level.xlsx"
)

QUARTERLY_COLUMNS = [
    "Year",
    "Quarter",
    "Price index (CPIAUCSL)",
    "Inflation (%, quarterly)",
    "Survey of Professional Forecasters Expected inflation from previous quarter (quarterly, %)",
    "Inflation shock (quarterly, %)",
]
MONTHLY_COLUMNS = [
    "Year",
    "Month",
    "Price index (CPIAUCSL)",
    "Inflation (monthly, %)",
    "Survey of Professional Forecasters Expected inflation from previous quarter (monthly, %)",
    "Inflation shock (monthly, %)",
]


def _download(url: str) -> bytes:
    try:
        out = subprocess.run(
            ["curl", "-sSfL", "--max-time", "120", url],
            check=True, capture_output=True,
        ).stdout
        if out:
            return out
    except (OSError, subprocess.CalledProcessError):
        pass
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _sanitize_xlsx(raw: bytes) -> bytes:
    """Replace docProps/core.xml, which the Philly Fed writes with a bare date
    that openpyxl rejects, with a minimal valid one."""
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dcterms:created xsi:type="dcterms:W3CDTF">2000-01-01T00:00:00Z</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2000-01-01T00:00:00Z</dcterms:modified>'
        "</cp:coreProperties>"
    )
    src = zipfile.ZipFile(io.BytesIO(raw))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in src.namelist():
            data = core.encode() if item == "docProps/core.xml" else src.read(item)
            zout.writestr(item, data)
    return out.getvalue()


def load_cpi_monthly(local_csv: Path | None = None) -> pd.Series:
    raw = local_csv.read_bytes() if local_csv else _download(FRED_CPI_URL)
    df = pd.read_csv(io.BytesIO(raw))
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(df[date_col].dt.to_period("M"))[value_col].astype(float)
    s.index.name = "Date"
    return s.sort_index()


def load_spf_quarterly(local_xlsx: Path | None = None) -> pd.Series:
    """Expected quarterly inflation (%) for target quarter t, from the
    survey taken in t-1: (PGDP3 / PGDP2 - 1) * 100."""
    raw = _sanitize_xlsx(
        local_xlsx.read_bytes() if local_xlsx else _download(SPF_PGDP_URL)
    )
    df = pd.read_excel(io.BytesIO(raw))
    df.columns = [str(c).upper() for c in df.columns]
    exp = (df["PGDP3"] / df["PGDP2"] - 1.0) * 100.0
    target = pd.PeriodIndex.from_fields(
        year=df["YEAR"], quarter=df["QUARTER"], freq="Q"
    ) + 1
    s = pd.Series(exp.values, index=target, name="SPF").dropna()
    s.index.name = "Date"
    return s.sort_index()


def build_quarterly(cpi_m: pd.Series, spf_q: pd.Series) -> pd.DataFrame:
    q_index = cpi_m.index.asfreq("Q")
    last_in_quarter = cpi_m.groupby(q_index).last()
    counts = cpi_m.groupby(q_index).size()
    complete = counts[counts == 3].index
    p = last_in_quarter.loc[complete].sort_index()

    df = pd.DataFrame(index=p.index)
    df["Year"] = df.index.year
    df["Quarter"] = df.index.quarter
    df["Price"] = p
    df["Inflation"] = p.pct_change(fill_method=None) * 100.0
    df["SPF"] = spf_q.reindex(df.index)
    df["Shock"] = df["Inflation"] - df["SPF"]
    return df


def build_monthly(cpi_m: pd.Series, spf_q: pd.Series) -> pd.DataFrame:
    spf_m_by_q = ((1.0 + spf_q / 100.0) ** (1.0 / 3.0) - 1.0) * 100.0
    df = pd.DataFrame(index=cpi_m.index)
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Price"] = cpi_m
    df["Inflation"] = cpi_m.pct_change(fill_method=None) * 100.0
    df["SPF"] = spf_m_by_q.reindex(df.index.asfreq("Q")).to_numpy()
    df["Shock"] = df["Inflation"] - df["SPF"]
    return df


def write_workbook(path: Path, q: pd.DataFrame, m: pd.DataFrame) -> None:
    def sheet_frame(data: pd.DataFrame, columns: list, desc: str) -> pd.DataFrame:
        rows = [
            ["CPIAUCSL", desc, None, None, None, None],
            [None] * 6,
            columns,
        ]
        for _, r in data.iterrows():
            rows.append(
                [int(r["Year"]), int(r["Month"] if "Month" in data.columns else r["Quarter"]),
                 r["Price"], r["Inflation"], r["SPF"], r["Shock"]]
            )
        return pd.DataFrame(rows)

    q_desc = (
        "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average, "
        "Index 1982-1984=100, Quarterly, Seasonally Adjusted"
    )
    m_desc = (
        "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average, "
        "Index 1982-1984=100, Monthly, Seasonally Adjusted"
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet_frame(q, QUARTERLY_COLUMNS, q_desc).to_excel(
            writer, sheet_name="Quarterly inflation", header=False, index=False
        )
        sheet_frame(m, MONTHLY_COLUMNS, m_desc).to_excel(
            writer, sheet_name="Monthly inflation", header=False, index=False
        )


def build_state_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Trimmed estimation dataset: keep rows where SPF_t and SPF_{t-1} exist."""
    out = pd.DataFrame(index=df.index)
    out["Inflation"] = df["Inflation"]
    out["Inflation_lag_1"] = df["Inflation"].shift(1)
    out["Inflation_lag_2"] = df["Inflation"].shift(2)
    out["SPF"] = df["SPF"]
    out["SPF_lag_1"] = df["SPF"].shift(1)
    out["SPF_shock"] = df["Inflation"] - df["SPF"]
    out = out[out["SPF"].notna() & out["SPF_lag_1"].notna()]
    out.index.name = "Date"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-quarterly-pkl",
        action="store_true",
        help="Also overwrite Aggregate_CPI_inflation_Quarterly.pkl (changes the "
        "canonical 1969Q2-2022Q4 effective sample; committed quarterly results "
        "are tied to the existing pickle).",
    )
    parser.add_argument(
        "--date-tag",
        default=dt.date.today().strftime("%Y%m%d"),
        help="Date tag used in the output workbook filename.",
    )
    parser.add_argument(
        "--cpi-csv", type=Path, default=None,
        help="Use a pre-downloaded FRED CPIAUCSL csv instead of downloading.",
    )
    parser.add_argument(
        "--spf-xlsx", type=Path, default=None,
        help="Use a pre-downloaded median_pgdp_level.xlsx instead of downloading.",
    )
    args = parser.parse_args()

    cpi_m = load_cpi_monthly(args.cpi_csv)
    spf_q = load_spf_quarterly(args.spf_xlsx)
    print(f"CPIAUCSL: {cpi_m.index[0]} .. {cpi_m.index[-1]}  ({len(cpi_m)} months)")
    print(f"SPF expected inflation targets: {spf_q.index[0]} .. {spf_q.index[-1]}")

    q = build_quarterly(cpi_m, spf_q)
    m = build_monthly(cpi_m, spf_q)

    xlsx_path = HERE / f"Aggregate_CPI_inflation_{args.date_tag}.xlsx"
    write_workbook(xlsx_path, q, m)
    print(f"Wrote {xlsx_path.name}: quarterly {q.index[0]}..{q.index[-1]}, "
          f"monthly {m.index[0]}..{m.index[-1]}")

    monthly_state = build_state_dataset(m.rename(columns={"Shock": "SPF_shock"}))
    monthly_pkl = HERE / "Aggregate_CPI_inflation_Monthly.pkl"
    pd.to_pickle(monthly_state, monthly_pkl)
    print(f"Wrote {monthly_pkl.name}: {monthly_state.index[0]}..{monthly_state.index[-1]} "
          f"({len(monthly_state)} obs)")

    if args.write_quarterly_pkl:
        quarterly_state = build_state_dataset(q)
        pd.to_pickle(quarterly_state, HERE / "Aggregate_CPI_inflation_Quarterly.pkl")
        print(f"Wrote Aggregate_CPI_inflation_Quarterly.pkl: "
              f"{quarterly_state.index[0]}..{quarterly_state.index[-1]} "
              f"({len(quarterly_state)} obs)")
    else:
        print("Quarterly pickle left unchanged (use --write-quarterly-pkl to regenerate).")


if __name__ == "__main__":
    main()
