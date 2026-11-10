"""Routine update of the aggregate CPI inflation data (quarterly and monthly).

Run from the repository root with the pinned project environment, then
regenerate the summary statistics:

    ~/.conda/envs/inflation_dynamics/bin/python DataSummary/update_inflation_data.py
    ~/.conda/envs/inflation_dynamics/bin/python DataSummary/data_summary.py

Data sources
------------
1. CPIAUCSL (FRED): Consumer Price Index for All Urban Consumers: All Items
   in U.S. City Average, Index 1982-1984=100, Monthly, Seasonally Adjusted.
   https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL
2. Survey of Professional Forecasters (Philadelphia Fed), median GNP/GDP
   price deflator *level* forecasts (Median_PGDP_Level.xlsx).
   https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters

Construction rules (reverse-engineered exactly from the advisor's reference
workbook Aggregate_CPI_inflation_20230513.xls):

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
- DATA DECISION -- unpublished CPIAUCSL months inside the sample are filled
  by geometric (log-linear) interpolation between the nearest published
  months. BLS never published October 2025 (prices were not collected during
  the federal shutdown), so 2025-10 = sqrt(CPI_2025-09 * CPI_2025-11) and
  monthly inflation in 2025-10 and 2025-11 splits the two-month change
  evenly. The raw file in raw/ keeps the month missing; filled months are
  printed and recorded in data_vintage.json ("interpolated_cpi_months").

Reference check
---------------
Before anything is written, the rebuilt series are compared with the
reference workbook on every overlapping date. The run aborts unless
- quarterly and monthly SPF expected inflation match exactly, and
- CPIAUCSL levels, inflation, and shocks match exactly before
  CPI_REVISION_START. BLS re-estimates the last five years of seasonal
  factors every February, so seasonally adjusted CPI from 2019 onward has
  been revised since the 2023-05-13 snapshot; those differences are reported.

Outputs (written to this directory)
-----------------------------------
- raw/CPIAUCSL_YYYYMMDD.csv, raw/median_pgdp_level_YYYYMMDD.xlsx : the source
  files exactly as downloaded (re-run offline with --cpi-csv/--spf-xlsx).
- Aggregate_CPI_inflation_YYYYMMDD.xlsx : two sheets in the reference layout.
- Aggregate_CPI_inflation_Quarterly.pkl / Aggregate_CPI_inflation_Monthly.pkl :
  trimmed estimation datasets (columns Inflation, Inflation_lag_1,
  Inflation_lag_2, SPF, SPF_lag_1, SPF_shock; PeriodIndex; trimmed to rows
  where SPF_t and SPF_{t-1} exist). All estimation code reads these.
- data_vintage.json : vintage tag, workbook name, effective-sample windows and
  sizes, and missing months. Read by data_summary.py and the BEGE runners.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import subprocess
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REQUIREMENTS = HERE.parent / "requirements.txt"

FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
SPF_PGDP_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files/median_pgdp_level.xlsx"
)

REFERENCE_WORKBOOK = HERE / "Aggregate_CPI_inflation_20230513.xls"
CPI_REVISION_START = pd.Period("2019-01", "M")
TOL = 1e-9

QUARTERLY_PKL = "Aggregate_CPI_inflation_Quarterly.pkl"
MONTHLY_PKL = "Aggregate_CPI_inflation_Monthly.pkl"
VINTAGE_JSON = "data_vintage.json"

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


class ReferenceMismatch(RuntimeError):
    pass


def check_environment() -> None:
    """Pickles written with another pandas/numpy (e.g. pandas 3 / numpy 2 on a
    laptop) cannot be read by the pinned Grid environment."""
    pins = dict(re.findall(r"^(pandas|numpy)==(\S+)$", REQUIREMENTS.read_text(), re.M))
    found = {"pandas": pd.__version__, "numpy": np.__version__}
    bad = {k: (found[k], v) for k, v in pins.items() if found[k] != v}
    if bad:
        detail = ", ".join(f"{k} {have} (pinned {want})" for k, (have, want) in bad.items())
        raise SystemExit(
            f"Environment mismatch: {detail}. Run with "
            "~/.conda/envs/inflation_dynamics/bin/python, or pass "
            "--allow-env-mismatch if the pickles do not need to load on the Grid."
        )


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


def parse_cpi_monthly(raw: bytes) -> pd.Series:
    """Monthly CPIAUCSL on a gap-free PeriodIndex; unpublished months are NaN."""
    df = pd.read_csv(io.BytesIO(raw))
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[value_col], errors="coerce")
    s.index = df[date_col].dt.to_period("M")
    s = s.sort_index()
    s = s.reindex(pd.period_range(s.index[0], s.index[-1], freq="M"))
    s.index.name = "Date"
    return s.astype(float)


def interpolate_cpi_gaps(cpi_m: pd.Series) -> tuple[pd.Series, list[dict]]:
    """Fill unpublished months inside the sample by log-linear interpolation
    between the nearest published months (a single missing month becomes the
    geometric mean of its neighbours). Leading/trailing gaps stay NaN."""
    # fillna leaves every published value bit-for-bit unchanged.
    filled = cpi_m.fillna(np.exp(np.log(cpi_m).interpolate(method="linear", limit_area="inside")))
    added = filled[cpi_m.isna() & filled.notna()]
    return filled, [{"month": str(p), "value": float(v)} for p, v in added.items()]


def parse_spf_quarterly(raw: bytes) -> pd.Series:
    """Expected quarterly inflation (%) for target quarter t, from the
    survey taken in t-1: (PGDP3 / PGDP2 - 1) * 100."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        df = pd.read_excel(io.BytesIO(_sanitize_xlsx(raw)))
    df.columns = [str(c).upper() for c in df.columns]
    exp = (df["PGDP3"] / df["PGDP2"] - 1.0) * 100.0
    target = pd.PeriodIndex.from_fields(
        year=df["YEAR"], quarter=df["QUARTER"], freq="Q"
    ) + 1
    s = pd.Series(exp.values, index=target, name="SPF").dropna()
    s.index.name = "Date"
    return s.sort_index()


def build_quarterly(cpi_m: pd.Series, spf_q: pd.Series) -> pd.DataFrame:
    # A quarter enters once its last month is in the CPI file; the price is
    # that month's level (NaN if unpublished), never an earlier month's.
    end_months = cpi_m[cpi_m.index.month % 3 == 0]
    p = pd.Series(end_months.to_numpy(), index=end_months.index.asfreq("Q"))

    df = pd.DataFrame(index=p.index)
    df.index.name = "Date"
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


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(REFERENCE_WORKBOOK)
    cols = ["Year", "Sub", "Price", "Inflation", "SPF", "Shock"]
    rq = xls.parse("Quarterly inflation", skiprows=2)
    rm = xls.parse("Monthly inflation", skiprows=2)
    rq.columns = cols
    rm.columns = cols
    rq.index = pd.PeriodIndex.from_fields(year=rq["Year"], quarter=rq["Sub"], freq="Q")
    rm.index = pd.PeriodIndex.from_fields(year=rm["Year"], month=rm["Sub"], freq="M")
    return rq, rm


def check_against_reference(q: pd.DataFrame, m: pd.DataFrame) -> dict:
    """Compare the rebuilt series with the 2023 reference workbook; raise
    ReferenceMismatch if the construction no longer replicates it."""
    rq, rm = load_reference()
    report, failures = {}, []
    for freq, ref, new in [("quarterly", rq, q), ("monthly", rm, m)]:
        new = new.reindex(ref.index)
        cutoff = CPI_REVISION_START.asfreq("Q") if freq == "quarterly" else CPI_REVISION_START
        pre = ref.index < cutoff
        out = {"reference_window": [str(ref.index[0]), str(ref.index[-1])]}
        for col in ["Price", "Inflation", "SPF", "Shock"]:
            a = ref[col].astype(float)
            b = new[col].astype(float)
            nan_mismatch = int((a.isna() != b.isna()).sum())
            diff = (a - b).abs()
            checked = slice(None) if col == "SPF" else pre
            max_checked = float(np.nan_to_num(diff[checked].max()))
            if nan_mismatch or max_checked > TOL:
                where = "all dates" if col == "SPF" else f"before {cutoff}"
                failures.append(
                    f"{freq} {col}: max |diff| {max_checked:.3e} on {where}, "
                    f"{nan_mismatch} missing-value mismatches"
                )
            entry = {"max_abs_diff_checked": max_checked}
            if col != "SPF":
                revised = diff[~pre]
                entry["max_abs_diff_since_revision_start"] = float(np.nan_to_num(revised.max()))
                entry["n_revised_since_revision_start"] = int((revised > TOL).sum())
            out[col] = entry
        report[freq] = out
    if failures:
        raise ReferenceMismatch(
            "Rebuilt data no longer replicate "
            f"{REFERENCE_WORKBOOK.name}:\n  - " + "\n  - ".join(failures)
        )
    return report


def write_workbook(path: Path, q: pd.DataFrame, m: pd.DataFrame) -> None:
    def sheet_frame(data: pd.DataFrame, columns: list, desc: str) -> pd.DataFrame:
        sub_col = "Month" if "Month" in data.columns else "Quarter"
        rows = [
            ["CPIAUCSL", desc, None, None, None, None],
            [None] * 6,
            columns,
        ]
        for _, r in data.iterrows():
            rows.append(
                [int(r["Year"]), int(r[sub_col]),
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


def sample_info(state: pd.DataFrame, pkl_name: str) -> dict:
    return {
        "pickle": pkl_name,
        "start": str(state.index[0]),
        "end": str(state.index[-1]),
        "nobs": len(state),
        "missing_inflation": [str(p) for p in state.index[state["Inflation"].isna()]],
        "rows_with_missing_values": [str(p) for p in state.index[state.isna().any(axis=1)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--date-tag",
        default=dt.date.today().strftime("%Y%m%d"),
        help="Vintage tag (YYYYMMDD) used in the workbook and raw-file names.",
    )
    parser.add_argument(
        "--cpi-csv", type=Path, default=None,
        help="Use a saved FRED CPIAUCSL csv (e.g. raw/CPIAUCSL_YYYYMMDD.csv) instead of downloading.",
    )
    parser.add_argument(
        "--spf-xlsx", type=Path, default=None,
        help="Use a saved median_pgdp_level.xlsx instead of downloading.",
    )
    parser.add_argument(
        "--allow-env-mismatch", action="store_true",
        help="Skip the pinned pandas/numpy check (pickles may not load on the Grid).",
    )
    args = parser.parse_args()
    vintage = dt.datetime.strptime(args.date_tag, "%Y%m%d").date()
    if not args.allow_env_mismatch:
        check_environment()

    cpi_raw = args.cpi_csv.read_bytes() if args.cpi_csv else _download(FRED_CPI_URL)
    spf_raw = args.spf_xlsx.read_bytes() if args.spf_xlsx else _download(SPF_PGDP_URL)
    cpi_published = parse_cpi_monthly(cpi_raw)
    spf_q = parse_spf_quarterly(spf_raw)
    missing_cpi = [str(p) for p in cpi_published.index[cpi_published.isna()]]
    cpi_m, interpolated = interpolate_cpi_gaps(cpi_published)
    print(f"CPIAUCSL: {cpi_m.index[0]} .. {cpi_m.index[-1]}  ({cpi_published.notna().sum()} published months)")
    print(f"  unpublished CPIAUCSL months: {missing_cpi or 'none'}")
    for item in interpolated:
        print(f"  DATA DECISION: {item['month']} filled by geometric interpolation = {item['value']:.4f}")
    print(f"SPF expected inflation targets: {spf_q.index[0]} .. {spf_q.index[-1]}")

    q = build_quarterly(cpi_m, spf_q)
    m = build_monthly(cpi_m, spf_q)

    try:
        report = check_against_reference(q, m)
    except ReferenceMismatch as err:
        raise SystemExit(f"ABORTED, nothing written. {err}") from None
    print(f"Reference check passed against {REFERENCE_WORKBOOK.name}:")
    for freq, out in report.items():
        print(f"  {freq} {out['reference_window'][0]}..{out['reference_window'][1]}: "
              f"SPF max |diff| {out['SPF']['max_abs_diff_checked']:.1e}; CPI identical before "
              f"{CPI_REVISION_START}; since then {out['Price']['n_revised_since_revision_start']} "
              f"revised levels (max {out['Price']['max_abs_diff_since_revision_start']:.3f} index pts, "
              f"inflation max {out['Inflation']['max_abs_diff_since_revision_start']:.3f} pp)")

    raw_dir = HERE / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / f"CPIAUCSL_{args.date_tag}.csv").write_bytes(cpi_raw)
    (raw_dir / f"median_pgdp_level_{args.date_tag}.xlsx").write_bytes(spf_raw)

    workbook = f"Aggregate_CPI_inflation_{args.date_tag}.xlsx"
    write_workbook(HERE / workbook, q, m)
    print(f"Wrote {workbook}: quarterly {q.index[0]}..{q.index[-1]}, "
          f"monthly {m.index[0]}..{m.index[-1]}")

    quarterly_state = build_state_dataset(q)
    monthly_state = build_state_dataset(m)
    pd.to_pickle(quarterly_state, HERE / QUARTERLY_PKL)
    pd.to_pickle(monthly_state, HERE / MONTHLY_PKL)

    vintage_info = {
        "vintage": vintage.isoformat(),
        "workbook": workbook,
        "reference_workbook": REFERENCE_WORKBOOK.name,
        "raw_files": {
            "cpi": f"raw/CPIAUCSL_{args.date_tag}.csv",
            "spf": f"raw/median_pgdp_level_{args.date_tag}.xlsx",
        },
        "cpi_coverage": [str(cpi_m.index[0]), str(cpi_m.index[-1])],
        "spf_target_coverage": [str(spf_q.index[0]), str(spf_q.index[-1])],
        "missing_cpi_months": missing_cpi,
        "cpi_gap_rule": "geometric (log-linear) interpolation between the nearest published months",
        "interpolated_cpi_months": interpolated,
        "quarterly": sample_info(quarterly_state, QUARTERLY_PKL),
        "monthly": sample_info(monthly_state, MONTHLY_PKL),
        "reference_check": report,
    }
    (HERE / VINTAGE_JSON).write_text(json.dumps(vintage_info, indent=2) + "\n")

    for freq in ["quarterly", "monthly"]:
        info = vintage_info[freq]
        print(f"Wrote {info['pickle']}: {info['start']}..{info['end']} ({info['nobs']} obs); "
              f"rows with missing values: {info['rows_with_missing_values'] or 'none'}")
    print(f"Wrote {VINTAGE_JSON}. Next: python DataSummary/data_summary.py")


if __name__ == "__main__":
    main()
