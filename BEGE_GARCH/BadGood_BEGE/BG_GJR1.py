from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.BEGE_GARCH import BG_GARCH


MEAN_PARAM_NAMES = {
    "constant": [],
    "ARX(1,1)": ["c", "rho_1", "phi_1"],
    "ARX(2,1)": ["c", "rho_1", "rho_2", "phi_1"],
    "ARX(2,2)": ["c", "rho_1", "rho_2", "phi_1", "phi_2"],
}
BG_PARAM_NAMES = ["p0", "n0", "rho_p", "rho_n", "phi_p", "phi_n", "sigma_p", "sigma_n"]


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
        {
            "mean_type": "constant",
            "Y": df["SPF_shock"].to_numpy(dtype=float),
            "X": None,
        },
        {
            "mean_type": "ARX(1,1)",
            "Y": df["Inflation"].to_numpy(dtype=float),
            "X": x_arx11,
        },
        {
            "mean_type": "ARX(2,1)",
            "Y": df["Inflation"].to_numpy(dtype=float),
            "X": x_arx21,
        },
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
        specs.append(
            {
                "mean_type": "ARX(2,2)",
                "Y": df["Inflation"].to_numpy(dtype=float),
                "X": x_arx22,
            }
        )

    return specs


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} requires columns {missing} in the effective sample file.")


def _result_to_row(result: dict, mean_type: str, draw: int, seed: int, random_state: int) -> dict:
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
    names = MEAN_PARAM_NAMES[mean_type] + BG_PARAM_NAMES

    for name, value in zip(names, params):
        row[f"param_{name}"] = float(value)
    for name, value in zip(names, ses):
        row[f"se_{name}"] = float(value)

    return row


def _save_seed_csv(rows: list[dict], path: Path) -> None:
    out = pd.DataFrame(rows).sort_values(["mean_type", "draw"])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def run_seed(
    seed: int,
    n_draws: int,
    n_starts: int,
    maxiter: int,
    tol: float,
    include_arx22: bool,
    print_summary: bool,
    density_hyperu_method: str,
    compute_se: bool,
) -> None:
    output_dir = SCRIPT_DIR / "output"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = load_effective_sample(PROJECT_ROOT)
    if len(df) != 215:
        print(f"[WARN] Effective sample length is {len(df)} (expected 215).")

    _require_columns(df, ["Inflation", "SPF", "SPF_shock"], "BadGood BEGE runner")
    specs = build_model_specs(df, include_arx22=include_arx22)

    total_starts = len(specs) * n_draws * n_starts
    print(
        f"Seed {seed}: estimating {len(specs)} mean process(es), "
        f"{n_draws} draws each, {n_starts} starts per draw "
        f"({total_starts:,} optimizer starts)."
    )

    all_rows: list[dict] = []
    for spec in specs:
        mean_type = spec["mean_type"]
        print(f"Estimating mean_type={mean_type} with {n_draws} draws...")

        for draw in range(1, n_draws + 1):
            rs = draw + seed * 10000
            try:
                result = BG_GARCH(
                    Y=spec["Y"],
                    X=spec["X"],
                    mean_type=mean_type,
                    n_starts=n_starts,
                    maxiter=maxiter,
                    tol=tol,
                    random_state=rs,
                    compute_se=compute_se,
                    density_hyperu_method=density_hyperu_method,
                    print_summary=print_summary,
                )
                row = _result_to_row(result, mean_type=mean_type, draw=draw, seed=seed, random_state=rs)
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
            out_file = raw_dir / f"draw_{seed:03d}.csv"
            _save_seed_csv(all_rows, out_file)

        print(f"Finished mean_type={mean_type}.")

    out_file = raw_dir / f"draw_{seed:03d}.csv"
    _save_seed_csv(all_rows, out_file)
    print(f"Saved completed seed results to: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BadGood BEGE-GARCH random-search estimation")
    parser.add_argument("--id", type=int, default=1, help="Seed id for job-array style runs")
    parser.add_argument("--n-draws", type=int, default=40, help="Number of random draws per mean specification")
    parser.add_argument("--n-starts", type=int, default=25, help="MLE restarts per draw")
    parser.add_argument("--maxiter", type=int, default=800, help="Max optimizer iterations")
    parser.add_argument("--tol", type=float, default=1e-8, help="Optimizer tolerance")
    parser.add_argument(
        "--skip-arx22",
        action="store_true",
        help="Skip ARX(2,2). By default all four mean processes are estimated.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print every optimizer summary. Default is quiet row-level logging.",
    )
    parser.add_argument(
        "--density-hyperu-method",
        choices=["scipy_approx", "scipy_fast", "mpmath"],
        default="scipy_approx",
        help="BEGE density backend. Default uses the stabilized SciPy/high-precision fallback.",
    )
    parser.add_argument(
        "--compute-se",
        action="store_true",
        help="Compute robust numerical standard errors. Default skips SEs for fast search.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_seed(
        seed=args.id,
        n_draws=args.n_draws,
        n_starts=args.n_starts,
        maxiter=args.maxiter,
        tol=args.tol,
        include_arx22=not args.skip_arx22,
        print_summary=args.print_summary,
        density_hyperu_method=args.density_hyperu_method,
        compute_se=args.compute_se,
    )
