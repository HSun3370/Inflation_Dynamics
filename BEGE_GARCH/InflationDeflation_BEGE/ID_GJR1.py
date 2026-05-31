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

from BEGE_GARCH.BEGE_GARCH import ID_GARCH


MEAN_PARAM_NAMES = {
    "constant": [],
    "ARX(1,1)": ["c", "rho_1", "phi_1"],
    "ARX(2,1)": ["c", "rho_1", "rho_2", "phi_1"],
    "ARX(2,2)": ["c", "rho_1", "rho_2", "phi_1", "phi_2"],
}
VOL_PARAM_NAMES = ["p0", "n0", "rho_p", "rho_n", "phi_p_plus", "phi_n_minus", "sigma_p", "sigma_n"]
MEAN_TYPES = ["constant", "ARX(1,1)", "ARX(2,1)", "ARX(2,2)"]


def load_effective_sample(project_root: Path) -> pd.DataFrame:
    data_path = project_root / "DataSummary" / "Aggregate_CPI_inflation_Quarterly.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing effective-sample file: {data_path}")

    df = pd.read_pickle(data_path)
    required = ["Inflation", "SPF", "SPF_shock", "Inflation_lag_1", "Inflation_lag_2", "SPF_lag_1"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Effective-sample file is missing required columns: {missing}")
    return df


def build_model_specs(df: pd.DataFrame) -> list[dict]:
    inflation = df["Inflation"].to_numpy(dtype=float)
    spf = df["SPF"].to_numpy(dtype=float)
    infl_lag1 = df["Inflation_lag_1"].to_numpy(dtype=float)
    infl_lag2 = df["Inflation_lag_2"].to_numpy(dtype=float)
    spf_lag1 = df["SPF_lag_1"].to_numpy(dtype=float)

    return [
        {
            "mean_type": "constant",
            "Y": df["SPF_shock"].to_numpy(dtype=float),
            "X": None,
        },
        {
            "mean_type": "ARX(1,1)",
            "Y": inflation,
            "X": {"SPF": spf, "Inflation_lag_1": infl_lag1},
        },
        {
            "mean_type": "ARX(2,1)",
            "Y": inflation,
            "X": {"SPF": spf, "Inflation_lag_1": infl_lag1, "Inflation_lag_2": infl_lag2},
        },
        {
            "mean_type": "ARX(2,2)",
            "Y": inflation,
            "X": {
                "SPF": spf,
                "Inflation_lag_1": infl_lag1,
                "Inflation_lag_2": infl_lag2,
                "SPF_lag_1": spf_lag1,
            },
        },
    ]


def result_to_row(result: dict, mean_type: str, seed: int, draw: int, random_state: int) -> dict:
    loglik = float(result["loglik"])
    aic = float(result["AIC"])
    bic = float(result["BIC"])
    optimizer_success = bool(getattr(result.get("opt"), "success", True))
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "draw": int(draw),
        "random_state": int(random_state),
        "mean_type": mean_type,
        "success": bool(np.isfinite(loglik) and np.isfinite(aic) and np.isfinite(bic)),
        "optimizer_success": optimizer_success,
        "status": int(getattr(result.get("opt"), "status", 0)),
        "message": str(getattr(result.get("opt"), "message", "")),
        "loglik": loglik,
        "AIC": aic,
        "BIC": bic,
    }

    names = MEAN_PARAM_NAMES[mean_type] + VOL_PARAM_NAMES
    params = np.asarray(result.get("params", []), dtype=float)
    ses = np.asarray(result.get("se", []), dtype=float)

    for name, value in zip(names, params):
        row[f"param_{name}"] = float(value)
    for name, value in zip(names, ses):
        row[f"se_{name}"] = float(value)
    return row


def failure_row(mean_type: str, seed: int, draw: int, random_state: int, exc: Exception) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "draw": int(draw),
        "random_state": int(random_state),
        "mean_type": mean_type,
        "success": False,
        "optimizer_success": False,
        "status": -999,
        "message": f"{type(exc).__name__}: {exc}",
        "loglik": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
    }


def save_incremental(rows: list[dict], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    key_cols = ["seed", "draw", "mean_type"]
    if set(key_cols).issubset(out.columns):
        out = out.drop_duplicates(subset=key_cols, keep="last")
        out = out.sort_values(["draw", "mean_type"]).reset_index(drop=True)
    out.to_csv(out_file, index=False)


def run_seed(seed: int, n_draws: int, n_starts: int, maxiter: int, tol: float, output_dir: Path | None = None) -> None:
    output_dir = output_dir or (SCRIPT_DIR / "output" / "raw")
    out_file = output_dir / f"draw_{seed:03d}.csv"

    df = load_effective_sample(PROJECT_ROOT)
    if len(df) != 215:
        print(f"[WARN] Effective sample length is {len(df)} (expected 215).", flush=True)
    specs = build_model_specs(df)

    if out_file.exists():
        existing = pd.read_csv(out_file)
        rows = existing.to_dict("records")
        completed = set(zip(existing["mean_type"], existing["draw"]))
        print(f"Resuming seed {seed}; found {len(existing)} saved rows in {out_file}", flush=True)
    else:
        rows = []
        completed = set()
        print(f"Starting seed {seed}; output file is {out_file}", flush=True)

    for draw in range(1, n_draws + 1):
        for spec_index, spec in enumerate(specs, start=1):
            mean_type = spec["mean_type"]
            if (mean_type, draw) in completed:
                continue

            random_state = seed * 100000 + draw * 10 + spec_index
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"seed={seed} draw={draw}/{n_draws} mean={mean_type}",
                flush=True,
            )

            try:
                result = ID_GARCH(
                    Y=spec["Y"],
                    X=spec["X"],
                    mean_type=mean_type,
                    n_starts=n_starts,
                    maxiter=maxiter,
                    tol=tol,
                    random_state=random_state,
                    compute_se=False,
                    print_summary=False,
                )
                row = result_to_row(result, mean_type, seed, draw, random_state)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"saved success mean={mean_type} AIC={row['AIC']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                row = failure_row(mean_type, seed, draw, random_state, exc)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"saved failure mean={mean_type}: {exc}",
                    flush=True,
                )

            rows.append(row)
            completed.add((mean_type, draw))
            save_incremental(rows, out_file)

    print(f"Finished seed {seed}; saved {len(rows)} rows to {out_file}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inflation/Deflation BEGE-GJR random-search estimation")
    parser.add_argument("--id", type=int, default=1, help="Seed id for job-array style runs")
    parser.add_argument("--n-draws", type=int, default=10, help="Number of saved estimation draws per mean type")
    parser.add_argument("--n-starts", type=int, default=10, help="Random starts per saved estimation")
    parser.add_argument("--maxiter", type=int, default=10, help="Max optimizer iterations per start")
    parser.add_argument("--tol", type=float, default=1e-8, help="Optimizer tolerance")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for per-seed raw CSV output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_seed(
        seed=args.id,
        n_draws=args.n_draws,
        n_starts=args.n_starts,
        maxiter=args.maxiter,
        tol=args.tol,
        output_dir=args.output_dir,
    )
