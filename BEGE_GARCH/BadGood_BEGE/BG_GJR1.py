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
CONSTANT_INIT_SHAPES_PATH = (
    SCRIPT_DIR.parent / "Constant_BEGE" / "results" / "constant_bege_initial_shapes_by_mean.csv"
)


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


def build_model_specs(
    df: pd.DataFrame,
    include_arx22: bool,
    mean_type_filter: str | None = None,
) -> list[dict]:
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

    if mean_type_filter is not None:
        specs = [spec for spec in specs if spec["mean_type"] == mean_type_filter]
        if not specs:
            raise ValueError(f"No model specification available for mean_type={mean_type_filter!r}.")

    return specs


def load_constant_bege_initial_shapes(path: Path = CONSTANT_INIT_SHAPES_PATH) -> dict[str, tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Constant BEGE initial-shape file: {path}")

    df = pd.read_csv(path)
    required = {"mean_type", "p_bar", "n_bar"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Constant BEGE initial-shape file is missing columns: {sorted(missing)}")

    values: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        mean_type = str(row["mean_type"])
        p_bar = float(row["p_bar"])
        n_bar = float(row["n_bar"])
        if not (np.isfinite(p_bar) and p_bar > 0.0 and np.isfinite(n_bar) and n_bar > 0.0):
            raise ValueError(f"Nonpositive or nonfinite Constant BEGE shape for {mean_type}.")
        values[mean_type] = (p_bar, n_bar)
    return values


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} requires columns {missing} in the effective sample file.")


def _result_to_row(
    result: dict,
    mean_type: str,
    draw: int,
    seed: int,
    random_state: int,
    shape_initial_values: tuple[float, float] | None = None,
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

    if shape_initial_values is not None:
        row["recursion_init_p"] = float(shape_initial_values[0])
        row["recursion_init_n"] = float(shape_initial_values[1])

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
    mean_type_filter: str | None = None,
    output_dir: Path | None = None,
    use_shape_initialization: bool = False,
) -> None:
    output_dir = SCRIPT_DIR / "output" if output_dir is None else Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = load_effective_sample(PROJECT_ROOT)
    if len(df) != 215:
        print(f"[WARN] Effective sample length is {len(df)} (expected 215).")

    _require_columns(df, ["Inflation", "SPF", "SPF_shock"], "BadGood BEGE runner")
    specs = build_model_specs(
        df,
        include_arx22=include_arx22,
        mean_type_filter=mean_type_filter,
    )
    constant_initial_shapes = load_constant_bege_initial_shapes() if use_shape_initialization else {}

    total_starts = len(specs) * n_draws * n_starts
    print(
        f"Seed {seed}: estimating {len(specs)} mean process(es), "
        f"{n_draws} draws each, {n_starts} starts per draw "
        f"({total_starts:,} optimizer starts)."
    )

    all_rows: list[dict] = []
    for spec in specs:
        mean_type = spec["mean_type"]
        shape_initial_values: tuple[float, float] | None = None
        print(f"Estimating mean_type={mean_type} with {n_draws} draws...")
        if use_shape_initialization:
            if mean_type not in constant_initial_shapes:
                raise KeyError(f"Missing Constant BEGE initial shapes for mean_type={mean_type}.")
            shape_initial_values = constant_initial_shapes[mean_type]
            print(
                f"Using Constant BEGE recursion initialization for {mean_type}: "
                f"p_init={shape_initial_values[0]:.6f}, n_init={shape_initial_values[1]:.6f}"
            )
        else:
            print(
                f"Using parameter-implied unconditional recursion initialization "
                f"for {mean_type}; fixed p_init/n_init disabled."
            )

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
                    shape_initial_values=shape_initial_values,
                )
                row = _result_to_row(
                    result,
                    mean_type=mean_type,
                    draw=draw,
                    seed=seed,
                    random_state=rs,
                    shape_initial_values=shape_initial_values,
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
                if shape_initial_values is not None:
                    row["recursion_init_p"] = float(shape_initial_values[0])
                    row["recursion_init_n"] = float(shape_initial_values[1])

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
    parser.add_argument(
        "--mean-type",
        choices=["all", "constant", "ARX(1,1)", "ARX(2,1)", "ARX(2,2)"],
        default="all",
        help="Estimate one mean specification. Default estimates all enabled mean processes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for raw output. Default is BadGood_BEGE/output.",
    )
    parser.add_argument(
        "--shape-initialization",
        dest="no_shape_initialization",
        action="store_false",
        help="Use fixed Constant-BEGE p/n initial states.",
    )
    parser.add_argument(
        "--no-shape-initialization",
        dest="no_shape_initialization",
        action="store_true",
        default=True,
        help="Use the parameter-implied unconditional recursion backcast instead of fixed Constant-BEGE p/n initial states.",
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
        mean_type_filter=None if args.mean_type == "all" else args.mean_type,
        output_dir=args.output_dir,
        use_shape_initialization=not args.no_shape_initialization,
    )
