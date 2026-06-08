from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.BEGE_GARCH import BEGE_AsymSharedGJR_MLE
from BEGE_GARCH.bege_batch import run_seed_estimation


SYMMETRIC_PARAM_NAMES = [
    "p0",
    "n0",
    "rho",
    "phi_plus",
    "phi_minus",
    "sigma_p",
    "sigma_n",
]


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
    run_seed_estimation(
        estimator=BEGE_AsymSharedGJR_MLE,
        model_label="Symmetric BEGE",
        script_dir=SCRIPT_DIR,
        project_root=PROJECT_ROOT,
        model_param_names=SYMMETRIC_PARAM_NAMES,
        seed=seed,
        n_draws=n_draws,
        n_starts=n_starts,
        maxiter=maxiter,
        tol=tol,
        include_arx22=include_arx22,
        print_summary=print_summary,
        density_hyperu_method=density_hyperu_method,
        compute_se=compute_se,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symmetric BEGE-GARCH random-search estimation")
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
