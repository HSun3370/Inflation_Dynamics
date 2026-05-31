from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.bege_batch import collect_results


SYMMETRIC_PARAM_NAMES = [
    "p0",
    "n0",
    "rho",
    "phi_plus",
    "phi_minus",
    "sigma_p",
    "sigma_n",
]


def main() -> None:
    collect_results(
        script_dir=SCRIPT_DIR,
        title="Symmetric BEGE Best Model Summary",
        model_param_names=SYMMETRIC_PARAM_NAMES,
        model_family="symmetric",
    )


if __name__ == "__main__":
    main()
