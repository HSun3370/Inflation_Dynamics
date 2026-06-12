from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.bege_batch import collect_results


CONSTANT_N_PARAM_NAMES = [
    "p0",
    "n0",
    "rho_p",
    "phi_p_plus",
    "phi_p_minus",
    "sigma_p",
    "sigma_n",
]


def main() -> None:
    collect_results(
        script_dir=SCRIPT_DIR,
        title="Constant-n Full BEGE Best Model Summary",
        model_param_names=CONSTANT_N_PARAM_NAMES,
        model_family="constant_n",
    )


if __name__ == "__main__":
    main()
