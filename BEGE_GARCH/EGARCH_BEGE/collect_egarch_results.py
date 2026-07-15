from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.bege_batch import collect_results


EGARCH_PARAM_NAMES = [
    "omega_p",
    "omega_n",
    "beta_p",
    "beta_n",
    "alpha_p",
    "alpha_n",
    "gamma_p",
    "gamma_n",
    "sigma_p",
    "sigma_n",
]


def main() -> None:
    collect_results(
        script_dir=SCRIPT_DIR,
        title="EGARCH BEGE Best Model Summary",
        model_param_names=EGARCH_PARAM_NAMES,
        model_family="full_egarch",
    )


if __name__ == "__main__":
    main()
