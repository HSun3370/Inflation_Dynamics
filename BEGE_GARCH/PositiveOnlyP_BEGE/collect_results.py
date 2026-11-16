from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.bege_batch import collect_results


from BEGE_GARCH.full_restrictions import parameter_names
MODEL_PARAM_NAMES = parameter_names('positive_p')


def main() -> None:
    # SGE dependencies release even after failed jobs: never publish a partial search.
    if "BEGE_EXPECTED_DRAWS" in os.environ:
        from BEGE_GARCH.bege_batch import MEAN_TYPES
        raw = Path(os.environ["BEGE_COLLECT_OUTPUT_DIR"]) / "raw"
        means = MEAN_TYPES if os.environ.get("BEGE_INCLUDE_ARX22", "1") == "1" else MEAN_TYPES[:-1]
        expected_draws = int(os.environ["BEGE_EXPECTED_DRAWS"])
        for seed in range(int(os.environ["START_ID"]), int(os.environ["END_ID"]) + 1):
            rows = pd.read_csv(raw / f"draw_{seed:03d}.csv")
            actual = set(zip(rows["mean_type"], rows["draw"]))
            expected = {(mean, draw) for mean in means for draw in range(1, expected_draws + 1)}
            if actual != expected or len(rows) != len(expected) or not rows["seed"].eq(seed).all():
                raise RuntimeError(f"Incomplete search in seed {seed}; results were not published.")
    collect_results(
        script_dir=SCRIPT_DIR,
        title="Positive-only-p/full-n BEGE Best Model Summary",
        model_param_names=MODEL_PARAM_NAMES,
        report_mean_types=["constant", "ARX(1,1)", "ARX(2,1)", "ARX(2,2)"],
        model_family='positive_p',
    )


if __name__ == "__main__":
    main()
