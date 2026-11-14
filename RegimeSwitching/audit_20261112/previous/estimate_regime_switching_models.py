"""Estimate regime-switching models across all mean-process specifications.

Outputs:
- results/regime_switching_results.csv
- results/regime_switching_parameters.csv
- results/regime_switching_attempts.csv.gz (every optimizer start; not tracked in git)
- results/regime_switching_stability_attempts.csv (final-solution perturbation checks)
- results/regime_switching_results.md
- results/regime_switching_best_model.md

Run modes
- Sequential (default): run every start of every specification, then write the outputs.
- Grid: ``--model-id Rxx --attempts FIRST LAST --raw-dir DIR`` runs one block of
  starts for one specification and pickles it to DIR; ``--collect --raw-dir DIR``
  merges the blocks and writes the outputs. ``submit_regime_switching_grid.sh``
  drives both on the CBS Grid. ``--list-models`` prints the specification ids.
- ``--verify-saved`` orders and locally verifies saved provisional estimates;
  reports retain the provenance of the original search's collection rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
import warnings
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "inflation_dynamics_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning

import MarkovRegression_t

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=EstimationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"MarkovRegression_t")


ROOT = Path(__file__).resolve().parents[1]
RS_DIR = Path(__file__).resolve().parent
OUT_DIR = RS_DIR / "results"
TYPST_PREAMBLE = "```{raw:typst}\n#set page(margin: auto)\n```"
MONTHLY_TYPST_STYLE = (
    '```{raw:typst}\n'
    '#set text(fill: rgb("#808080"))\n'
    '#show text: set text(fill: rgb("#808080"))\n'
    '```\n\n'
)


def style_monthly_report(markdown: str, data_path: Path | None) -> str:
    """De-emphasize monthly PDF reports without changing estimation results."""
    if data_path is not None and "monthly" in data_path.name.lower():
        return markdown.replace(
            TYPST_PREAMBLE, TYPST_PREAMBLE + "\n\n" + MONTHLY_TYPST_STYLE, 1
        )
    return markdown


N_STARTS = 5000
# Raised from 50 at the user's request (2026-11-11): in a 40-start test on the leading quarterly
# specification, more EM iterations brought more starts to the best optimum without longer run times.
EM_ITER = 1000
MLE_MAXITER = 1500
STATIONARITY_MARGIN = 1e-6
# Bekaert et al. specify the perturb-and-refit check, but not these tuning values.
STABILITY_REPS = 20
STABILITY_SCALE = 0.05
STABILITY_LLF_TOL = 1e-4
STABILITY_MAX_ROUNDS = 3
# v2: EM initialization raised from 50 to 1,000 iterations; blocks from v1 searches are not merged.
METHODOLOGY_VERSION = "bekaert-identification-stability-v2-em1000"
REGIME_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]


@dataclass(frozen=True)
class MeanProcessSpec:
    name: str
    endog_col: str
    exog_cols: tuple[str, ...]


@dataclass(frozen=True)
class SwitchingSpec:
    k_regimes: int
    switching_ar: bool
    switching_spf: bool
    switching_distribution: bool = True


@dataclass(frozen=True)
class ModelSpec:
    mean_process: MeanProcessSpec
    error_distribution: str
    switch_spec: SwitchingSpec
    switching_nu: bool = False


def format_bool(v: bool) -> str:
    return "Y" if v else "N"


def _yn_to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().upper() in {"Y", "TRUE", "1"}


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]

    def clean(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("\n", "<br>").replace("|", r"\|")

    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(clean(row[col]) for col in df.columns) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, separator, *body])


def load_effective_sample(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = ROOT / "DataSummary" / "Aggregate_CPI_inflation_Quarterly.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing effective sample file: {path}")

    df = pd.read_pickle(path).copy()
    required = {
        "Inflation",
        "Inflation_lag_1",
        "Inflation_lag_2",
        "SPF",
        "SPF_lag_1",
        "SPF_shock",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # The effective sample itself should be used directly for all estimates.
    return df


def mean_processes() -> list[MeanProcessSpec]:
    return [
        MeanProcessSpec(
            name="Constant",
            endog_col="SPF_shock",
            exog_cols=(),
        ),
        MeanProcessSpec(
            name="ARX(1,1)",
            endog_col="Inflation",
            exog_cols=("Inflation_lag_1", "SPF"),
        ),
        MeanProcessSpec(
            name="ARX(2,1)",
            endog_col="Inflation",
            exog_cols=("Inflation_lag_1", "Inflation_lag_2", "SPF"),
        ),
        MeanProcessSpec(
            name="ARX(2,2)",
            endog_col="Inflation",
            exog_cols=("Inflation_lag_1", "Inflation_lag_2", "SPF", "SPF_lag_1"),
        ),
    ]


def switching_structures() -> list[SwitchingSpec]:
    return [
        SwitchingSpec(k_regimes=2, switching_ar=True, switching_spf=True),
        SwitchingSpec(k_regimes=2, switching_ar=True, switching_spf=False),
        SwitchingSpec(k_regimes=2, switching_ar=False, switching_spf=False),
        SwitchingSpec(k_regimes=3, switching_ar=True, switching_spf=False),
        SwitchingSpec(k_regimes=3, switching_ar=False, switching_spf=False),
    ]


def is_ar_col(col: str) -> bool:
    return col.startswith("Inflation_lag_")


def is_spf_col(col: str) -> bool:
    return col.startswith("SPF")


def exog_switch_mask(exog_cols: tuple[str, ...], switch: SwitchingSpec) -> list[bool]:
    mask: list[bool] = []
    for col in exog_cols:
        sw = (switch.switching_ar and is_ar_col(col)) or (switch.switching_spf and is_spf_col(col))
        mask.append(sw)
    return mask


def feasibility_reason(mean: MeanProcessSpec, switch: SwitchingSpec) -> str | None:
    ar_cols = [c for c in mean.exog_cols if is_ar_col(c)]
    spf_cols = [c for c in mean.exog_cols if is_spf_col(c)]

    if switch.switching_ar and len(ar_cols) == 0:
        return "No AR regressors available for switching AR"
    if switch.switching_spf and len(spf_cols) == 0:
        return "No SPF regressors available for switching SPF"

    return None


def build_model_specs(include_student_t: bool = False) -> tuple[list[ModelSpec], list[dict[str, Any]]]:
    """Build the estimation menu.

    Student-t variants are excluded by default: the canonical reporting uses
    normal residuals only (see README), so estimating t models doubles compute
    without affecting any report. Pass ``include_student_t=True`` (CLI flag
    ``--include-student-t``) to estimate them as a robustness exercise.
    """
    specs: list[ModelSpec] = []
    skipped: list[dict[str, Any]] = []

    for mean in mean_processes():
        for switch in switching_structures():
            reason = feasibility_reason(mean, switch)
            if reason is not None:
                skipped.append(
                    {
                        "mean_process": mean.name,
                        "k_regimes": switch.k_regimes,
                        "switching_ar": format_bool(switch.switching_ar),
                        "switching_spf": format_bool(switch.switching_spf),
                        "reason": reason,
                    }
                )
                continue

            specs.append(
                ModelSpec(
                    mean_process=mean,
                    error_distribution="Normal",
                    switch_spec=switch,
                    switching_nu=False,
                )
            )
            if include_student_t:
                specs.append(
                    ModelSpec(
                        mean_process=mean,
                        error_distribution="Student t",
                        switch_spec=switch,
                        switching_nu=switch.switching_distribution,
                    )
                )

    return specs, skipped


def build_model(df: pd.DataFrame, spec: ModelSpec) -> tuple[Any, list[bool], bool]:
    mean = spec.mean_process
    switch = spec.switch_spec

    endog = df[mean.endog_col]
    exog = df[list(mean.exog_cols)] if mean.exog_cols else None

    switching_exog: bool | list[bool]
    mask: list[bool] = []
    if exog is None:
        switching_exog = False
    else:
        mask = exog_switch_mask(mean.exog_cols, switch)
        switching_exog = mask

    has_arx_intercept = mean.name != "Constant"
    kwargs = dict(
        endog=endog,
        exog=exog,
        k_regimes=switch.k_regimes,
        trend="c" if has_arx_intercept else "n",
        switching_trend=(switch.switching_ar if has_arx_intercept else False),
        switching_exog=switching_exog,
        switching_variance=switch.switching_distribution,
    )

    if spec.error_distribution == "Normal":
        model = sm.tsa.MarkovRegression(**kwargs)
    else:
        model = MarkovRegression_t.MarkovRegression_t(
            **kwargs,
            switching_nu=spec.switching_nu,
        )

    return model, mask, has_arx_intercept


def _safe_get(d: dict[str, Any], key: str) -> Any:
    return d.get(key) if isinstance(d, dict) else None


def _is_converged(result: Any) -> bool:
    retvals = getattr(result, "mle_retvals", {})
    conv = _safe_get(retvals, "converged")
    if conv is not None:
        return bool(conv)
    warnflag = _safe_get(retvals, "warnflag")
    if warnflag is not None:
        try:
            return int(warnflag) == 0
        except Exception:
            return bool(not warnflag)
    return True


def _stable_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _format_check(v: bool) -> str:
    return "Y" if bool(v) else "N"


def _display_param_name(
    raw_name: str, exog_cols: tuple[str, ...], switching_cols: list[bool]
) -> str:
    # With trend='c', statsmodels often labels exog as x1, x2, ...; remap
    # to the canonical project regressor names for clearer reporting.
    name = raw_name
    if "[" in raw_name and raw_name.endswith("]"):
        base = raw_name.split("[", 1)[0]
        suffix = "[" + raw_name.split("[", 1)[1]
    else:
        base = raw_name
        suffix = ""

    if len(base) >= 2 and base[0] == "x" and base[1:].isdigit():
        idx = int(base[1:]) - 1
        if 0 <= idx < len(exog_cols):
            col_name = exog_cols[idx]
            is_switching = switching_cols[idx] if idx < len(switching_cols) else True
            return f"{col_name}{suffix}" if is_switching else col_name
    return raw_name


def _estimate_dict(result: Any, mean: MeanProcessSpec, mask: list[bool]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_name, value in zip(result.model.param_names, result.params):
        display_name = _display_param_name(raw_name, mean.exog_cols, mask)
        out[display_name] = float(np.real(value))
    return out


def _value_by_regime(estimates: dict[str, float], base_name: str, regime: int) -> float:
    regime_key = f"{base_name}[{regime}]"
    if regime_key in estimates:
        return float(estimates[regime_key])
    if base_name in estimates:
        return float(estimates[base_name])
    return np.nan


def _variance_ordered_params(model: Any, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel every switching block and both transition axes by shock variance."""
    original = np.asarray(params, dtype=float)
    variances = np.asarray([
        original[model.parameters[r, "variance"]][0] for r in range(model.k_regimes)
    ])
    order = np.argsort(variances, kind="stable")
    ordered = original.copy()
    for block in model.parameters.k_parameters:
        if block == "regime_transition":
            continue
        for new_regime, old_regime in enumerate(order):
            ordered[model.parameters[new_regime, block]] = original[
                model.parameters[int(old_regime), block]
            ]
    transition = model.regime_transition_matrix(original)[:, :, 0]
    transition = transition[np.ix_(order, order)]
    ordered[model.parameters["regime_transition"]] = transition[:-1].ravel()
    return ordered, order


def _variance_ordered_result(result: Any) -> tuple[Any, np.ndarray]:
    params, order = _variance_ordered_params(result.model, result.params)
    if np.array_equal(order, np.arange(result.model.k_regimes)):
        return result, order
    ordered = result.model.smooth(params, transformed=True, cov_type="none")
    for attr in ("mle_retvals", "mle_settings"):
        if hasattr(result, attr):
            setattr(ordered, attr, getattr(result, attr))
    if not np.isclose(ordered.llf, result.llf, rtol=0, atol=1e-7):
        raise ValueError("Variance relabeling changed the log likelihood.")
    return ordered, order


def _optimization_coordinates(model: Any, params: np.ndarray) -> np.ndarray:
    """Invert the transition softmax analytically, including near-boundary fits.

    statsmodels 0.14.2 uses a nonlinear root solve for K > 2. Avoid that solve
    when constructing local perturbations. Clipping only makes finite starting
    logits; it is not a constraint on subsequent likelihood evaluations.
    """
    interior = np.asarray(params, dtype=float).copy()
    interior[model.parameters["regime_transition"]] = 1.0 / model.k_regimes
    unconstrained = model.untransform_params(interior)
    transition = np.maximum(model.regime_transition_matrix(params)[:, :, 0], 1e-12)
    transition /= transition.sum(axis=0)
    for origin in range(model.k_regimes):
        unconstrained[model.parameters[origin, "regime_transition"]] = np.log(
            transition[:-1, origin] / transition[-1, origin]
        )
    if not np.all(np.isfinite(unconstrained)):
        raise ValueError("Cannot construct finite optimization coordinates.")
    return unconstrained


def _ar_stationarity_check(
    result: Any, spec: ModelSpec, mask: list[bool]
) -> tuple[bool, str]:
    mean = spec.mean_process
    ar_cols = [c for c in mean.exog_cols if is_ar_col(c)]
    if not ar_cols:
        return True, "no_ar_terms"

    estimates = _estimate_dict(result, mean, mask)
    k_regimes = spec.switch_spec.k_regimes
    bad: list[str] = []

    for regime in range(k_regimes):
        rho1 = _value_by_regime(estimates, "Inflation_lag_1", regime)
        if not np.isfinite(rho1):
            bad.append(f"regime_{regime}_missing_rho1")
            continue

        if len(ar_cols) == 1:
            if abs(rho1) >= 1.0 - STATIONARITY_MARGIN:
                bad.append(f"regime_{regime}_ar1_nonstationary")
            continue

        rho2 = _value_by_regime(estimates, "Inflation_lag_2", regime)
        if not np.isfinite(rho2):
            bad.append(f"regime_{regime}_missing_rho2")
            continue

        stationary = (
            rho1 + rho2 < 1.0 - STATIONARITY_MARGIN
            and rho2 - rho1 < 1.0 - STATIONARITY_MARGIN
            and rho2 > -1.0 + STATIONARITY_MARGIN
        )
        if not stationary:
            bad.append(f"regime_{regime}_ar2_nonstationary")

    if bad:
        return False, ";".join(bad)
    return True, "stationary"


def _distribution_check(result: Any) -> tuple[bool, str]:
    reasons: list[str] = []
    for raw_name, value in zip(result.model.param_names, result.params):
        val = float(np.real(value))
        if raw_name.startswith("sigma2"):
            if not np.isfinite(val) or val <= 0:
                reasons.append(f"{raw_name}_nonpositive")
        if raw_name.startswith("nu"):
            lower = getattr(MarkovRegression_t, "NU_LOWER", 2.05)
            upper = getattr(MarkovRegression_t, "NU_UPPER", 200.0)
            gaussian_cutoff = getattr(MarkovRegression_t, "NU_GAUSSIAN_CUTOFF", 80.0)
            if not np.isfinite(val) or val <= lower + 1e-6 or val >= upper - 1e-6:
                reasons.append(f"{raw_name}_boundary")
            elif val >= gaussian_cutoff:
                reasons.append(f"{raw_name}_gaussian_limit")
    if reasons:
        return False, ";".join(reasons)
    return True, "ok"


def _transition_check(result: Any) -> tuple[bool, str]:
    try:
        mat = np.asarray(result.regime_transition[:, :, 0], dtype=float).T
    except Exception as exc:
        return False, f"transition_unavailable:{exc}"
    if not np.all(np.isfinite(mat)):
        return False, "transition_nonfinite"
    if np.nanmin(mat) < -1e-8 or np.nanmax(mat) > 1.0 + 1e-8:
        return False, "transition_out_of_bounds"
    if not np.allclose(mat.sum(axis=1), 1.0, atol=1e-6):
        return False, "transition_rows_not_stochastic"
    return True, "ok"


def _result_checks(result: Any, spec: ModelSpec, mask: list[bool]) -> dict[str, Any]:
    llf = float(getattr(result, "llf", np.nan))
    converged = _is_converged(result)
    stationary, stationarity_reason = _ar_stationarity_check(result, spec, mask)
    dist_ok, distribution_reason = _distribution_check(result)
    trans_ok, transition_reason = _transition_check(result)
    finite_llf = bool(np.isfinite(llf))
    checks_passed = bool(converged and finite_llf and stationary and dist_ok and trans_ok)
    return {
        "llf": llf,
        "converged": converged,
        "finite_llf": finite_llf,
        "stationary": stationary,
        "stationarity_reason": stationarity_reason,
        "distribution_valid": dist_ok,
        "distribution_reason": distribution_reason,
        "transition_valid": trans_ok,
        "transition_reason": transition_reason,
        "checks_passed": checks_passed,
    }


def _randomized_start_params(model: Any, rng: np.random.Generator) -> np.ndarray:
    params = np.asarray(model.start_params, dtype=float).copy()
    names = list(model.param_names)

    for i, name in enumerate(names):
        value = float(params[i])
        if name.startswith("p["):
            # Keep transition starts valid; EM/MLE can still move them.
            continue
        if name.startswith("sigma2"):
            params[i] = max(1e-6, value * float(np.exp(rng.normal(0.0, 0.45))))
        elif name.startswith("nu"):
            lower = getattr(MarkovRegression_t, "NU_LOWER", 2.05)
            upper = getattr(MarkovRegression_t, "NU_UPPER", 200.0)
            params[i] = float(np.clip(value + rng.normal(0.0, 3.0), lower + 0.1, min(upper - 0.1, 60.0)))
        else:
            params[i] = value + float(rng.normal(0.0, 0.25 * max(1.0, abs(value))))

    return params


def model_label(spec: ModelSpec) -> str:
    switch = spec.switch_spec
    return (
        f"{spec.mean_process.name}_"
        f"{spec.error_distribution.replace(' ', '_').lower()}_"
        f"r{switch.k_regimes}_"
        f"ar{format_bool(switch.switching_ar)}_"
        f"spf{format_bool(switch.switching_spf)}_"
        f"nu{format_bool(spec.switching_nu)}"
    )


def _fit_with_restarts(
    model: Any,
    spec: ModelSpec,
    mask: list[bool],
    seed_base: int,
    model_label: str,
    attempts: range,
) -> tuple[tuple | None, tuple | None, int, int, list[dict[str, Any]]]:
    """Run the given optimizer starts and keep only the two best fits.

    Returns the highest-likelihood start that passed all checks and the
    highest-likelihood start with a finite likelihood, each as
    ``(llf, attempt, tag, result)`` or None (the earliest start wins ties),
    followed by the converged and valid counts and one audit row per start.
    """
    best_valid: tuple | None = None
    best_any: tuple | None = None
    n_converged = 0
    n_valid = 0
    attempt_rows: list[dict[str, Any]] = []

    for attempt in attempts:
        seed = seed_base + 1009 * attempt
        np.random.seed(seed)
        rng = np.random.default_rng(seed + 17)
        method = "lbfgs" if attempt % 2 else "bfgs"
        start_params = None if attempt == 1 else _randomized_start_params(model, rng)

        try:
            result = model.fit(
                start_params=start_params,
                transformed=True,
                method=method,
                maxiter=MLE_MAXITER,
                em_iter=EM_ITER,
                search_reps=0,
                search_iter=10,
                search_scale=1.0,
                cov_type="none",
                disp=False,
            )
            checks = _result_checks(result, spec, mask)
            retvals = getattr(result, "mle_retvals", {})
            tag = f"{method}#{attempt}"
            attempt_rows.append(
                {
                    "model_label": model_label,
                    "attempt": attempt,
                    "method": method,
                    "llf": checks["llf"],
                    "converged": _format_check(checks["converged"]),
                    "stationary": _format_check(checks["stationary"]),
                    "distribution_valid": _format_check(checks["distribution_valid"]),
                    "transition_valid": _format_check(checks["transition_valid"]),
                    "checks_passed": _format_check(checks["checks_passed"]),
                    "iterations": _safe_get(retvals, "iterations"),
                    "warnflag": _safe_get(retvals, "warnflag"),
                    "stationarity_reason": checks["stationarity_reason"],
                    "distribution_reason": checks["distribution_reason"],
                    "transition_reason": checks["transition_reason"],
                    "error": "",
                }
            )
            if checks["finite_llf"]:
                llf = checks["llf"]
                n_converged += bool(checks["converged"])
                if best_any is None or llf > best_any[0]:
                    best_any = (llf, attempt, tag, result)
                if checks["checks_passed"]:
                    n_valid += 1
                    if best_valid is None or llf > best_valid[0]:
                        best_valid = (llf, attempt, tag, result)
        except Exception as exc:
            attempt_rows.append(
                {
                    "model_label": model_label,
                    "attempt": attempt,
                    "method": method,
                    "llf": np.nan,
                    "converged": "N",
                    "stationary": "N",
                    "distribution_valid": "N",
                    "transition_valid": "N",
                    "checks_passed": "N",
                    "iterations": np.nan,
                    "warnflag": np.nan,
                    "stationarity_reason": "fit_failed",
                    "distribution_reason": "fit_failed",
                    "transition_reason": "fit_failed",
                    "error": str(exc),
                }
            )

    return best_valid, best_any, n_converged, n_valid, attempt_rows


def _candidate(
    df: pd.DataFrame,
    spec: ModelSpec,
    mask: list[bool],
    has_arx_intercept: bool,
    best: tuple,
) -> dict[str, Any]:
    """Summary and parameter rows of one fitted start, for selection across start blocks."""
    llf, attempt, tag, result = best
    result, order = _variance_ordered_result(result)
    mean = spec.mean_process
    switch = spec.switch_spec
    label = model_label(spec)
    retvals = getattr(result, "mle_retvals", {})
    selected_checks = _result_checks(result, spec, mask)

    # Search-level fields (None here) are filled in by select_model.
    summary_row: dict[str, Any] = {
        "model_label": label,
        "mean_process": mean.name,
        "has_intercept": format_bool(has_arx_intercept),
        "intercept_switches_by_regime": format_bool(bool(has_arx_intercept and switch.switching_ar)),
        "endog": mean.endog_col,
        "exog_cols": ",".join(mean.exog_cols),
        "error_distribution": spec.error_distribution,
        "k_regimes": switch.k_regimes,
        "switching_ar": format_bool(switch.switching_ar),
        "switching_spf": format_bool(switch.switching_spf),
        "switching_distribution": format_bool(switch.switching_distribution),
        "switching_nu": format_bool(spec.switching_nu),
        "sample_start": str(df.index.min()),
        "sample_end": str(df.index.max()),
        "nobs": int(result.nobs),
        "llf": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "hqic": float(result.hqic),
        # No variance floor is imposed, so a regime variance near zero marks a degenerate optimum.
        "min_sigma2": float(min(
            np.real(v) for n, v in zip(result.model.param_names, result.params) if n.startswith("sigma2")
        )),
        "converged": _is_converged(result),
        "stationary": selected_checks["stationary"],
        "stationarity_reason": selected_checks["stationarity_reason"],
        "distribution_valid": selected_checks["distribution_valid"],
        "distribution_reason": selected_checks["distribution_reason"],
        "transition_valid": selected_checks["transition_valid"],
        "transition_reason": selected_checks["transition_reason"],
        "checks_passed": selected_checks["checks_passed"],
        "selection_status": None,
        "iterations": _safe_get(retvals, "iterations"),
        "warnflag": _safe_get(retvals, "warnflag"),
        "fit_strategy": None,
        "fit_attempts": None,
        "fit_converged_candidates": None,
        "fit_valid_candidates": None,
        "regime_identification": "increasing_innovation_variance",
        "regime_order_from_optimizer": ",".join(str(int(r)) for r in order),
    }

    parameter_rows: list[dict[str, Any]] = []
    for name, val in zip(result.model.param_names, result.params):
        display_name = _display_param_name(name, mean.exog_cols, mask)
        val_real = float(np.real(val))
        if name.startswith("p["):
            ptype = "transition_probability"
        elif name.startswith("sigma2") or name.startswith("nu"):
            ptype = "distribution"
        else:
            ptype = "mean_process"

        parameter_rows.append(
            {
                "model_label": label,
                "mean_process": mean.name,
                "error_distribution": spec.error_distribution,
                "k_regimes": switch.k_regimes,
                "parameter": display_name,
                "parameter_raw": name,
                "parameter_type": ptype,
                "estimate": val_real,
            }
        )

    return {
        "llf": llf,
        "attempt": attempt,
        "tag": tag,
        "summary_row": summary_row,
        "parameter_rows": parameter_rows,
    }


def search_model(df: pd.DataFrame, spec: ModelSpec, attempts: range) -> dict[str, Any]:
    """Run one block of starts for one specification; the record is picklable."""
    model, mask, has_arx_intercept = build_model(df, spec)
    label = model_label(spec)
    best_valid, best_any, n_converged, n_valid, attempt_rows = _fit_with_restarts(
        model,
        spec=spec,
        mask=mask,
        seed_base=_stable_seed(label),
        model_label=label,
        attempts=attempts,
    )
    return {
        "model_label": label,
        "search_methodology": METHODOLOGY_VERSION,
        "attempts": (attempts.start, attempts.stop - 1),
        "n_attempts": len(attempts),
        "n_converged": n_converged,
        "n_valid": n_valid,
        "best_valid": None if best_valid is None else _candidate(df, spec, mask, has_arx_intercept, best_valid),
        "best_any": None if best_any is None else _candidate(df, spec, mask, has_arx_intercept, best_any),
        "attempt_rows": attempt_rows,
    }


def select_model(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Combine the start blocks of one specification and select its estimate.

    The estimate is the highest-likelihood start that passed all checks; only
    when no start passed is the highest finite likelihood kept, flagged as a
    fallback. The earliest start wins ties, so the selection does not depend on
    how the starts were split into blocks.
    """
    if not records:
        raise RuntimeError("No start blocks were collected.")
    attempt_rows = [row for rec in records for row in rec["attempt_rows"]]
    valid = [rec["best_valid"] for rec in records if rec["best_valid"] is not None]
    fallback = [rec["best_any"] for rec in records if rec["best_any"] is not None]
    if valid:
        chosen = max(valid, key=lambda c: (c["llf"], -c["attempt"]))
        selection_status = "passed_checks"
        fit_strategy = chosen["tag"]
    elif fallback:
        chosen = max(fallback, key=lambda c: (c["llf"], -c["attempt"]))
        selection_status = "fallback_highest_llf_no_valid_attempt"
        fit_strategy = f"{chosen['tag']}|{selection_status}"
    else:
        raise RuntimeError("All optimization attempts failed.")

    summary_row = dict(chosen["summary_row"])
    summary_row.update(
        {
            "selection_status": selection_status,
            "fit_strategy": fit_strategy,
            "fit_attempts": sum(rec["n_attempts"] for rec in records),
            "fit_converged_candidates": sum(rec["n_converged"] for rec in records),
            "fit_valid_candidates": sum(rec["n_valid"] for rec in records),
            "search_methodology": records[0].get("search_methodology", "legacy_cutoffs"),
        }
    )
    return summary_row, chosen["parameter_rows"], attempt_rows


def build_results_markdown(results_df: pd.DataFrame) -> str:
    # Keep only valid Normal estimates; fallbacks and Student-t excluded from display.
    valid_df = results_df[
        (results_df["checks_passed"] == True) &
        (results_df["error_distribution"] == "Normal")
    ]
    view = valid_df[
        [
            "mean_process",
            "k_regimes",
            "switching_ar",
            "switching_spf",
            "switching_distribution",
            "llf",
            "aic",
            "bic",
        ]
    ].copy()

    # Ordered reporting: mean model -> #regime -> switching flags.
    mean_order = {"Constant": 0, "ARX(1,1)": 1, "ARX(2,1)": 2, "ARX(2,2)": 3}
    yn_order = {"N": 0, "Y": 1}
    view["_mean_order"] = view["mean_process"].map(mean_order).fillna(999)
    view["_sw_ar_order"] = view["switching_ar"].map(yn_order).fillna(999)
    view["_sw_spf_order"] = view["switching_spf"].map(yn_order).fillna(999)
    view["_sw_var_order"] = view["switching_distribution"].map(yn_order).fillna(999)

    view = view.sort_values(
        [
            "_mean_order",
            "k_regimes",
            "_sw_ar_order",
            "_sw_spf_order",
            "_sw_var_order",
            "aic",
        ],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)

    view = view.rename(
        columns={
            "mean_process": "Mean",
            "k_regimes": "K",
            "switching_ar": "Sw.AR",
            "switching_spf": "Sw.SPF",
            "switching_distribution": "Sw.Dist",
            "llf": "LogLik",
            "aic": "AIC",
            "bic": "BIC",
        }
    )
    view = view.drop(
        columns=[
            "_mean_order",
            "_sw_ar_order",
            "_sw_spf_order",
            "_sw_var_order",
        ]
    )

    for c in ["LogLik", "AIC", "BIC"]:
        view[c] = view[c].map(lambda x: f"{x:.4f}")

    n_starts = int(results_df["fit_attempts"].max())
    intro = (
        f"Each model is selected from {n_starts:,} starts; only estimates that passed all "
        "checks (optimizer convergence, AR stationarity, positive finite variances, "
        "valid transition probabilities, and final-solution stability) are reported. "
        "Regimes are labeled by increasing innovation variance. `Sw.Dist` reflects "
        "whether $\\sigma^2$ is regime-specific."
    )
    if "verification_methodology" not in results_df:
        intro = (
            f"These saved estimates were selected from {n_starts:,} starts under the "
            "previous collection rules. Regimes are relabeled by increasing innovation "
            "variance. Final-solution stability has not yet been verified. "
            "`Sw.Dist` reflects whether $\\sigma^2$ is regime-specific."
        )
    if "search_methodology" in results_df and (results_df["search_methodology"] == "legacy_cutoffs").any():
        intro += (
            "\n\nThe initial search used the former transition and variance cutoffs. "
            "A fresh 5,000-start search is required to rank all candidates under the "
            "updated collection rules; verification of saved estimates checks their "
            "local stability only."
        )
    return TYPST_PREAMBLE + "\n\n" + intro + "\n\n" + _markdown_table(view) + _unverified_markdown(results_df)


def _unverified_markdown(results_df: pd.DataFrame) -> str:
    """List the Normal specifications that have no verified estimate, with the reason."""
    if "verification_methodology" not in results_df:
        return ""
    failed = results_df[
        ~results_df["checks_passed"].astype(bool) & (results_df["error_distribution"] == "Normal")
    ]
    if failed.empty:
        return ""
    mean_order = {"Constant": 0, "ARX(1,1)": 1, "ARX(2,1)": 2, "ARX(2,2)": 3}
    failed = failed.assign(_order=failed["mean_process"].map(mean_order).fillna(999)).sort_values(
        ["_order", "k_regimes", "switching_ar", "switching_spf"]
    )
    reasons = {
        "not_verified": "Not all perturbation refits returned",
        "likelihood_not_reproducible": "Likelihood not reproduced by its parameters",
        "ineligible_before_verification": "Failed the model checks",
    }
    rows = []
    for _, r in failed.iterrows():
        verified_batch = r["stability_status"] == "not_verified"
        rows.append({
            "Mean": r["mean_process"], "K": r["k_regimes"], "Sw.AR": r["switching_ar"],
            "Sw.SPF": r["switching_spf"], "LogLik": f"{float(r['llf']):.4f}",
            "Min $\\sigma^2$": f"{float(r['min_sigma2']):.2e}",
            "Refits returned": (f"{int(r['stability_returns'])}/{int(r['stability_reps'])}"
                                if verified_batch else "--"),
            "Reason": reasons.get(r["stability_status"], str(r["stability_status"])),
        })
    return (
        "\n\nThe following specifications have no verified estimate and are excluded from the "
        "table above and from model selection. LogLik and the smallest regime variance "
        "$\\min_k\\sigma_k^2$ refer to the rejected estimate. No variance floor is imposed, so the "
        "Gaussian likelihood is unbounded as a regime variance approaches zero; an estimate with "
        "$\\min_k\\sigma_k^2$ near zero is such a degenerate optimum.\n\n"
        + _markdown_table(pd.DataFrame(rows))
    )


def _fmt4(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.4f}"


def _parameter_label(base_name: str) -> str:
    labels = {
        "const": "$c$",
        "Inflation_lag_1": "$\\rho_1$",
        "Inflation_lag_2": "$\\rho_2$",
        "SPF": "$\\phi_1$",
        "SPF_lag_1": "$\\phi_2$",
        "sigma2": "$\\sigma^2$",
        "nu": "$\\nu$",
    }
    return labels.get(base_name, f"`{base_name}`")


def _inflation_term(col: str) -> str:
    terms = {
        "Inflation_lag_1": "\\pi_t",
        "Inflation_lag_2": "\\pi_{t-1}",
        "SPF": "SPF_t",
        "SPF_lag_1": "SPF_{t-1}",
    }
    return terms.get(col, col)


def _coefficient_symbol(col: str, switches: bool) -> str:
    base = {
        "Inflation_lag_1": "\\rho_1",
        "Inflation_lag_2": "\\rho_2",
        "SPF": "\\phi_1",
        "SPF_lag_1": "\\phi_2",
    }.get(col, col)
    if not switches:
        return base
    if "_" in base:
        root, sub = base.split("_", 1)
        return f"{root}_{{{sub},s_t}}"
    return f"{base}_{{s_t}}"


def _mean_equation_markdown(best_row: pd.Series) -> str:
    if best_row["mean_process"] == "Constant":
        return "$$\n\\pi_{t+1} = SPF_t + u_{t+1}\n$$"

    exog_cols = [c for c in str(best_row["exog_cols"]).split(",") if c]
    terms: list[str] = []
    intercept = "c_{s_t}" if best_row["intercept_switches_by_regime"] == "Y" else "c"
    terms.append(intercept)
    for col in exog_cols:
        switches = (best_row["switching_ar"] == "Y" and is_ar_col(col)) or (
            best_row["switching_spf"] == "Y" and is_spf_col(col)
        )
        terms.append(f"{_coefficient_symbol(col, switches)}\\,{_inflation_term(col)}")

    rhs = " + ".join(terms)
    return "$$\n\\pi_{t+1} = " + rhs + " + u_{t+1}\n$$"


def _parallel_mean_parameter_table(best_row: pd.Series, best_params: pd.DataFrame) -> pd.DataFrame:
    k_regimes = int(best_row["k_regimes"])
    exog_cols = [c for c in str(best_row["exog_cols"]).split(",") if c]
    has_intercept = best_row["has_intercept"] == "Y"

    estimates = dict(zip(best_params["parameter"], best_params["estimate"]))
    rows: list[dict[str, Any]] = []

    entries: list[tuple[str, str, bool]] = []
    if has_intercept:
        entries.append(("const", "$c$", best_row["intercept_switches_by_regime"] == "Y"))
    for col in exog_cols:
        switches = (best_row["switching_ar"] == "Y" and is_ar_col(col)) or (
            best_row["switching_spf"] == "Y" and is_spf_col(col)
        )
        entries.append((col, _parameter_label(col), switches))

    for base_name, label, switches in entries:
        row: dict[str, Any] = {"Parameter": label, "Switching": format_bool(switches)}
        for r in range(k_regimes):
            row[f"Regime {r}"] = _fmt4(_value_by_regime(estimates, base_name, r))
        rows.append(row)

    return pd.DataFrame(rows)


def _parallel_distribution_parameter_table(
    best_row: pd.Series, best_params: pd.DataFrame
) -> pd.DataFrame:
    k_regimes = int(best_row["k_regimes"])
    estimates = dict(zip(best_params["parameter"], best_params["estimate"]))
    rows: list[dict[str, Any]] = []

    # The model estimates sigma^2; the report shows the shock standard deviation sigma.
    entries = [("sigma2", "$\\sigma$", best_row["switching_distribution"] == "Y")]
    if any(str(p).startswith("nu") for p in best_params["parameter"]):
        entries.append(("nu", "$\\nu$", best_row["switching_nu"] == "Y"))

    for base_name, label, switches in entries:
        row: dict[str, Any] = {"Parameter": label, "Switching": format_bool(switches)}
        for r in range(k_regimes):
            value = _value_by_regime(estimates, base_name, r)
            if base_name == "sigma2":
                value = np.sqrt(value)
            row[f"Regime {r}"] = _fmt4(value)
        rows.append(row)

    return pd.DataFrame(rows)


def _transition_matrix_table(result: Any) -> pd.DataFrame:
    mat = np.asarray(result.regime_transition[:, :, 0], dtype=float)
    # statsmodels uses [to_regime, from_regime]; report in a
    # row-stochastic "from -> to" layout.
    from_to = mat.T
    k = from_to.shape[0]
    cols = ["From \\ To"] + [f"Regime {j}" for j in range(k)] + ["Row Sum"]
    rows: list[list[Any]] = []
    for i in range(k):
        probs = from_to[i].tolist()
        rows.append([f"Regime {i}", *[_fmt4(v) for v in probs], _fmt4(np.sum(from_to[i]))])
    return pd.DataFrame(rows, columns=cols)


def _switching_parts(best_row: pd.Series) -> tuple[str, str]:
    blocks: list[tuple[str, str]] = []
    if best_row["has_intercept"] == "Y" or "Inflation_lag" in str(best_row["exog_cols"]):
        blocks.append(("AR block, including $c$", best_row["switching_ar"]))
    if any(is_spf_col(c) for c in str(best_row["exog_cols"]).split(",") if c):
        blocks.append(("SPF block", best_row["switching_spf"]))
    # σ² and ν treated as a joint shock-distribution block.
    if best_row["error_distribution"] == "Student t":
        blocks.append(("shock distribution ($\\sigma^2$, $\\nu$)", best_row["switching_distribution"]))
    else:
        blocks.append(("$\\sigma^2$", best_row["switching_distribution"]))

    switching = [label for label, flag in blocks if flag == "Y"]
    fixed = [label for label, flag in blocks if flag != "Y"]
    return ", ".join(switching) if switching else "None", ", ".join(fixed) if fixed else "None"


def _switching_block_table(best_row: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if best_row["has_intercept"] == "Y" or "Inflation_lag" in str(best_row["exog_cols"]):
        rows.append({"Block": "AR block, including $c$", "Switching": best_row["switching_ar"]})
    if any(is_spf_col(c) for c in str(best_row["exog_cols"]).split(",") if c):
        rows.append({"Block": "SPF block", "Switching": best_row["switching_spf"]})
    # σ² and ν shown as a joint distribution block; ν is held common across regimes.
    if best_row["error_distribution"] == "Student t":
        rows.append({"Block": "Shock distribution ($\\sigma^2$, $\\nu$)", "Switching": best_row["switching_distribution"]})
    else:
        rows.append({"Block": "Shock distribution ($\\sigma^2$)", "Switching": best_row["switching_distribution"]})
    return pd.DataFrame(rows)


def _save_regime_plot(df: pd.DataFrame, best_row: pd.Series, best_result: Any, out_dir: Path) -> str:
    probs = best_result.smoothed_marginal_probabilities.copy()
    if isinstance(probs, np.ndarray):
        probs = pd.DataFrame(probs, index=df.index)

    regime_path = probs.idxmax(axis=1).astype(int)
    inflation = df["Inflation"].astype(float).reindex(probs.index)
    x_index = inflation.index.to_timestamp(how="end") if isinstance(inflation.index, pd.PeriodIndex) else inflation.index

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_index, inflation.values, color="black", linewidth=1.6, label="Inflation", zorder=3)

    k = int(best_row["k_regimes"])
    unique_regimes = sorted(int(v) for v in pd.unique(regime_path.values))
    for regime in unique_regimes:
        mask = (regime_path.values == regime).astype(int)
        starts: list[int] = []
        ends: list[int] = []
        in_block = False
        for i, val in enumerate(mask):
            if val == 1 and not in_block:
                starts.append(i)
                in_block = True
            elif val == 0 and in_block:
                ends.append(i - 1)
                in_block = False
        if in_block:
            ends.append(len(mask) - 1)

        for j, (s, e) in enumerate(zip(starts, ends)):
            left = x_index[s]
            right = x_index[e]
            ax.axvspan(
                left,
                right,
                color=REGIME_COLORS[regime % len(REGIME_COLORS)],
                alpha=0.22,
                linewidth=0,
                label=(f"Regime {regime}" if j == 0 else None),
                zorder=1,
            )

    ax.set_title(
        f"Inflation with Smoothed Regime Shading: {best_row['model_label']}",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflation")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()

    plot_name = "best_model_regime_classification.png"
    plot_path = out_dir / plot_name
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_name


def _conditional_variance_path(result: Any) -> np.ndarray:
    """Var(u_t | information through t-1) for u_t = pi_t - E_{t-1}[pi_t]: the
    predicted-probability mixture of the regime distributions,
    sum_k xi_k (sigma_k^2 + m_k^2) - (sum_k xi_k m_k)^2."""
    model = result.model
    params = np.asarray(result.params, dtype=float)
    means = np.asarray(model.predict_conditional(params), dtype=float)[:, 0, :]
    probs = np.asarray(result.predicted_marginal_probabilities, dtype=float).T
    variances = np.array([np.atleast_1d(params[model.parameters[k, "variance"]])[0]
                          for k in range(model.k_regimes)])
    if means.shape != probs.shape:
        raise ValueError(f"Regime means {means.shape} and probabilities {probs.shape} do not align")
    mixture_mean = (probs * means).sum(axis=0)
    path = (probs * (variances[:, None] + means**2)).sum(axis=0) - mixture_mean**2
    if not np.all(np.isfinite(path)) or np.any(path <= 0):
        raise ValueError("Conditional variance path is not finite and positive")
    return path


def _variance_quantile_table(result: Any) -> pd.DataFrame:
    q05, q50, q95 = np.quantile(_conditional_variance_path(result), [0.05, 0.50, 0.95])
    return pd.DataFrame([{"Series": "$\\sigma_t^2$", "5%": _fmt4(q05), "Median": _fmt4(q50), "95%": _fmt4(q95)}])


def build_best_model_markdown(
    best_row: pd.Series,
    best_params: pd.DataFrame,
    best_result: Any,
    df: pd.DataFrame,
    out_dir: Path,
) -> str:
    mean_params = best_params[best_params["parameter_type"] == "mean_process"]
    dist_params = best_params[best_params["parameter_type"] == "distribution"]
    mean_regime_table = _parallel_mean_parameter_table(best_row, mean_params)
    dist_regime_table = _parallel_distribution_parameter_table(best_row, dist_params)
    transition_matrix = _transition_matrix_table(best_result)
    plot_name = _save_regime_plot(df, best_row, best_result, out_dir)
    switching_parts, fixed_parts = _switching_parts(best_row)

    summary_table = pd.DataFrame(
        [
            {
                "Mean process": best_row["mean_process"],
                "Regime-switching parts": switching_parts,
                "Non-switching parts": fixed_parts,
                "LogLik": _fmt4(best_row["llf"]),
                "AIC": _fmt4(best_row["aic"]),
                "BIC": _fmt4(best_row["bic"]),
            }
        ]
    )
    switching_table = _switching_block_table(best_row)

    lines = [
        "# Best Regime-Switching Model",
        "",
        "Selected by AIC among the eligible estimates. Regimes are labeled by increasing innovation variance.",
        "",
        "Final-solution stability: " + str(best_row.get("stability_status", "not_checked")) + ".",
        "",
        "## Model Summary",
        _markdown_table(summary_table),
        "",
        "## Empirical Path Quantiles",
        "Quantiles over the effective sample of the conditional variance of the residual "
        "$u_t=\\pi_t-\\mathrm{E}(\\pi_t\\mid\\mathcal{F}_{t-1})$, "
        "$\\sigma_t^2=\\sum_k \\xi_{k,t|t-1}\\,(\\sigma_k^2+m_{k,t}^2)-"
        "\\big(\\sum_k \\xi_{k,t|t-1}\\,m_{k,t}\\big)^2$, where $\\xi_{k,t|t-1}$ is the predicted "
        "probability of regime $k$ and $m_{k,t}$ its conditional mean.",
        "",
        _markdown_table(_variance_quantile_table(best_result)),
        "",
        "## Switching Blocks",
        _markdown_table(switching_table),
        "",
        "## Mean Process",
        _mean_equation_markdown(best_row),
        "",
        "## Mean Parameter Values",
        _markdown_table(mean_regime_table),
        "",
        "## Distribution Parameters",
        _markdown_table(dist_regime_table),
        "",
        "## Transition Probability Matrix",
        _markdown_table(transition_matrix),
        "",
        "## Regime Classification Plot",
        f"![Inflation with Smoothed Predicted Regimes]({plot_name})",
    ]
    if best_row.get("search_methodology", "legacy_cutoffs") == "legacy_cutoffs":
        lines[2:2] = [
            "The initial search used the former transition and variance cutoffs. "
            "The expanded admissible set requires a fresh 5,000-start search.", "",
        ]
    return TYPST_PREAMBLE + "\n\n" + "\n".join(lines)


def _spec_from_row(row: pd.Series) -> ModelSpec:
    mean_lookup = {m.name: m for m in mean_processes()}
    mean_name = str(row["mean_process"])
    if mean_name not in mean_lookup:
        raise ValueError(f"Unknown mean process in saved results: {mean_name}")

    switch = SwitchingSpec(
        k_regimes=int(row["k_regimes"]),
        switching_ar=_yn_to_bool(row["switching_ar"]),
        switching_spf=_yn_to_bool(row["switching_spf"]),
        switching_distribution=_yn_to_bool(row["switching_distribution"]),
    )
    return ModelSpec(
        mean_process=mean_lookup[mean_name],
        error_distribution=str(row["error_distribution"]),
        switch_spec=switch,
        switching_nu=_yn_to_bool(row["switching_nu"]),
    )


def _reconstruct_result_from_saved_params(
    df: pd.DataFrame,
    best_row: pd.Series,
    best_params: pd.DataFrame,
) -> Any:
    spec = _spec_from_row(best_row)
    model, _, _ = build_model(df, spec)
    estimates = dict(zip(best_params["parameter_raw"], best_params["estimate"].astype(float)))
    missing = [name for name in model.param_names if name not in estimates]
    if missing:
        raise ValueError(
            "Saved parameter table does not contain all model parameters: "
            + ", ".join(missing)
        )
    params = np.asarray([estimates[name] for name in model.param_names], dtype=float)
    return model.smooth(params, transformed=True, cov_type="none")


def _check_saved_sample(df: pd.DataFrame, row: pd.Series) -> None:
    """Never validate saved parameters against a different effective sample."""
    actual = (str(df.index.min()), str(df.index.max()), len(df))
    saved = (str(row["sample_start"]), str(row["sample_end"]), int(row["nobs"]))
    if actual != saved:
        raise ValueError(f"Saved sample {saved} differs from effective sample {actual}.")


def _verify_solution(
    result: Any, spec: ModelSpec, mask: list[bool], n_reps: int = STABILITY_REPS,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Bekaert Appendix B: perturb the provisional optimum and maximize again.

    If a better admissible maximum is found, verify that new maximum with a
    fresh batch. A verification batch passes only if every refit converges,
    passes the model checks, and returns to the center's likelihood tolerance.
    """
    if n_reps < 1:
        raise ValueError("Solution verification requires at least one perturbation.")
    center, _ = _variance_ordered_result(result)
    audit: list[dict[str, Any]] = []
    final_batch: list[dict[str, Any]] = []
    verified = False
    label = model_label(spec)
    initial_llf = float(center.llf)
    for round_no in range(1, STABILITY_MAX_ROUNDS + 1):
        coordinates = _optimization_coordinates(center.model, center.params)
        best = center
        final_batch = []
        for repetition in range(1, n_reps + 1):
            seed = _stable_seed(f"{label}|stability|{round_no}|{repetition}")
            rng = np.random.default_rng(seed)
            start = coordinates + rng.normal(
                size=len(coordinates)
            ) * STABILITY_SCALE * np.maximum(1.0, np.abs(coordinates))
            method = "lbfgs" if repetition % 2 else "bfgs"
            row: dict[str, Any] = {
                "model_label": label, "round": round_no, "perturbation": repetition,
                "seed": seed, "method": method, "center_llf": float(center.llf),
                "llf": np.nan, "llf_difference": np.nan,
                "converged": False, "checks_passed": False, "returned": False,
                "warnflag": np.nan, "iterations": np.nan,
                "max_scaled_parameter_difference": np.nan, "error": "",
            }
            try:
                options = ({"pgtol": 1e-7, "factr": 1e4} if method == "lbfgs"
                           else {"gtol": 1e-7})
                fitted = center.model.fit(
                    start_params=start, transformed=False, method=method,
                    maxiter=MLE_MAXITER, em_iter=0, search_reps=0,
                    cov_type="none", disp=False, **options,
                )
                fitted, _ = _variance_ordered_result(fitted)
                checks = _result_checks(fitted, spec, mask)
                retvals = getattr(fitted, "mle_retvals", {})
                difference = float(fitted.llf - center.llf)
                row.update(
                    llf=float(fitted.llf), llf_difference=difference,
                    converged=checks["converged"], checks_passed=checks["checks_passed"],
                    warnflag=retvals.get("warnflag"), iterations=retvals.get("iterations"),
                    returned=bool(checks["checks_passed"] and abs(difference) <= STABILITY_LLF_TOL),
                    max_scaled_parameter_difference=float(np.max(
                        np.abs(np.asarray(fitted.params) - np.asarray(center.params))
                        / np.maximum(1.0, np.abs(center.params))
                    )),
                    error=";".join(
                        checks[k] for k in ("stationarity_reason", "distribution_reason", "transition_reason")
                        if checks[k] not in {"stationary", "no_ar_terms", "ok"}
                    ),
                )
                if not checks["converged"]:
                    row["error"] = ";".join(filter(None, ["optimizer_not_converged", row["error"]]))
                if checks["checks_passed"] and fitted.llf > best.llf:
                    best = fitted
            except Exception as exc:
                row["error"] = str(exc)
            audit.append(row)
            final_batch.append(row)
        improved = best.llf > center.llf + STABILITY_LLF_TOL
        if improved:
            center = best
        else:
            verified = all(row["returned"] for row in final_batch)
            break
    diagnostics = {
        "stability_verified": verified,
        "stability_reps": n_reps,
        "stability_rounds": round_no,
        "stability_fit_attempts": len(audit),
        "stability_returns": sum(row["returned"] for row in final_batch),
        "stability_initial_llf": initial_llf,
        "stability_llf_improvement": float(center.llf - initial_llf),
        "stability_status": "verified" if verified else "not_verified",
    }
    return center, diagnostics, audit


def _finalize_estimates(
    df: pd.DataFrame, results_df: pd.DataFrame, params_df: pd.DataFrame,
    n_reps: int = STABILITY_REPS, verify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Identify and verify each model after all its Grid blocks are merged.

    A saved likelihood that its saved parameters do not reproduce (to 1e-6) marks a numerically
    degenerate optimum, such as a regime variance near zero; that specification is recorded as
    `likelihood_not_reproducible`, keeps its saved likelihood, and is not verified. If no saved
    likelihood is reproduced, the estimates do not belong to this dataset and collection stops.
    """
    summaries, parameters, audit = [], [], []
    n_unreproduced = 0
    for _, saved in results_df.iterrows():
        _check_saved_sample(df, saved)
        spec = _spec_from_row(saved)
        subset = params_df[params_df["model_label"] == saved["model_label"]]
        result = _reconstruct_result_from_saved_params(df, saved, subset)
        result.mle_retvals = {
            "converged": _yn_to_bool(saved["converged"]),
            "iterations": saved.get("iterations"), "warnflag": saved.get("warnflag"),
        }
        result, order = _variance_ordered_result(result)
        mask = exog_switch_mask(spec.mean_process.exog_cols, spec.switch_spec)
        row = saved.to_dict()
        row.setdefault("regime_order_from_optimizer", ",".join(str(int(r)) for r in order))
        row.setdefault("search_fit_strategy", str(saved["fit_strategy"]))
        basic_checks = _result_checks(result, spec, mask)
        reproduced = bool(np.isclose(result.llf, float(saved["llf"]), atol=1e-6, rtol=0))
        if not reproduced:
            n_unreproduced += 1
            row.update(stability_verified=False, stability_status="likelihood_not_reproducible")
            print(
                f"[VERIFY] {model_label(spec)}: likelihood_not_reproducible, saved llf="
                f"{float(saved['llf']):.4f}, recomputed llf={float(result.llf):.4f}", flush=True,
            )
        elif verify and basic_checks["checks_passed"]:
            result, diagnostics, attempts = _verify_solution(result, spec, mask, n_reps)
            row.update(diagnostics)
            if diagnostics["stability_llf_improvement"] > STABILITY_LLF_TOL:
                row["fit_strategy"] = str(row["search_fit_strategy"]) + "|stability_refit"
            audit.extend(attempts)
            print(
                f"[VERIFY] {model_label(spec)}: {diagnostics['stability_status']}, "
                f"returns={diagnostics['stability_returns']}/{n_reps}, "
                f"llf={result.llf:.4f}", flush=True,
            )
        elif verify:
            row.update(stability_verified=False, stability_status="ineligible_before_verification")
        elif "stability_verified" not in row:
            row.update(stability_verified=False, stability_status="not_checked")
        identified = _candidate(
            df, spec, mask, spec.mean_process.name != "Constant",
            (float(result.llf), 0, str(saved["fit_strategy"]), result),
        )
        keep_saved = {"selection_status", "fit_strategy", "fit_attempts",
                      "fit_converged_candidates", "fit_valid_candidates",
                      "regime_order_from_optimizer"}
        if not reproduced:
            keep_saved |= {"llf", "aic", "bic", "hqic"}
        for key, value in identified["summary_row"].items():
            if key not in keep_saved:
                row[key] = value
        row["regime_order_before_finalization"] = ",".join(str(int(r)) for r in order)
        row.setdefault("search_methodology", "legacy_cutoffs")
        if verify or "verification_methodology" in row:
            row["checks_passed"] = bool(
                row["checks_passed"] and _yn_to_bool(row.get("stability_verified", False))
            )
            row["selection_status"] = (
                "passed_checks_and_stability" if row["checks_passed"] else "failed_final_checks"
            )
            row["verification_methodology"] = METHODOLOGY_VERSION
        summaries.append(row)
        parameters.extend(identified["parameter_rows"])
    if summaries and n_unreproduced == len(summaries):
        raise ValueError("No saved likelihood is reproduced by its saved parameters; "
                         "the saved estimates do not belong to this dataset.")
    return pd.DataFrame(summaries), pd.DataFrame(parameters), pd.DataFrame(audit)


def rebuild_reports_from_csv(
    data_path: Path | None = None, out_dir: Path | None = None
) -> None:
    out_dir = OUT_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_effective_sample(data_path)

    results_csv = out_dir / "regime_switching_results.csv"
    params_csv = out_dir / "regime_switching_parameters.csv"
    md_path = out_dir / "regime_switching_results.md"
    best_md_path = out_dir / "regime_switching_best_model.md"

    results_df = pd.read_csv(results_csv)
    params_df = pd.read_csv(params_csv)
    results_df, params_df, _ = _finalize_estimates(df, results_df, params_df, verify=False)

    eligible_df = results_df[
        results_df["checks_passed"] &
        (results_df["error_distribution"] == "Normal")
    ].copy()
    if eligible_df.empty:
        raise RuntimeError(
            "No Normal-distribution estimate passed all checks; refusing to "
            "report a fallback best model (see README selection policy)."
        )
    best_aic = eligible_df.sort_values("aic", ascending=True).iloc[0]
    best_bic = eligible_df.sort_values("bic", ascending=True).iloc[0]
    best_params = params_df[params_df["model_label"] == best_aic["model_label"]].copy()
    best_result = _reconstruct_result_from_saved_params(df, best_aic, best_params)
    results_df.to_csv(results_csv, index=False)
    params_df.to_csv(params_csv, index=False)

    md_path.write_text(
        style_monthly_report(build_results_markdown(results_df), data_path),
        encoding="utf-8",
    )
    best_md_path.write_text(
        style_monthly_report(
            build_best_model_markdown(best_aic, best_params, best_result, df, out_dir), data_path
        ),
        encoding="utf-8",
    )

    print("Rebuilt markdown reports from saved CSVs:")
    print(f"- {md_path}")
    print(f"- {best_md_path}")


def write_outputs(
    df: pd.DataFrame,
    data_path: Path | None,
    out_dir: Path,
    results_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    stability_reps: int = STABILITY_REPS,
) -> None:
    if not results_rows:
        raise RuntimeError("No model estimated successfully.")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results_rows)
    params_df = pd.DataFrame(parameter_rows)
    attempts_df = pd.DataFrame(attempt_rows)
    if attempts_df.empty:
        attempts_df = pd.DataFrame(columns=["model_label", "attempt"])
    skipped_df = pd.DataFrame(skipped)
    results_df, params_df, stability_df = _finalize_estimates(
        df, results_df, params_df, n_reps=stability_reps,
    )
    stability_csv = out_dir / "regime_switching_stability_attempts.csv"
    stability_df.to_csv(stability_csv, index=False)

    eligible_df = results_df[
        results_df["checks_passed"] &
        (results_df["error_distribution"] == "Normal")
    ].copy()
    failed_checks_df = results_df[~results_df["checks_passed"]].copy()
    if eligible_df.empty:
        results_df.to_csv(out_dir / "regime_switching_results.csv", index=False)
        params_df.to_csv(out_dir / "regime_switching_parameters.csv", index=False)
        attempts_df.to_csv(out_dir / "regime_switching_attempts.csv.gz", index=False)
        failed_checks_df.to_csv(out_dir / "regime_switching_failed_checks.csv", index=False)
        skipped_df.to_csv(out_dir / "regime_switching_skipped_models.csv", index=False)
        results_df[~results_df["converged"]].to_csv(
            out_dir / "regime_switching_nonconverged.csv", index=False,
        )
        (out_dir / "regime_switching_results.md").write_text(
            style_monthly_report(build_results_markdown(results_df), data_path), encoding="utf-8",
        )
        (out_dir / "regime_switching_best_model.md").write_text(
            style_monthly_report(
                TYPST_PREAMBLE + "\n\n# Best Regime-Switching Model\n\n"
                "No Normal-distribution estimate passed all checks and final-solution "
                "stability verification. See the diagnostic CSVs.\n", data_path,
            ), encoding="utf-8",
        )
        print(f"No verified Normal estimate; diagnostic outputs saved in {out_dir}.")
        return

    eligible_df["aic_rank"] = eligible_df["aic"].rank(method="dense")
    eligible_df["bic_rank"] = eligible_df["bic"].rank(method="dense")

    best_aic = eligible_df.sort_values("aic", ascending=True).iloc[0]
    best_params = params_df[params_df["model_label"] == best_aic["model_label"]].copy()
    best_result = _reconstruct_result_from_saved_params(df, best_aic, best_params)

    results_df = results_df.sort_values(
        ["mean_process", "error_distribution", "k_regimes", "aic"]
    ).reset_index(drop=True)
    params_df = params_df.sort_values(["model_label", "parameter_type", "parameter"]).reset_index(drop=True)
    attempts_df = attempts_df.sort_values(["model_label", "attempt"]).reset_index(drop=True)

    results_csv = out_dir / "regime_switching_results.csv"
    params_csv = out_dir / "regime_switching_parameters.csv"
    attempts_csv = out_dir / "regime_switching_attempts.csv.gz"
    md_path = out_dir / "regime_switching_results.md"
    best_md_path = out_dir / "regime_switching_best_model.md"
    skipped_csv = out_dir / "regime_switching_skipped_models.csv"
    nonconv_csv = out_dir / "regime_switching_nonconverged.csv"
    failed_checks_csv = out_dir / "regime_switching_failed_checks.csv"
    err_json = out_dir / "regime_switching_errors.json"

    results_df.to_csv(results_csv, index=False)
    params_df.to_csv(params_csv, index=False)
    attempts_df.to_csv(attempts_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)
    results_df[~results_df["converged"]].to_csv(nonconv_csv, index=False)
    failed_checks_df.to_csv(failed_checks_csv, index=False)

    md_path.write_text(
        style_monthly_report(build_results_markdown(results_df), data_path),
        encoding="utf-8",
    )
    best_md_path.write_text(
        style_monthly_report(
            build_best_model_markdown(best_aic, best_params, best_result, df, out_dir), data_path
        ),
        encoding="utf-8",
    )

    if errors:
        err_json.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    elif err_json.exists():
        err_json.unlink()

    print("\nWrote files:")
    print(f"- {results_csv}")
    print(f"- {params_csv}")
    print(f"- {attempts_csv}")
    print(f"- {stability_csv}")
    print(f"- {md_path}")
    print(f"- {best_md_path}")
    print(f"- {skipped_csv}")
    print(f"- {nonconv_csv}")
    print(f"- {failed_checks_csv}")
    if errors:
        print(f"- {err_json} (contains failed-spec traces)")


def model_entries(include_student_t: bool) -> tuple[list[tuple[str, ModelSpec]], list[dict[str, Any]]]:
    """Specification menu with stable ids R01, R02, ... used by the grid jobs."""
    specs, skipped = build_model_specs(include_student_t=include_student_t)
    return [(f"R{i:02d}", spec) for i, spec in enumerate(specs, start=1)], skipped


def _spec_tag(spec: ModelSpec) -> str:
    switch = spec.switch_spec
    return (
        f"{spec.mean_process.name} | {spec.error_distribution} | "
        f"K={switch.k_regimes} | AR={format_bool(switch.switching_ar)} | "
        f"SPF={format_bool(switch.switching_spf)}"
    )


def _error_row(spec: ModelSpec, exc: Exception) -> dict[str, Any]:
    switch = spec.switch_spec
    return {
        "mean_process": spec.mean_process.name,
        "error_distribution": spec.error_distribution,
        "k_regimes": switch.k_regimes,
        "switching_ar": format_bool(switch.switching_ar),
        "switching_spf": format_bool(switch.switching_spf),
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _select_and_record(
    spec: ModelSpec,
    records: list[dict[str, Any]],
    results_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    tag = _spec_tag(spec)
    try:
        row, params, attempts = select_model(records)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {tag}: {exc}")
        errors.append(_error_row(spec, exc))
        return
    results_rows.append(row)
    parameter_rows.extend(params)
    attempt_rows.extend(attempts)
    print(
        f"[OK] {tag}: llf={row['llf']:.3f}, aic={row['aic']:.3f}, "
        f"bic={row['bic']:.3f}, checks={format_bool(row['checks_passed'])}, "
        f"selection={row['selection_status']}, nobs={row['nobs']}"
    )


def run_model_job(
    df: pd.DataFrame, model_id: str, spec: ModelSpec, attempts: range, raw_dir: Path
) -> None:
    """Grid job: run one block of starts for one specification and pickle the record."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    record = search_model(df, spec, attempts)
    record["model_id"] = model_id
    name = f"{model_id}_{attempts.start:04d}-{attempts.stop - 1:04d}"
    tmp = raw_dir / f".{name}.pkl.tmp"
    pd.to_pickle(record, tmp)
    tmp.replace(raw_dir / f"{name}.pkl")  # atomic: the collector never sees a partial file
    best = record["best_valid"] or record["best_any"]
    best_llf = "NA" if best is None else f"{best['llf']:.4f}"
    print(
        f"{model_id} {record['model_label']} starts {attempts.start}-{attempts.stop - 1}: "
        f"converged={record['n_converged']}, valid={record['n_valid']}, best llf={best_llf}",
        flush=True,
    )


def collect_model_jobs(
    df: pd.DataFrame,
    data_path: Path | None,
    raw_dir: Path,
    out_dir: Path,
    entries: list[tuple[str, ModelSpec]],
    skipped: list[dict[str, Any]],
    n_starts: int,
    allow_missing: bool,
    stability_reps: int = STABILITY_REPS,
) -> None:
    """Merge the start blocks written by run_model_job and write all outputs."""
    expected = set(range(1, n_starts + 1))
    by_model: dict[str, list[dict[str, Any]]] = {}
    incomplete: list[str] = []
    for model_id, spec in entries:
        records = [pd.read_pickle(p) for p in sorted(raw_dir.glob(f"{model_id}_*.pkl"))]
        for rec in records:
            if rec.get("search_methodology") != METHODOLOGY_VERSION:
                raise SystemExit(
                    f"{model_id}: raw blocks use older collection rules; run a fresh "
                    "search, or use --verify-saved to check previously selected estimates."
                )
            if rec["model_label"] != model_label(spec):
                raise SystemExit(f"{raw_dir}: {model_id} holds {rec['model_label']}, expected {model_label(spec)}")
        covered = [a for rec in records for a in range(rec["attempts"][0], rec["attempts"][1] + 1)]
        if len(covered) != len(set(covered)) or set(covered) != expected:
            incomplete.append(f"{model_id} ({len(set(covered) & expected)}/{n_starts} starts)")
        by_model[model_id] = records
    if incomplete and not allow_missing:
        raise SystemExit(f"Incomplete start blocks in {raw_dir}: {incomplete}")
    if incomplete:
        print(f"WARNING: collecting with incomplete start blocks: {incomplete}")

    results_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for model_id, spec in entries:
        _select_and_record(spec, by_model[model_id], results_rows, parameter_rows, attempt_rows, errors)
    write_outputs(df, data_path, out_dir, results_rows, parameter_rows, attempt_rows, skipped, errors,
                  stability_reps=stability_reps)


def verify_saved_estimates(data_path: Path | None, out_dir: Path, n_reps: int) -> None:
    """Verify saved provisional optima; this does not repeat the initial search."""
    df = load_effective_sample(data_path)
    results = pd.read_csv(out_dir / "regime_switching_results.csv")
    params = pd.read_csv(out_dir / "regime_switching_parameters.csv")
    attempts_path = out_dir / "regime_switching_attempts.csv.gz"
    if not attempts_path.exists():
        attempts_path = out_dir / "regime_switching_attempts.csv"
    attempts = pd.read_csv(attempts_path).to_dict("records") if attempts_path.exists() else []
    skipped_path = out_dir / "regime_switching_skipped_models.csv"
    try:
        skipped = pd.read_csv(skipped_path).to_dict("records") if skipped_path.exists() else []
    except pd.errors.EmptyDataError:
        skipped = []
    errors_path = out_dir / "regime_switching_errors.json"
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else []
    write_outputs(df, data_path, out_dir, results.to_dict("records"), params.to_dict("records"),
                  attempts, skipped, errors, stability_reps=n_reps)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=None, help="Effective-sample pickle (default: quarterly).")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Directory for the collected outputs.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild the markdown reports from saved CSVs.")
    parser.add_argument("--verify-saved", action="store_true", help="Identify and locally verify saved estimates; does not rerun the initial search.")
    parser.add_argument("--stability-reps", type=int, default=STABILITY_REPS, help="Local perturbations per final-solution verification round (default: 20).")
    parser.add_argument("--include-student-t", action="store_true", help="Also estimate Student-t variants.")
    parser.add_argument("--n-starts", type=int, default=N_STARTS, help="Optimizer starts per specification.")
    parser.add_argument("--list-models", action="store_true", help="Print specification ids and labels.")
    parser.add_argument("--model-id", default=None, help="Grid job mode: specification id from --list-models.")
    parser.add_argument(
        "--attempts", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None,
        help="Grid job mode: run starts FIRST..LAST (inclusive) of --model-id.",
    )
    parser.add_argument("--collect", action="store_true", help="Merge the start blocks in --raw-dir.")
    parser.add_argument("--allow-missing", action="store_true", help="Collect even if start blocks are missing.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory of per-block pickles.")
    args = parser.parse_args(argv)
    if args.stability_reps < 1:
        parser.error("--stability-reps must be positive")
    if sum(bool(v) for v in (args.report_only, args.verify_saved, args.model_id, args.collect)) > 1:
        parser.error("Choose only one of --report-only, --verify-saved, --model-id, or --collect")
    if (args.model_id or args.collect) and args.raw_dir is None:
        parser.error("--model-id and --collect require --raw-dir")
    if args.model_id and args.attempts is None:
        parser.error("--model-id requires --attempts FIRST LAST")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_path, out_dir = args.data_path, args.output_dir
    entries, skipped = model_entries(args.include_student_t)

    if args.list_models:
        for model_id, spec in entries:
            print(f"{model_id}\t{model_label(spec)}")
        return

    if args.report_only:
        rebuild_reports_from_csv(data_path=data_path, out_dir=out_dir)
        return

    if args.verify_saved:
        verify_saved_estimates(data_path, out_dir, args.stability_reps)
        return

    np.random.seed(20260521)
    df = load_effective_sample(data_path)

    if args.model_id:
        specs_by_id = dict(entries)
        if args.model_id not in specs_by_id:
            raise SystemExit(f"Unknown model id {args.model_id}; see --list-models")
        first, last = args.attempts
        run_model_job(df, args.model_id, specs_by_id[args.model_id], range(first, last + 1), args.raw_dir)
        return

    if args.collect:
        collect_model_jobs(
            df, data_path, args.raw_dir, out_dir, entries, skipped, args.n_starts,
            args.allow_missing, stability_reps=args.stability_reps,
        )
        return

    results_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for _, spec in entries:
        records = [search_model(df, spec, range(1, args.n_starts + 1))]
        _select_and_record(spec, records, results_rows, parameter_rows, attempt_rows, errors)
    write_outputs(df, data_path, out_dir, results_rows, parameter_rows, attempt_rows, skipped, errors,
                  stability_reps=args.stability_reps)


if __name__ == "__main__":
    main()
