"""Estimate regime-switching models across all mean-process specifications.

Outputs:
- results/regime_switching_results.csv
- results/regime_switching_parameters.csv
- results/regime_switching_results.md
- results/regime_switching_best_model.md
"""

from __future__ import annotations

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
N_STARTS = 50
EM_ITER = 50
MLE_MAXITER = 1500
STATIONARITY_MARGIN = 1e-6
SIGMA2_BOUNDARY_TOL = 1e-6
SIGMA2_RELATIVE_FLOOR = 1e-3
TRANSITION_BOUNDARY_TOL = 1e-4
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
    endog_var = float(np.nanvar(np.asarray(result.model.endog, dtype=float)))
    sigma2_floor = max(SIGMA2_BOUNDARY_TOL, SIGMA2_RELATIVE_FLOOR * endog_var)
    for raw_name, value in zip(result.model.param_names, result.params):
        val = float(np.real(value))
        if raw_name.startswith("sigma2"):
            if not np.isfinite(val) or val <= 0:
                reasons.append(f"{raw_name}_nonpositive")
            elif val <= sigma2_floor:
                reasons.append(f"{raw_name}_boundary")
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
    if (
        np.nanmin(mat) <= TRANSITION_BOUNDARY_TOL
        or np.nanmax(mat) >= 1.0 - TRANSITION_BOUNDARY_TOL
    ):
        return False, "transition_boundary"
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


def _fit_with_restarts(
    model: Any,
    spec: ModelSpec,
    mask: list[bool],
    seed_base: int,
    model_label: str,
) -> tuple[Any, str, int, int, int, str, list[dict[str, Any]]]:
    successful: list[tuple[float, bool, str, Any, dict[str, Any]]] = []
    attempt_rows: list[dict[str, Any]] = []

    for attempt in range(1, N_STARTS + 1):
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
                successful.append((checks["llf"], checks["checks_passed"], tag, result, checks))
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

    if not successful:
        raise RuntimeError("All optimization attempts failed.")

    valid = [item for item in successful if item[1]]
    n_converged = sum(1 for _, _, _, _, checks in successful if checks["converged"])
    n_valid = len(valid)

    if valid:
        valid.sort(key=lambda x: x[0], reverse=True)
        best_llf, _, best_tag, best_result, _ = valid[0]
        return (
            best_result,
            best_tag,
            N_STARTS,
            n_converged,
            n_valid,
            "passed_checks",
            attempt_rows,
        )

    successful.sort(key=lambda x: x[0], reverse=True)
    best_llf, _, best_tag, best_result, _ = successful[0]
    return (
        best_result,
        f"{best_tag}|fallback_highest_llf_no_valid_attempt",
        N_STARTS,
        n_converged,
        n_valid,
        "fallback_highest_llf_no_valid_attempt",
        attempt_rows,
    )

def fit_one(
    df: pd.DataFrame, spec: ModelSpec
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Any]:
    mean = spec.mean_process
    switch = spec.switch_spec
    model, mask, has_arx_intercept = build_model(df, spec)

    model_label = (
        f"{mean.name}_"
        f"{spec.error_distribution.replace(' ', '_').lower()}_"
        f"r{switch.k_regimes}_"
        f"ar{format_bool(switch.switching_ar)}_"
        f"spf{format_bool(switch.switching_spf)}_"
        f"nu{format_bool(spec.switching_nu)}"
    )
    (
        result,
        fit_strategy,
        fit_attempts,
        fit_converged_candidates,
        fit_valid_candidates,
        selection_status,
        attempt_rows,
    ) = _fit_with_restarts(
        model,
        spec=spec,
        mask=mask,
        seed_base=_stable_seed(model_label),
        model_label=model_label,
    )
    retvals = getattr(result, "mle_retvals", {})
    selected_checks = _result_checks(result, spec, mask)

    summary_row: dict[str, Any] = {
        "model_label": model_label,
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
        "converged": _is_converged(result),
        "stationary": selected_checks["stationary"],
        "stationarity_reason": selected_checks["stationarity_reason"],
        "distribution_valid": selected_checks["distribution_valid"],
        "distribution_reason": selected_checks["distribution_reason"],
        "transition_valid": selected_checks["transition_valid"],
        "transition_reason": selected_checks["transition_reason"],
        "checks_passed": selected_checks["checks_passed"],
        "selection_status": selection_status,
        "iterations": _safe_get(retvals, "iterations"),
        "warnflag": _safe_get(retvals, "warnflag"),
        "fit_strategy": fit_strategy,
        "fit_attempts": fit_attempts,
        "fit_converged_candidates": fit_converged_candidates,
        "fit_valid_candidates": fit_valid_candidates,
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
                "model_label": model_label,
                "mean_process": mean.name,
                "error_distribution": spec.error_distribution,
                "k_regimes": switch.k_regimes,
                "parameter": display_name,
                "parameter_raw": name,
                "parameter_type": ptype,
                "estimate": val_real,
            }
        )

    return summary_row, parameter_rows, attempt_rows, result


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

    intro = (
        "Each model is selected from 50 starts; only estimates that passed all "
        "checks (optimizer convergence, AR stationarity, parameter bounds, and "
        "valid transition probabilities) are reported. Specifications where no "
        "start passed all checks are excluded. `Sw.Dist` reflects whether "
        "$\\sigma^2$ is regime-specific."
    )
    return TYPST_PREAMBLE + "\n\n" + intro + "\n\n" + _markdown_table(view)


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

    entries = [("sigma2", "$\\sigma^2$", best_row["switching_distribution"] == "Y")]
    if any(str(p).startswith("nu") for p in best_params["parameter"]):
        entries.append(("nu", "$\\nu$", best_row["switching_nu"] == "Y"))

    for base_name, label, switches in entries:
        row: dict[str, Any] = {"Parameter": label, "Switching": format_bool(switches)}
        for r in range(k_regimes):
            row[f"Regime {r}"] = _fmt4(_value_by_regime(estimates, base_name, r))
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
        "Selected by AIC among estimates that passed optimizer convergence, stationarity, parameter-bound, distribution, and transition checks.",
        "",
        "## Model Summary",
        _markdown_table(summary_table),
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

    md_path.write_text(
        build_results_markdown(results_df),
        encoding="utf-8",
    )
    best_md_path.write_text(
        build_best_model_markdown(best_aic, best_params, best_result, df, out_dir),
        encoding="utf-8",
    )

    print("Rebuilt markdown reports from saved CSVs:")
    print(f"- {md_path}")
    print(f"- {best_md_path}")


def _argv_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(f"Missing value for {flag}")
    return argv[idx + 1]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    data_path_arg = _argv_value(argv, "--data-path")
    out_dir_arg = _argv_value(argv, "--output-dir")
    data_path = Path(data_path_arg) if data_path_arg else None
    out_dir = Path(out_dir_arg) if out_dir_arg else OUT_DIR

    if "--report-only" in argv:
        rebuild_reports_from_csv(data_path=data_path, out_dir=out_dir)
        return

    np.random.seed(20260521)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_effective_sample(data_path)
    specs, skipped = build_model_specs(
        include_student_t="--include-student-t" in argv
    )

    results_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    results_obj_by_label: dict[str, Any] = {}

    for spec in specs:
        switch = spec.switch_spec
        tag = (
            f"{spec.mean_process.name} | {spec.error_distribution} | "
            f"K={switch.k_regimes} | AR={format_bool(switch.switching_ar)} | "
            f"SPF={format_bool(switch.switching_spf)}"
        )
        try:
            row, params, attempts, result_obj = fit_one(df, spec)
            results_rows.append(row)
            parameter_rows.extend(params)
            attempt_rows.extend(attempts)
            results_obj_by_label[row["model_label"]] = result_obj
            print(
                f"[OK] {tag}: llf={row['llf']:.3f}, aic={row['aic']:.3f}, "
                f"bic={row['bic']:.3f}, checks={format_bool(row['checks_passed'])}, "
                f"selection={row['selection_status']}, nobs={row['nobs']}"
            )
        except Exception as exc:  # pragma: no cover
            tb = traceback.format_exc()
            print(f"[ERROR] {tag}: {exc}")
            errors.append(
                {
                    "mean_process": spec.mean_process.name,
                    "error_distribution": spec.error_distribution,
                    "k_regimes": switch.k_regimes,
                    "switching_ar": format_bool(switch.switching_ar),
                    "switching_spf": format_bool(switch.switching_spf),
                    "error": str(exc),
                    "traceback": tb,
                }
            )

    if not results_rows:
        raise RuntimeError("No model estimated successfully.")

    results_df = pd.DataFrame(results_rows)
    params_df = pd.DataFrame(parameter_rows)
    attempts_df = pd.DataFrame(attempt_rows)
    skipped_df = pd.DataFrame(skipped)

    eligible_df = results_df[
        results_df["checks_passed"] &
        (results_df["error_distribution"] == "Normal")
    ].copy()
    failed_checks_df = results_df[~results_df["checks_passed"]].copy()
    if eligible_df.empty:
        raise RuntimeError(
            "No Normal-distribution estimate passed all checks; refusing to "
            "report a fallback best model (see README selection policy)."
        )

    eligible_df["aic_rank"] = eligible_df["aic"].rank(method="dense")
    eligible_df["bic_rank"] = eligible_df["bic"].rank(method="dense")

    best_aic = eligible_df.sort_values("aic", ascending=True).iloc[0]
    best_bic = eligible_df.sort_values("bic", ascending=True).iloc[0]
    best_result = results_obj_by_label[best_aic["model_label"]]

    best_params = params_df[params_df["model_label"] == best_aic["model_label"]].copy()

    results_df = results_df.sort_values(
        ["mean_process", "error_distribution", "k_regimes", "aic"]
    ).reset_index(drop=True)
    params_df = params_df.sort_values(["model_label", "parameter_type", "parameter"]).reset_index(drop=True)
    attempts_df = attempts_df.sort_values(["model_label", "attempt"]).reset_index(drop=True)

    results_csv = out_dir / "regime_switching_results.csv"
    params_csv = out_dir / "regime_switching_parameters.csv"
    attempts_csv = out_dir / "regime_switching_attempts.csv"
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
        build_results_markdown(results_df),
        encoding="utf-8",
    )
    best_md_path.write_text(
        build_best_model_markdown(best_aic, best_params, best_result, df, out_dir),
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
    print(f"- {md_path}")
    print(f"- {best_md_path}")
    print(f"- {skipped_csv}")
    print(f"- {nonconv_csv}")
    print(f"- {failed_checks_csv}")
    if errors:
        print(f"- {err_json} (contains failed-spec traces)")


if __name__ == "__main__":
    main()
