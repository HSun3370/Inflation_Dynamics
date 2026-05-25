"""Estimate regime-switching models across all mean-process specifications.

Outputs:
- results/regime_switching_results.csv
- results/regime_switching_parameters.csv
- results/regime_switching_results.md
- results/regime_switching_best_model.md
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import MarkovRegression_t


ROOT = Path(__file__).resolve().parents[1]
RS_DIR = Path(__file__).resolve().parent
OUT_DIR = RS_DIR / "results"
TYPST_PREAMBLE = "```{raw:typst}\n#set page(margin: auto)\n```"


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


def load_effective_sample() -> pd.DataFrame:
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


def build_model_specs() -> tuple[list[ModelSpec], list[dict[str, Any]]]:
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
            specs.append(
                ModelSpec(
                    mean_process=mean,
                    error_distribution="Student t",
                    switch_spec=switch,
                    switching_nu=True,
                )
            )

    return specs, skipped


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


def _fit_with_retries(model: Any, seed_base: int) -> tuple[Any, str, int]:
    attempts = [
        (
            "baseline_bfgs",
            {"method": "bfgs", "maxiter": 300, "em_iter": 10, "search_reps": 0, "search_iter": 5},
            1,
            None,
        ),
        (
            "search_bfgs",
            {"method": "bfgs", "maxiter": 800, "em_iter": 20, "search_reps": 40, "search_iter": 20, "search_scale": 1.0},
            2,
            None,
        ),
        (
            "search_lbfgs",
            {"method": "lbfgs", "maxiter": 1500, "em_iter": 30, "search_reps": 80, "search_iter": 25, "search_scale": 1.5},
            2,
            None,
        ),
        (
            "perturbed_bfgs",
            {"method": "bfgs", "maxiter": 1200, "em_iter": 20, "search_reps": 20, "search_iter": 15, "search_scale": 1.0},
            3,
            0.15,
        ),
        (
            "perturbed_lbfgs",
            {"method": "lbfgs", "maxiter": 1800, "em_iter": 20, "search_reps": 20, "search_iter": 15, "search_scale": 1.0},
            3,
            0.25,
        ),
    ]

    best_any = None
    best_any_llf = -np.inf
    best_any_tag = "none"
    n_attempts = 0

    for attempt_name, fit_kwargs, reps, perturb_scale in attempts:
        for rep in range(reps):
            n_attempts += 1
            attempt_seed = seed_base + 1009 * n_attempts + 37 * rep
            np.random.seed(attempt_seed)

            kwargs = dict(fit_kwargs)
            if perturb_scale is not None:
                base = np.asarray(model.start_params, dtype=float)
                rng = np.random.default_rng(attempt_seed + 17)
                noise = rng.normal(loc=0.0, scale=perturb_scale, size=base.shape)
                kwargs["start_params"] = base + noise
                kwargs["transformed"] = True

            try:
                res = model.fit(disp=False, **kwargs)
            except Exception:
                continue

            llf = float(res.llf)
            tag = f"{attempt_name}#{rep+1}"
            if llf > best_any_llf:
                best_any = res
                best_any_llf = llf
                best_any_tag = tag

            if _is_converged(res):
                return res, tag, n_attempts

    if best_any is None:
        raise RuntimeError("All optimization attempts failed.")
    return best_any, best_any_tag, n_attempts


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


def fit_one(
    df: pd.DataFrame, spec: ModelSpec
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
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

    model_label = (
        f"{mean.name}_"
        f"{spec.error_distribution.replace(' ', '_').lower()}_"
        f"r{switch.k_regimes}_"
        f"ar{format_bool(switch.switching_ar)}_"
        f"spf{format_bool(switch.switching_spf)}"
    )
    result, fit_strategy, fit_attempts = _fit_with_retries(model, seed_base=_stable_seed(model_label))
    retvals = getattr(result, "mle_retvals", {})

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
        "iterations": _safe_get(retvals, "iterations"),
        "warnflag": _safe_get(retvals, "warnflag"),
        "fit_strategy": fit_strategy,
        "fit_attempts": fit_attempts,
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

    return summary_row, parameter_rows, result


def build_results_markdown(
    results_df: pd.DataFrame,
    best_aic: pd.Series,
    best_bic: pd.Series,
    skipped_df: pd.DataFrame,
) -> str:
    # Keep only the compact estimated-model table for reporting.
    view = results_df[
        [
            "mean_process",
            "error_distribution",
            "k_regimes",
            "switching_ar",
            "switching_spf",
            "switching_distribution",
            "switching_nu",
            "llf",
            "aic",
            "bic",
        ]
    ].copy()

    # Ordered reporting: mean model -> distribution -> #regime -> switching flags.
    mean_order = {"Constant": 0, "ARX(1,1)": 1, "ARX(2,1)": 2, "ARX(2,2)": 3}
    dist_order = {"Normal": 0, "Student t": 1}
    yn_order = {"N": 0, "Y": 1}
    view["_mean_order"] = view["mean_process"].map(mean_order).fillna(999)
    view["_dist_order"] = view["error_distribution"].map(dist_order).fillna(999)
    view["_sw_ar_order"] = view["switching_ar"].map(yn_order).fillna(999)
    view["_sw_spf_order"] = view["switching_spf"].map(yn_order).fillna(999)
    view["_sw_var_order"] = view["switching_distribution"].map(yn_order).fillna(999)
    view["_sw_nu_order"] = view["switching_nu"].map(yn_order).fillna(999)

    view = view.sort_values(
        [
            "_mean_order",
            "_dist_order",
            "k_regimes",
            "_sw_ar_order",
            "_sw_spf_order",
            "_sw_var_order",
            "_sw_nu_order",
            "aic",
        ],
        ascending=[True, True, True, True, True, True, True, True],
    ).reset_index(drop=True)

    view = view.rename(
        columns={
            "mean_process": "Mean",
            "error_distribution": "Dist",
            "k_regimes": "K",
            "switching_ar": "Sw.AR",
            "switching_spf": "Sw.SPF",
            "switching_distribution": "Sw.Var",
            "switching_nu": "Sw.nu",
            "llf": "LogLik",
            "aic": "AIC",
            "bic": "BIC",
        }
    )
    view = view.drop(
        columns=[
            "_mean_order",
            "_dist_order",
            "_sw_ar_order",
            "_sw_spf_order",
            "_sw_var_order",
            "_sw_nu_order",
        ]
    )

    for c in ["LogLik", "AIC", "BIC"]:
        view[c] = view[c].map(lambda x: f"{x:.3f}")

    return TYPST_PREAMBLE + "\n\n" + view.to_markdown(index=False)


def _mean_equation_regime_table(best_row: pd.Series, best_params: pd.DataFrame) -> pd.DataFrame:
    k_regimes = int(best_row["k_regimes"])
    exog_cols = [c for c in str(best_row["exog_cols"]).split(",") if c]
    has_intercept = best_row["has_intercept"] == "Y"

    estimates = dict(zip(best_params["parameter"], best_params["estimate"]))
    intercept_keys = [k for k in estimates if k.startswith("const[")]
    shared_intercept = estimates[intercept_keys[0]] if len(intercept_keys) == 1 else None

    rows: list[dict[str, Any]] = []
    for r in range(k_regimes):
        row: dict[str, Any] = {"regime": r}
        if has_intercept:
            key = f"const[{r}]"
            if key in estimates:
                row["Intercept"] = estimates[key]
            elif "const" in estimates:
                row["Intercept"] = estimates["const"]
            elif shared_intercept is not None:
                row["Intercept"] = shared_intercept

        for col in exog_cols:
            sw_key = f"{col}[{r}]"
            non_sw_key = col
            if sw_key in estimates:
                row[col] = estimates[sw_key]
            elif non_sw_key in estimates:
                row[col] = estimates[non_sw_key]
            else:
                shared = [k for k in estimates if k.startswith(f"{col}[")]
                if len(shared) == 1:
                    row[col] = estimates[shared[0]]
                else:
                    row[col] = np.nan
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
        rows.append([f"Regime {i}", *probs, float(np.sum(from_to[i]))])
    return pd.DataFrame(rows, columns=cols)


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
    cmap = plt.get_cmap("tab10", k)
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
                color=cmap(regime),
                alpha=0.20,
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
    trans_params = best_params[best_params["parameter_type"] == "transition_probability"]
    mean_regime_table = _mean_equation_regime_table(best_row, mean_params)
    transition_matrix = _transition_matrix_table(best_result)
    plot_name = _save_regime_plot(df, best_row, best_result, out_dir)

    exog_cols = [c for c in str(best_row["exog_cols"]).split(",") if c]
    rhs_terms = []
    if best_row["has_intercept"] == "Y":
        rhs_terms.append("c_(s_t)")
    for col in exog_cols:
        if best_row["switching_ar"] == "Y" and is_ar_col(col):
            rhs_terms.append(f"beta_{{{col}}}(s_t) * {col}")
        elif best_row["switching_spf"] == "Y" and is_spf_col(col):
            rhs_terms.append(f"beta_{{{col}}}(s_t) * {col}")
        else:
            rhs_terms.append(f"beta_{{{col}}} * {col}")

    rhs = " + ".join(rhs_terms) if rhs_terms else "0"
    eqn = f"{best_row['endog']} = {rhs} + u_t"

    lines = [
        "# Best Regime-Switching Model",
        "",
        "Selected by AIC among converged models.",
        "",
        "## Model Summary",
        f"- Model label: `{best_row['model_label']}`",
        f"- Mean process: `{best_row['mean_process']}`",
        f"- Mean equation target: `{best_row['endog']}`",
        f"- Intercept included: `{best_row['has_intercept']}`",
        f"- Intercept switches by regime: `{best_row['intercept_switches_by_regime']}`",
        f"- Error distribution: `{best_row['error_distribution']}`",
        f"- #Regime: `{int(best_row['k_regimes'])}`",
        f"- Switching AR: `{best_row['switching_ar']}`",
        f"- Switching SPF: `{best_row['switching_spf']}`",
        f"- Switching distribution variance: `{best_row['switching_distribution']}`",
        f"- Switching nu: `{best_row['switching_nu']}`",
        f"- Sample: `{best_row['sample_start']} to {best_row['sample_end']}`",
        f"- Nobs: `{int(best_row['nobs'])}`",
        f"- LogLik: `{best_row['llf']:.3f}`",
        f"- AIC: `{best_row['aic']:.3f}`",
        f"- BIC: `{best_row['bic']:.3f}`",
        "",
        "## Mean Model Specification",
        f"`{eqn}`",
        "",
        "Regime-specific mean coefficients:",
        mean_regime_table.to_markdown(index=False),
        "",
        "## Mean Process Parameters",
        mean_params[["parameter", "estimate"]].to_markdown(index=False),
        "",
        "## Distribution Parameters",
        dist_params[["parameter", "estimate"]].to_markdown(index=False),
        "",
        "## Transition Probabilities (Vector Form)",
        trans_params[["parameter", "estimate"]].to_markdown(index=False),
        "",
        "## Transition Probability Matrix",
        transition_matrix.to_markdown(index=False),
        "",
        "## Regime Classification Plot",
        f"![Inflation with Smoothed Predicted Regimes]({plot_name})",
    ]
    return TYPST_PREAMBLE + "\n\n" + "\n".join(lines)


def main() -> None:
    np.random.seed(20260521)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_effective_sample()
    specs, skipped = build_model_specs()

    results_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
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
            row, params, result_obj = fit_one(df, spec)
            results_rows.append(row)
            parameter_rows.extend(params)
            results_obj_by_label[row["model_label"]] = result_obj
            print(
                f"[OK] {tag}: llf={row['llf']:.3f}, aic={row['aic']:.3f}, "
                f"bic={row['bic']:.3f}, converged={row['converged']}, nobs={row['nobs']}"
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
    skipped_df = pd.DataFrame(skipped)

    # Reported model comparisons should include converged fits only.
    converged_df = results_df[results_df["converged"]].copy()
    nonconverged_df = results_df[~results_df["converged"]].copy()
    if converged_df.empty:
        raise RuntimeError("No converged model after retry strategy.")

    converged_df["aic_rank"] = converged_df["aic"].rank(method="dense")
    converged_df["bic_rank"] = converged_df["bic"].rank(method="dense")

    best_aic = converged_df.sort_values("aic", ascending=True).iloc[0]
    best_bic = converged_df.sort_values("bic", ascending=True).iloc[0]
    best_result = results_obj_by_label[best_aic["model_label"]]

    best_params = params_df[params_df["model_label"] == best_aic["model_label"]].copy()

    results_df = converged_df.sort_values(
        ["mean_process", "error_distribution", "k_regimes", "aic"]
    ).reset_index(drop=True)
    params_df = params_df.sort_values(["model_label", "parameter_type", "parameter"]).reset_index(drop=True)

    results_csv = OUT_DIR / "regime_switching_results.csv"
    params_csv = OUT_DIR / "regime_switching_parameters.csv"
    md_path = OUT_DIR / "regime_switching_results.md"
    best_md_path = OUT_DIR / "regime_switching_best_model.md"
    skipped_csv = OUT_DIR / "regime_switching_skipped_models.csv"
    nonconv_csv = OUT_DIR / "regime_switching_nonconverged.csv"
    err_json = OUT_DIR / "regime_switching_errors.json"

    results_df.to_csv(results_csv, index=False)
    params_df.to_csv(params_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)
    nonconverged_df.to_csv(nonconv_csv, index=False)

    md_path.write_text(
        build_results_markdown(results_df, best_aic, best_bic, skipped_df),
        encoding="utf-8",
    )
    best_md_path.write_text(
        build_best_model_markdown(best_aic, best_params, best_result, df, OUT_DIR),
        encoding="utf-8",
    )

    if errors:
        err_json.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    elif err_json.exists():
        err_json.unlink()

    print("\nWrote files:")
    print(f"- {results_csv}")
    print(f"- {params_csv}")
    print(f"- {md_path}")
    print(f"- {best_md_path}")
    print(f"- {skipped_csv}")
    print(f"- {nonconv_csv}")
    if errors:
        print(f"- {err_json} (contains failed-spec traces)")


if __name__ == "__main__":
    main()
