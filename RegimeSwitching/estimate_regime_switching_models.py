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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

import MarkovRegression_t


ROOT = Path(__file__).resolve().parents[1]
RS_DIR = Path(__file__).resolve().parent
OUT_DIR = RS_DIR / "results"


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


def fit_one(df: pd.DataFrame, spec: ModelSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean = spec.mean_process
    switch = spec.switch_spec

    endog = df[mean.endog_col]
    exog = df[list(mean.exog_cols)] if mean.exog_cols else None

    switching_exog: bool | list[bool]
    if exog is None:
        switching_exog = False
    else:
        mask = exog_switch_mask(mean.exog_cols, switch)
        switching_exog = mask

    kwargs = dict(
        endog=endog,
        exog=exog,
        k_regimes=switch.k_regimes,
        trend="n",
        switching_trend=False,
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

    result = model.fit(disp=False)
    retvals = getattr(result, "mle_retvals", {})

    model_label = (
        f"{mean.name}_"
        f"{spec.error_distribution.replace(' ', '_').lower()}_"
        f"r{switch.k_regimes}_"
        f"ar{format_bool(switch.switching_ar)}_"
        f"spf{format_bool(switch.switching_spf)}"
    )

    summary_row: dict[str, Any] = {
        "model_label": model_label,
        "mean_process": mean.name,
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
        "converged": bool(_safe_get(retvals, "converged")),
        "iterations": _safe_get(retvals, "iterations"),
        "warnflag": _safe_get(retvals, "warnflag"),
    }

    parameter_rows: list[dict[str, Any]] = []
    for name, val in zip(result.model.param_names, result.params):
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
                "parameter": name,
                "parameter_type": ptype,
                "estimate": val_real,
            }
        )

    return summary_row, parameter_rows


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

    for c in ["LogLik", "AIC", "BIC"]:
        view[c] = view[c].map(lambda x: f"{x:.3f}")
    view = view.sort_values(["Mean", "AIC", "K"], ascending=[True, True, True]).reset_index(drop=True)
    return view.to_markdown(index=False)


def build_best_model_markdown(best_row: pd.Series, best_params: pd.DataFrame) -> str:
    mean_params = best_params[best_params["parameter_type"] == "mean_process"]
    dist_params = best_params[best_params["parameter_type"] == "distribution"]
    trans_params = best_params[best_params["parameter_type"] == "transition_probability"]

    lines = [
        "# Best Regime-Switching Model",
        "",
        "Selected by AIC among converged models.",
        "",
        "## Model Summary",
        f"- Model label: `{best_row['model_label']}`",
        f"- Mean process: `{best_row['mean_process']}`",
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
        "## Mean Process Parameters",
        mean_params[["parameter", "estimate"]].to_markdown(index=False),
        "",
        "## Distribution Parameters",
        dist_params[["parameter", "estimate"]].to_markdown(index=False),
        "",
        "## Transition Probabilities",
        trans_params[["parameter", "estimate"]].to_markdown(index=False),
    ]
    return "\n".join(lines)


def main() -> None:
    np.random.seed(20260521)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_effective_sample()
    specs, skipped = build_model_specs()

    results_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for spec in specs:
        switch = spec.switch_spec
        tag = (
            f"{spec.mean_process.name} | {spec.error_distribution} | "
            f"K={switch.k_regimes} | AR={format_bool(switch.switching_ar)} | "
            f"SPF={format_bool(switch.switching_spf)}"
        )
        try:
            row, params = fit_one(df, spec)
            results_rows.append(row)
            parameter_rows.extend(params)
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

    results_df["aic_rank"] = results_df["aic"].rank(method="dense")
    results_df["bic_rank"] = results_df["bic"].rank(method="dense")

    # Best models should be chosen among converged fits when available.
    converged_df = results_df[results_df["converged"]].copy()
    if converged_df.empty:
        converged_df = results_df.copy()

    best_aic = converged_df.sort_values("aic", ascending=True).iloc[0]
    best_bic = converged_df.sort_values("bic", ascending=True).iloc[0]

    best_params = params_df[params_df["model_label"] == best_aic["model_label"]].copy()

    results_df = results_df.sort_values(
        ["mean_process", "error_distribution", "k_regimes", "aic"]
    ).reset_index(drop=True)
    params_df = params_df.sort_values(["model_label", "parameter_type", "parameter"]).reset_index(drop=True)

    results_csv = OUT_DIR / "regime_switching_results.csv"
    params_csv = OUT_DIR / "regime_switching_parameters.csv"
    md_path = OUT_DIR / "regime_switching_results.md"
    best_md_path = OUT_DIR / "regime_switching_best_model.md"
    skipped_csv = OUT_DIR / "regime_switching_skipped_models.csv"
    err_json = OUT_DIR / "regime_switching_errors.json"

    results_df.to_csv(results_csv, index=False)
    params_df.to_csv(params_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)

    md_path.write_text(
        build_results_markdown(results_df, best_aic, best_bic, skipped_df),
        encoding="utf-8",
    )
    best_md_path.write_text(
        build_best_model_markdown(best_aic, best_params),
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
    if errors:
        print(f"- {err_json} (contains failed-spec traces)")


if __name__ == "__main__":
    main()
