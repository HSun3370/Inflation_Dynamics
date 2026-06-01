from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ANALYSIS_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from BEGE_density import BEGE_log_density as my_bege_log_density
from BEGE_density_Justin import BEGE_log_density as justin_bege_log_density
from BEGE_density_Numerical_Integration import loglikedgam_constant as numerical_log_density


DATA_PATH = PROJECT_ROOT / "DataSummary" / "Aggregate_CPI_inflation_Quarterly.pkl"
REPORT_PATH = PACKAGE_DIR / "BEGE_Density.md"

RANGE_CSV = ANALYSIS_DIR / "BEGE_density_shape_ranges.csv"
REPRESENTATIVE_CSV = ANALYSIS_DIR / "BEGE_density_representative_sets.csv"
ANALYTIC_CSV = ANALYSIS_DIR / "BEGE_density_comparison.csv"
NUMERICAL_CSV = ANALYSIS_DIR / "BEGE_density_numerical_grid.csv"
CONSISTENCY_CSV = ANALYSIS_DIR / "BEGE_density_consistency.csv"
LLF_TABLE_CSV = ANALYSIS_DIR / "BEGE_density_llf_table.csv"
SPEED_TABLE_CSV = ANALYSIS_DIR / "BEGE_density_speed_table.csv"

NUMERICAL_FIG = ANALYSIS_DIR / "BEGE_Density_numerical_convergence.png"
CONSISTENCY_FIG = ANALYSIS_DIR / "BEGE_Density_shape_consistency.png"

NUMERICAL_POINTS = [100, 500, 1000, 5000, 10000, 50000]
CONSISTENCY_VALUES = [
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
]
CONSISTENCY_NUMERICAL_POINTS = 5000
METHOD_COLUMNS = [
    ("my_function", "My function"),
    ("justin_function", "Justin function"),
    *[(f"numerical_{npoints}", f"Numerical {npoints}") for npoints in NUMERICAL_POINTS],
]


def arx11_residuals(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    y = data["Inflation"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(data)),
            data["Inflation_lag_1"].to_numpy(dtype=float),
            data["SPF"].to_numpy(dtype=float),
        ]
    )
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coef
    nobs = len(residuals)
    sigma2 = float(np.mean(residuals * residuals))
    llf = -0.5 * nobs * (np.log(2.0 * np.pi * sigma2) + 1.0)
    k = 3
    summary = {
        "coef_const": float(coef[0]),
        "coef_inflation_lag_1": float(coef[1]),
        "coef_spf": float(coef[2]),
        "nobs": float(nobs),
        "loglik": float(llf),
        "aic": float(2 * k - 2 * llf),
        "bic": float(np.log(nobs) * k - 2 * llf),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)),
        "min": float(np.min(residuals)),
        "p05": float(np.quantile(residuals, 0.05)),
        "p25": float(np.quantile(residuals, 0.25)),
        "median": float(np.median(residuals)),
        "p75": float(np.quantile(residuals, 0.75)),
        "p95": float(np.quantile(residuals, 0.95)),
        "max": float(np.max(residuals)),
        "skewness": float(stats.skew(residuals, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(residuals, fisher=True, bias=False)),
    }
    return residuals, coef, summary


def gjr_recursion(residuals, cont, rho, phi_p, phi_n, sigma) -> np.ndarray:
    r = np.asarray(residuals, dtype=np.float64)
    out = np.empty(r.shape[0], dtype=np.float64)
    floor = 1e-4
    denom = 1.0 - float(rho) - 0.5 * (float(phi_p) + float(phi_n))
    backcast = floor if denom <= 1e-12 else float(cont) / denom
    out[0] = max(backcast, floor)
    inv_scale = 1.0 / (2.0 * float(sigma) * float(sigma))
    for t in range(1, r.shape[0]):
        phi = float(phi_p) if r[t - 1] > 0.0 else float(phi_n)
        value = float(cont) + float(rho) * out[t - 1] + phi * r[t - 1] * r[t - 1] * inv_scale
        out[t] = max(value, floor)
    return out


def finite_sum(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(np.isposinf(arr)):
        return float("inf")
    if np.any(np.isneginf(arr)):
        return float("-inf")
    if not np.all(np.isfinite(arr)):
        return float("nan")
    return float(np.sum(arr))


def timed_loglik(func) -> tuple[float, float, int]:
    start = time.perf_counter()
    values = np.asarray(func(), dtype=np.float64).reshape(-1)
    elapsed = time.perf_counter() - start
    return finite_sum(values), float(elapsed), int(np.sum(~np.isfinite(values)))


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _load_estimation_table(results_dir: Path) -> pd.DataFrame:
    estimates = pd.read_csv(results_dir / "all_estimations.csv")
    diagnostics = pd.read_csv(results_dir / "selection_diagnostics.csv")
    keep_cols = [
        "seed",
        "draw",
        "mean_type",
        "corrected_loglik",
        "corrected_AIC",
        "corrected_BIC",
        "selection_eligible",
        "selection_reason",
    ]
    diagnostics = diagnostics[[c for c in keep_cols if c in diagnostics.columns]]
    return estimates.merge(diagnostics, on=["seed", "draw", "mean_type"], how="left")


def _shape_paths_for_row(model: str, row: pd.Series, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma_p = float(row["param_sigma_p"])
    sigma_n = float(row["param_sigma_n"])
    if model == "BadGood":
        p_path = gjr_recursion(residuals, row.param_p0, row.param_rho_p, row.param_phi_p, row.param_phi_p, sigma_p)
        n_path = gjr_recursion(residuals, row.param_n0, row.param_rho_n, row.param_phi_n, row.param_phi_n, sigma_n)
    else:
        raise ValueError(f"Unknown BEGE model {model}.")
    return p_path, n_path


def collect_shape_ranges(residuals: np.ndarray) -> pd.DataFrame:
    specs = {
        "BadGood": PACKAGE_DIR / "BadGood_BEGE" / "results",
    }
    range_rows = []

    for model, results_dir in specs.items():
        table = _load_estimation_table(results_dir)
        eligible = table[
            (table["mean_type"] == "ARX(1,1)")
            & (table["selection_eligible"].fillna(False).astype(bool))
        ].copy()

        if eligible.empty:
            eligible = table[(table["mean_type"] == "ARX(1,1)") & table["success"].fillna(False)].copy()

        p_all = []
        n_all = []
        for _, row in eligible.iterrows():
            p_path, n_path = _shape_paths_for_row(model, row, residuals)
            p_all.append(p_path)
            n_all.append(n_path)

        if p_all:
            p_all = np.concatenate(p_all)
            n_all = np.concatenate(n_all)
            sigma_p = eligible["param_sigma_p"].to_numpy(dtype=float)
            sigma_n = eligible["param_sigma_n"].to_numpy(dtype=float)
            p_summary = quantile_summary(p_all)
            n_summary = quantile_summary(n_all)
            range_rows.append(
                {
                    "model": model,
                    "eligible_ARX11_rows": int(len(eligible)),
                    "p_q05": p_summary["q05"],
                    "p_median": p_summary["median"],
                    "p_q95": p_summary["q95"],
                    "n_q05": n_summary["q05"],
                    "n_median": n_summary["median"],
                    "n_q95": n_summary["q95"],
                    "sigma_p_median": float(np.median(sigma_p)),
                    "sigma_n_median": float(np.median(sigma_n)),
                }
            )

    return pd.DataFrame(range_rows)


def representative_sets(range_df: pd.DataFrame) -> pd.DataFrame:
    if range_df.empty:
        raise ValueError("No BadGood ARX(1,1) range was available for density representative sets.")

    bg = range_df[range_df["model"] == "BadGood"]
    if bg.empty:
        raise ValueError("BadGood range row is required for density representative sets.")

    bg = bg.iloc[0]
    sigma_p = float(bg["sigma_p_median"])
    sigma_n = float(bg["sigma_n_median"])
    p_q05 = float(bg["p_q05"])
    p_median = float(bg["p_median"])
    p_q95 = float(bg["p_q95"])
    n_q05 = float(bg["n_q05"])
    n_median = float(bg["n_median"])
    n_q95 = float(bg["n_q95"])

    rows = [
        {
            "set": "p q05, n median",
            "p": p_q05,
            "n": n_median,
            "sigma_p": sigma_p,
            "sigma_n": sigma_n,
        },
        {
            "set": "p median, n median",
            "p": p_median,
            "n": n_median,
            "sigma_p": sigma_p,
            "sigma_n": sigma_n,
        },
        {
            "set": "p q95, n median",
            "p": p_q95,
            "n": n_median,
            "sigma_p": sigma_p,
            "sigma_n": sigma_n,
        },
        {
            "set": "p median, n q05",
            "p": p_median,
            "n": n_q05,
            "sigma_p": sigma_p,
            "sigma_n": sigma_n,
        },
        {
            "set": "p median, n q95",
            "p": p_median,
            "n": n_q95,
            "sigma_p": sigma_p,
            "sigma_n": sigma_n,
        },
    ]
    return pd.DataFrame(rows)


def median_badgood_parameters(range_df: pd.DataFrame) -> dict[str, float]:
    if range_df.empty:
        raise ValueError("No BadGood ARX(1,1) range was available for tail consistency.")

    bg = range_df[range_df["model"] == "BadGood"]
    if bg.empty:
        raise ValueError("BadGood range row is required for tail consistency.")

    bg = bg.iloc[0]
    return {
        "p": float(bg["p_median"]),
        "n": float(bg["n_median"]),
        "sigma_p": float(bg["sigma_p_median"]),
        "sigma_n": float(bg["sigma_n_median"]),
    }


def compare_density_methods(residuals: np.ndarray, reps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analytic_rows = []
    numerical_rows = []

    # Warm up imports and vectorized paths outside the timed tables.
    first = reps.iloc[0]
    my_bege_log_density(residuals[:3], first.p, first.n, first.sigma_p, first.sigma_n)
    justin_bege_log_density(residuals[:3], first.p, first.n, first.sigma_p, first.sigma_n)
    numerical_log_density(residuals[:3], first.p, first.n, first.sigma_p, first.sigma_n, npoints=50)

    for _, spec in reps.iterrows():
        p = float(spec["p"])
        n = float(spec["n"])
        sigma_p = float(spec["sigma_p"])
        sigma_n = float(spec["sigma_n"])

        my_ll, my_seconds, my_bad = timed_loglik(
            lambda: my_bege_log_density(residuals, p, n, sigma_p, sigma_n, hyperu_method="scipy_approx")
        )
        justin_ll, justin_seconds, justin_bad = timed_loglik(
            lambda: justin_bege_log_density(residuals, p, n, sigma_p, sigma_n, hyperu_method="scipy")
        )

        analytic_rows.append(
            {
                **spec.to_dict(),
                "my_loglik": my_ll,
                "justin_loglik": justin_ll,
                "my_minus_justin": my_ll - justin_ll,
                "my_seconds": my_seconds,
                "justin_seconds": justin_seconds,
                "speedup_vs_justin": justin_seconds / my_seconds if my_seconds > 0 else np.nan,
                "my_bad_obs": my_bad,
                "justin_bad_obs": justin_bad,
            }
        )

        for npoints in NUMERICAL_POINTS:
            numerical_ll, numerical_seconds, numerical_bad = timed_loglik(
                lambda npoints=npoints: numerical_log_density(
                    residuals, p, n, sigma_p, sigma_n, npoints=int(npoints)
                )
            )
            numerical_rows.append(
                {
                    **spec.to_dict(),
                    "npoints": int(npoints),
                    "numerical_loglik": numerical_ll,
                    "numerical_minus_justin": numerical_ll - justin_ll,
                    "numerical_minus_my": numerical_ll - my_ll,
                    "numerical_seconds": numerical_seconds,
                    "numerical_bad_obs": numerical_bad,
                }
            )

    return pd.DataFrame(analytic_rows), pd.DataFrame(numerical_rows)


def wide_method_tables(analytic_df: pd.DataFrame, numerical_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    llf_rows = []
    speed_rows = []

    for _, analytic in analytic_df.iterrows():
        set_name = analytic["set"]
        numerical = numerical_df[numerical_df["set"] == set_name].set_index("npoints")
        llf_row = {
            "set": set_name,
            "my_function": float(analytic["my_loglik"]),
            "justin_function": float(analytic["justin_loglik"]),
        }
        speed_row = {
            "set": set_name,
            "my_function": float(analytic["my_seconds"]),
            "justin_function": float(analytic["justin_seconds"]),
        }

        for npoints in NUMERICAL_POINTS:
            col = f"numerical_{npoints}"
            if npoints in numerical.index:
                llf_row[col] = float(numerical.loc[npoints, "numerical_loglik"])
                speed_row[col] = float(numerical.loc[npoints, "numerical_seconds"])
            else:
                llf_row[col] = np.nan
                speed_row[col] = np.nan

        llf_rows.append(llf_row)
        speed_rows.append(speed_row)

    return pd.DataFrame(llf_rows), pd.DataFrame(speed_rows)


def run_consistency(residuals: np.ndarray, base: dict[str, float]) -> pd.DataFrame:
    rows = []
    for vary in ["p", "n"]:
        values = np.unique(np.r_[np.asarray(CONSISTENCY_VALUES, dtype=np.float64), float(base[vary])])
        for value in values:
            p = float(base["p"])
            n = float(base["n"])
            if vary == "p":
                p = float(value)
            else:
                n = float(value)

            my_ll, my_seconds, my_bad = timed_loglik(
                lambda: my_bege_log_density(residuals, p, n, base["sigma_p"], base["sigma_n"])
            )
            justin_ll, justin_seconds, justin_bad = timed_loglik(
                lambda: justin_bege_log_density(residuals, p, n, base["sigma_p"], base["sigma_n"])
            )
            numerical_ll, numerical_seconds, numerical_bad = timed_loglik(
                lambda: numerical_log_density(
                    residuals,
                    p,
                    n,
                    base["sigma_p"],
                    base["sigma_n"],
                    npoints=CONSISTENCY_NUMERICAL_POINTS,
                )
            )
            rows.append(
                {
                    "vary": vary,
                    "value": float(value),
                    "p": p,
                    "n": n,
                    "my_loglik": my_ll,
                    "justin_loglik": justin_ll,
                    "numerical_loglik": numerical_ll,
                    "my_minus_justin": my_ll - justin_ll,
                    "numerical_minus_justin": numerical_ll - justin_ll,
                    "my_seconds": my_seconds,
                    "justin_seconds": justin_seconds,
                    "numerical_seconds": numerical_seconds,
                    "my_bad_obs": my_bad,
                    "justin_bad_obs": justin_bad,
                    "numerical_bad_obs": numerical_bad,
                }
            )
    return pd.DataFrame(rows)


def fmt(value, digits=4) -> str:
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if np.isinf(float(value)):
        return "inf" if float(value) > 0 else "-inf"
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns, labels=None, digits=None) -> str:
    labels = labels or columns
    digits = digits or {}
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in columns) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            vals.append(fmt(row[col], digits.get(col, 4)))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_figures(numerical_df: pd.DataFrame, consistency_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9.5, 5.6))
    for set_name, group in numerical_df.groupby("set"):
        group = group.sort_values("npoints")
        plt.plot(group["npoints"], group["numerical_minus_justin"], marker="o", linewidth=1.5, label=set_name)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xscale("log")
    plt.xlabel("Numerical integration grid points")
    plt.ylabel("Numerical log likelihood minus Justin analytic")
    plt.title("Numerical integration convergence on ARX(1,1) residuals")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(NUMERICAL_FIG, dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, vary in zip(axes, ["p", "n"]):
        group = consistency_df[consistency_df["vary"] == vary].sort_values("value")
        ax.plot(group["value"], group["my_loglik"], marker="o", label="My density")
        ax.plot(group["value"], group["justin_loglik"], marker="x", label="Justin")
        ax.plot(group["value"], group["numerical_loglik"], marker="s", label=f"Numerical {CONSISTENCY_NUMERICAL_POINTS}")
        ax.set_xscale("log")
        ax.set_xlabel(vary)
        ax.set_ylabel("Log likelihood")
        ax.set_title(f"Varying {vary}, other parameters fixed")
        ax.axvline(200.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CONSISTENCY_FIG, dpi=180)
    plt.close()


def write_report(
    data: pd.DataFrame,
    residual_summary: dict[str, float],
    range_df: pd.DataFrame,
    reps: pd.DataFrame,
    llf_df: pd.DataFrame,
    speed_df: pd.DataFrame,
    consistency_base: dict[str, float],
) -> None:
    residual_rows = pd.DataFrame(
        [
            {"Statistic": "Date start", "Value": str(data.index.min())},
            {"Statistic": "Date end", "Value": str(data.index.max())},
            {"Statistic": "Observations", "Value": int(residual_summary["nobs"])},
            {"Statistic": "Mean", "Value": residual_summary["mean"]},
            {"Statistic": "Std", "Value": residual_summary["std"]},
            {"Statistic": "Min", "Value": residual_summary["min"]},
            {"Statistic": "P5", "Value": residual_summary["p05"]},
            {"Statistic": "Median", "Value": residual_summary["median"]},
            {"Statistic": "P95", "Value": residual_summary["p95"]},
            {"Statistic": "Max", "Value": residual_summary["max"]},
            {"Statistic": "Skewness", "Value": residual_summary["skewness"]},
            {"Statistic": "Excess kurtosis", "Value": residual_summary["excess_kurtosis"]},
        ]
    )

    range_report = range_df[
        [
            "model",
            "eligible_ARX11_rows",
            "p_q05",
            "p_median",
            "p_q95",
            "n_q05",
            "n_median",
            "n_q95",
            "sigma_p_median",
            "sigma_n_median",
        ]
    ].copy()

    method_columns = ["set"] + [col for col, _ in METHOD_COLUMNS]
    method_labels = ["Parameter set"] + [label for _, label in METHOD_COLUMNS]
    figure_path = NUMERICAL_FIG.relative_to(PACKAGE_DIR).as_posix()
    consistency_figure_path = CONSISTENCY_FIG.relative_to(PACKAGE_DIR).as_posix()
    artifact_paths = [
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in [
            RANGE_CSV,
            REPRESENTATIVE_CSV,
            ANALYTIC_CSV,
            NUMERICAL_CSV,
            LLF_TABLE_CSV,
            SPEED_TABLE_CSV,
            CONSISTENCY_CSV,
        ]
    ]

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        "# BEGE Density",
        "",
        "This audit compares three fixed-shape BEGE density implementations on the ARX(1,1) residuals from the canonical effective sample.",
        "",
        "The ARX(1,1) residual is computed as",
        "",
        "$$",
        "u_t = \\pi_t - \\left(c + \\rho_1 \\pi_{t-1} + \\phi_1 SPF_t\\right),",
        "$$",
        "",
        "using the OLS coefficients re-estimated on the 1969Q2--2022Q4 sample:",
        "",
        "$$",
        f"\\hat\\pi_t = {residual_summary['coef_const']:.6f}"
        f" + {residual_summary['coef_inflation_lag_1']:.6f}\\pi_{{t-1}}"
        f" + {residual_summary['coef_spf']:.6f}SPF_t.",
        "$$",
        "",
        f"The Gaussian OLS log likelihood is `{residual_summary['loglik']:.3f}`, AIC is `{residual_summary['aic']:.3f}`, and BIC is `{residual_summary['bic']:.3f}`.",
        "",
        "## ARX(1,1) Residual Summary",
        "",
        markdown_table(residual_rows, ["Statistic", "Value"], ["Statistic", "Value"], {"Value": 6}),
        "",
        "## Shape Parameter Range From Previous BEGE Runs",
        "",
        "The range below uses eligible ARX(1,1) rows from the BadGood BEGE results only. For each saved parameter vector I recomputed the recursive shape paths on the common ARX(1,1) residuals, then summarized all observations and all eligible rows. The density comparison itself keeps the selected shape values constant across time.",
        "",
        markdown_table(
            range_report,
            [
                "model",
                "eligible_ARX11_rows",
                "p_q05",
                "p_median",
                "p_q95",
                "n_q05",
                "n_median",
                "n_q95",
                "sigma_p_median",
                "sigma_n_median",
            ],
            [
                "Model",
                "Rows",
                "p q05",
                "p median",
                "p q95",
                "n q05",
                "n median",
                "n q95",
                "sigma_p med",
                "sigma_n med",
            ],
            {c: 4 for c in range_report.columns},
        ),
        "",
        "## Fixed-Shape Parameter Sets",
        "",
        "The log likelihood and timing comparison uses the following five BadGood-based parameter sets. In each row, `sigma_p` and `sigma_n` are held at their BadGood medians.",
        "",
        markdown_table(
            reps,
            ["set", "p", "n", "sigma_p", "sigma_n"],
            ["Set", "p", "n", "sigma_p", "sigma_n"],
            {"p": 6, "n": 6, "sigma_p": 6, "sigma_n": 6},
        ),
        "",
        "## Log Likelihood Comparison",
        "",
        "`BEGE_density.py` is the current implementation. `BEGE_density_Justin.py` is Justin's analytic formula, and the numerical columns use `BEGE_density_Numerical_Integration.py::loglikedgam_constant` at the stated grid size.",
        "",
        markdown_table(
            llf_df,
            method_columns,
            method_labels,
            {col: 6 for col in method_columns if col != "set"},
        ),
        "",
        "## Evaluation Speed Comparison",
        "",
        markdown_table(
            speed_df,
            method_columns,
            method_labels,
            {col: 4 for col in method_columns if col != "set"},
        ),
        "",
        f"![Numerical integration convergence]({figure_path})",
        "",
        "## Shape Tail Consistency",
        "",
        f"Holding the other parameters at the BadGood medians (`p={consistency_base['p']:.4f}`, `n={consistency_base['n']:.4f}`, `sigma_p={consistency_base['sigma_p']:.4f}`, `sigma_n={consistency_base['sigma_n']:.4f}`), I vary either `p` or `n` up to 5000 and compare the three density functions. The numerical integration line uses {CONSISTENCY_NUMERICAL_POINTS:,} grid points.",
        "",
        f"![Shape consistency]({consistency_figure_path})",
        "",
        "## Findings",
        "",
        "- BadGood quantiles provide the fixed-shape examples used in the likelihood and timing tables.",
        "- The numerical integration function is useful as a convergence diagnostic, but it is much slower than the analytic functions and remains grid-sensitive.",
        "- The shape-tail plot varies one shape parameter at a time while holding the other shape and both scale parameters at the BadGood medians.",
        "",
        "Generated audit files:",
        "",
        *[f"- `{path}`" for path in artifact_paths],
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_audit():
    data = pd.read_pickle(DATA_PATH)
    residuals, _, residual_summary = arx11_residuals(data)
    range_df = collect_shape_ranges(residuals)
    reps = representative_sets(range_df)
    analytic_df, numerical_df = compare_density_methods(residuals, reps)
    llf_df, speed_df = wide_method_tables(analytic_df, numerical_df)
    consistency_base = median_badgood_parameters(range_df)
    consistency_df = run_consistency(residuals, consistency_base)

    range_df.to_csv(RANGE_CSV, index=False)
    reps.to_csv(REPRESENTATIVE_CSV, index=False)
    analytic_df.to_csv(ANALYTIC_CSV, index=False)
    numerical_df.to_csv(NUMERICAL_CSV, index=False)
    llf_df.to_csv(LLF_TABLE_CSV, index=False)
    speed_df.to_csv(SPEED_TABLE_CSV, index=False)
    consistency_df.to_csv(CONSISTENCY_CSV, index=False)

    write_figures(numerical_df, consistency_df)
    write_report(
        data,
        residual_summary,
        range_df,
        reps,
        llf_df,
        speed_df,
        consistency_base,
    )
    return range_df, reps, analytic_df, numerical_df, llf_df, speed_df, consistency_df


if __name__ == "__main__":
    outputs = run_audit()
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RANGE_CSV}")
    print(f"Wrote {REPRESENTATIVE_CSV}")
    print(f"Wrote {ANALYTIC_CSV}")
    print(f"Wrote {NUMERICAL_CSV}")
    print(f"Wrote {LLF_TABLE_CSV}")
    print(f"Wrote {SPEED_TABLE_CSV}")
    print(f"Wrote {CONSISTENCY_CSV}")
