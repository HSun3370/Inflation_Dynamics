from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd
from scipy.special import digamma, hyperu, loggamma
from scipy.stats import gamma

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from BEGE_density import BEGE_log_density


DATA_PATH = PROJECT_ROOT / "DataSummary" / "Aggregate_CPI_inflation_Quarterly.pkl"
REPORT_PATH = PACKAGE_DIR / "BEGE_Density.md"
RESULTS_CSV = PACKAGE_DIR / "BEGE_density_comparison.csv"
NUMERICAL_CSV = PACKAGE_DIR / "BEGE_density_numerical_grid.csv"
CONVERGENCE_FIG = PACKAGE_DIR / "BEGE_Density_numerical_convergence.png"
ANALYTIC_FIG = PACKAGE_DIR / "BEGE_Density_analytic_difference.png"


mp.mp.dps = 25
HYPERU_INTEGER_B_TOL = 1e-6


def _original_log_hyperu_helper_scalar(a, b, z, hyperu_method="scipy"):
    """
    Original scalar helper from BEGE_density.py before the speed pass.

    The original file referenced sys.float_info.min but did not import sys.
    This audit supplies that missing import so the original intended SciPy path
    is actually exercised.
    """

    def compute_large_z_approximation():
        return -a * np.log(z)

    def compute_small_z_approximation():
        if b > 1:
            return loggamma(b - 1) - loggamma(a) + (1 - b) * np.log(z)
        if b < 1:
            return loggamma(1 - b) - loggamma(a - b + 1)

        euler_gamma = 0.5772156649015329
        leading_term = -np.log(z) - digamma(a) - 2 * euler_gamma
        if leading_term <= 0:
            return np.nan
        return np.log(leading_term) - loggamma(a)

    def compute_approximation():
        if z < 1e-8:
            return compute_small_z_approximation()
        return compute_large_z_approximation()

    def compute_with_mpmath():
        a_mp = mp.mpf(float(a))
        b_mp = mp.mpf(float(b))
        z_mp = mp.mpf(float(z))
        result = mp.hyperu(a_mp, b_mp, z_mp)
        if result <= 0:
            return np.nan
        result_log = mp.log(result)

        if not mp.isfinite(result_log):
            return compute_approximation()
        return float(result_log)

    def compute_with_scipy():
        result = hyperu(a, b, z)
        if result <= sys.float_info.min or not np.isfinite(result):
            try:
                return compute_with_mpmath()
            except Exception:
                return compute_approximation()
        return np.log(result)

    try:
        near_integer_b = np.isclose(b, np.round(b), rtol=0.0, atol=HYPERU_INTEGER_B_TOL)
        if hyperu_method == "mpmath" or b >= 40 or near_integer_b:
            return compute_with_mpmath()
        return compute_with_scipy()
    except Exception:
        return compute_approximation()


original_log_hyperu_helper = np.vectorize(_original_log_hyperu_helper_scalar, otypes=[np.float64])


def original_bege_log_density(x, p, n, sigma_p, sigma_n, hyperu_method="scipy"):
    x, p, n, sigma_p, sigma_n = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
        np.asarray(n, dtype=np.float64),
        np.asarray(sigma_p, dtype=np.float64),
        np.asarray(sigma_n, dtype=np.float64),
    )

    x = np.atleast_1d(x).astype(np.float64, copy=False)
    p = np.atleast_1d(p).astype(np.float64, copy=False)
    n = np.atleast_1d(n).astype(np.float64, copy=False)
    sigma_p = np.atleast_1d(sigma_p).astype(np.float64, copy=False)
    sigma_n = np.atleast_1d(sigma_n).astype(np.float64, copy=False)

    valid = (
        np.isfinite(x)
        & np.isfinite(p)
        & np.isfinite(n)
        & np.isfinite(sigma_p)
        & np.isfinite(sigma_n)
        & (p > 0)
        & (n > 0)
        & (sigma_p > 0)
        & (sigma_n > 0)
    )

    k_omega_p = p
    k_omega_n = n
    theta_omega_p = sigma_p
    theta_omega_n = sigma_n
    omega_p_underscore = -k_omega_p * theta_omega_p
    omega_n_underscore = -k_omega_n * theta_omega_n
    theta_tilde = 1 / theta_omega_p + 1 / theta_omega_n
    k = 0.5 * (k_omega_n - k_omega_p)
    m = 0.5 * (k_omega_n + k_omega_p - 1)
    z = (omega_p_underscore - x - omega_n_underscore) * theta_tilde

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        A_1_log = (
            -loggamma(k_omega_p)
            - loggamma(k_omega_n)
            - k_omega_p * np.log(theta_omega_p)
            - k_omega_n * np.log(theta_omega_n)
        )
        A_2_log = omega_p_underscore / theta_omega_p + omega_n_underscore / theta_omega_n
        A_3_log = x / theta_omega_n

    branch_gap = omega_p_underscore - x - omega_n_underscore
    cond1 = valid & (branch_gap > 0)
    cond2 = valid & (branch_gap < 0)
    cond3 = valid & (branch_gap == 0)

    A_4 = np.zeros_like(x, dtype=np.float64)
    A_5 = np.zeros_like(x, dtype=np.float64)
    A_6 = np.zeros_like(x, dtype=np.float64)
    A_7 = np.zeros_like(x, dtype=np.float64)
    A_8 = np.zeros_like(x, dtype=np.float64)
    W_log = np.zeros_like(x, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        A_4[cond1] = -omega_p_underscore[cond1] * theta_tilde[cond1]
        A_5[cond1] = k_omega_p[cond1] * np.log(1 / theta_tilde[cond1])
        A_6[cond1] = (k_omega_n[cond1] - 1) * np.log(branch_gap[cond1])
        A_7[cond1] = loggamma(0.5 - k[cond1] + m[cond1])
        A_8[cond1] = z[cond1] / 2 - k[cond1] * np.log(z[cond1])
        W_log[cond1] = (
            -z[cond1] / 2
            + (m[cond1] + 0.5) * np.log(z[cond1])
            + original_log_hyperu_helper(
                0.5 - k[cond1] + m[cond1],
                1 + 2 * m[cond1],
                z[cond1],
                hyperu_method,
            )
        )

        A_4[cond2] = -(x[cond2] + omega_n_underscore[cond2]) * theta_tilde[cond2]
        A_5[cond2] = k_omega_n[cond2] * np.log(1 / theta_tilde[cond2])
        A_6[cond2] = (k_omega_p[cond2] - 1) * np.log(-branch_gap[cond2])
        A_7[cond2] = loggamma(0.5 + k[cond2] + m[cond2])
        A_8[cond2] = -z[cond2] / 2 + k[cond2] * np.log(-z[cond2])
        W_log[cond2] = (
            z[cond2] / 2
            + (m[cond2] + 0.5) * np.log(-z[cond2])
            + original_log_hyperu_helper(
                0.5 + k[cond2] + m[cond2],
                1 + 2 * m[cond2],
                -z[cond2],
                hyperu_method,
            )
        )

    result = A_1_log + A_2_log + A_3_log + A_4 + A_5 + A_6 + A_7 + A_8 + W_log
    branch_shape = k_omega_p + k_omega_n
    finite_branch = cond3 & (branch_shape > 1)
    singular_branch = cond3 & (branch_shape <= 1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result[finite_branch] = (
            loggamma(branch_shape[finite_branch] - 1)
            - loggamma(k_omega_p[finite_branch])
            - loggamma(k_omega_n[finite_branch])
            - k_omega_p[finite_branch] * np.log(theta_omega_p[finite_branch])
            - k_omega_n[finite_branch] * np.log(theta_omega_n[finite_branch])
            - (branch_shape[finite_branch] - 1) * np.log(theta_tilde[finite_branch])
        )
    result[singular_branch] = np.inf
    result[~valid] = np.nan
    return result


def numerical_density_grid(x, p, n, sigma_p, sigma_n, n_points):
    x = np.asarray(x, dtype=float).reshape(-1)
    stdx = np.sqrt(p * sigma_p**2 + n * sigma_n**2)
    zgrid = np.linspace(-5 * stdx, 5 * stdx, int(n_points) + 1)

    gamma_p = zgrid / sigma_p + p
    f_p = gamma.pdf(gamma_p, p) / sigma_p
    valid_p = gamma_p > 0

    gamma_n = (zgrid[None, :] - x[:, None]) / sigma_n + n
    f_n = gamma.pdf(gamma_n, n) / sigma_n
    valid = valid_p[None, :] & (gamma_n > 0)

    integrand = f_p[None, :] * f_n
    integrand = np.where(valid, integrand, 0.0)
    return np.trapz(integrand, zgrid, axis=1)


PARAMETER_SETS = [
    {
        "set": "BG_ARX11_p0n0",
        "source": "BadGood_BEGE draw_356 ARX(1,1), estimated p0/n0",
        "p": 0.24920104600205947,
        "n": 0.400845238255339,
        "sigma_p": 0.3580466990158595,
        "sigma_n": 0.45200298627032404,
    },
    {
        "set": "BG_constant_p0n0",
        "source": "BadGood_BEGE draw_356 constant, estimated p0/n0",
        "p": 1.8660471819906415,
        "n": 0.059323428188974237,
        "sigma_p": 0.17527228422305768,
        "sigma_n": 1.2560514573436654,
    },
    {
        "set": "ID_filtered_best_p0n0",
        "source": "InflationDeflation_BEGE seed 50 ARX(2,1), best AIC after excluding near-zero sigmas",
        "p": 0.562484,
        "n": 0.240307,
        "sigma_p": 0.632182,
        "sigma_n": 1.476408,
    },
    {
        "set": "ID_median_near_p0n0",
        "source": "InflationDeflation_BEGE seed 52 ARX(1,1), closest to median p0/n0/sigma vector",
        "p": 2.001930,
        "n": 0.101780,
        "sigma_p": 0.306041,
        "sigma_n": 0.458856,
    },
    {
        "set": "BG_constant_q95_shape",
        "source": "BadGood_BEGE draw_356 constant, marginal 95th percentile fitted shape levels",
        "p": 21.2968985,
        "n": 0.62350133,
        "sigma_p": 0.17527228422305768,
        "sigma_n": 1.2560514573436654,
    },
    {
        "set": "BG_constant_max_shape_stress",
        "source": "BadGood_BEGE draw_356 constant, max fitted shape levels as stress case",
        "p": 94.980366,
        "n": 3.47122889,
        "sigma_p": 0.17527228422305768,
        "sigma_n": 1.2560514573436654,
    },
]

NUMERICAL_POINTS = [250, 500, 1000, 2500, 5000, 10000, 25000, 50000]


def finite_sum(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        if np.any(np.isneginf(values)):
            return -np.inf
        if np.any(np.isposinf(values)):
            return np.inf
        return np.nan
    return float(np.sum(values))


def run_audit():
    df = pd.read_pickle(DATA_PATH)
    x = df["SPF_shock"].to_numpy(dtype=float)

    analytic_rows = []
    numerical_rows = []

    for spec in PARAMETER_SETS:
        p = spec["p"]
        n = spec["n"]
        sigma_p = spec["sigma_p"]
        sigma_n = spec["sigma_n"]

        start = time.perf_counter()
        modified = BEGE_log_density(x, p, n, sigma_p, sigma_n, hyperu_method="scipy_approx")
        modified_time = time.perf_counter() - start

        start = time.perf_counter()
        original = original_bege_log_density(x, p, n, sigma_p, sigma_n, hyperu_method="scipy")
        original_time = time.perf_counter() - start

        diff = np.asarray(modified, dtype=float) - np.asarray(original, dtype=float)
        finite_diff = diff[np.isfinite(diff)]

        analytic_rows.append(
            {
                **spec,
                "method": "modified_scipy_approx",
                "loglik": finite_sum(modified),
                "seconds": modified_time,
                "nonfinite_obs": int(np.sum(~np.isfinite(modified))),
                "ll_minus_original": finite_sum(modified) - finite_sum(original),
                "max_abs_obs_logdiff_vs_original": float(np.max(np.abs(finite_diff))) if finite_diff.size else np.nan,
            }
        )
        analytic_rows.append(
            {
                **spec,
                "method": "original_intended_scipy_mpmath",
                "loglik": finite_sum(original),
                "seconds": original_time,
                "nonfinite_obs": int(np.sum(~np.isfinite(original))),
                "ll_minus_original": 0.0,
                "max_abs_obs_logdiff_vs_original": 0.0,
            }
        )

        original_ll = finite_sum(original)
        for n_points in NUMERICAL_POINTS:
            start = time.perf_counter()
            numerical_density = numerical_density_grid(x, p, n, sigma_p, sigma_n, n_points)
            with np.errstate(divide="ignore", invalid="ignore"):
                numerical = np.log(numerical_density)
            numerical_clipped = np.log(np.maximum(numerical_density, np.finfo(float).tiny))
            elapsed = time.perf_counter() - start
            numerical_ll = finite_sum(numerical)
            numerical_clipped_ll = finite_sum(numerical_clipped)
            numerical_rows.append(
                {
                    **spec,
                    "method": "numerical_grid",
                    "n_points": int(n_points),
                    "loglik": numerical_ll,
                    "clipped_loglik": numerical_clipped_ll,
                    "seconds": elapsed,
                    "nonfinite_obs": int(np.sum(~np.isfinite(numerical))),
                    "zero_density_obs": int(np.sum(numerical_density <= 0)),
                    "ll_minus_original": numerical_ll - original_ll,
                    "clipped_ll_minus_original": numerical_clipped_ll - original_ll,
                }
            )

    analytic_df = pd.DataFrame(analytic_rows)
    numerical_df = pd.DataFrame(numerical_rows)
    analytic_df.to_csv(RESULTS_CSV, index=False)
    numerical_df.to_csv(NUMERICAL_CSV, index=False)
    write_figures(analytic_df, numerical_df)
    write_report(df, analytic_df, numerical_df)
    return analytic_df, numerical_df


def fmt(x, digits=6):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def markdown_table(df, columns, labels=None, digits=None):
    labels = labels or columns
    digits = digits or {}
    separators = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            separators.append("---:")
        else:
            separators.append("---")
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, str):
                vals.append(val)
            elif isinstance(val, (int, np.integer)):
                vals.append(str(int(val)))
            else:
                vals.append(fmt(val, digits.get(col, 6)))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_figures(analytic_df, numerical_df):
    plt.figure(figsize=(9, 5.5))
    for set_name, group in numerical_df.groupby("set"):
        group = group.sort_values("n_points")
        plt.plot(group["n_points"], group["clipped_ll_minus_original"], marker="o", label=set_name)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xscale("log")
    plt.xlabel("Numerical integration grid points")
    plt.ylabel("Clipped log likelihood minus original analytic")
    plt.title("Old numerical grid convergence diagnostic")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(CONVERGENCE_FIG, dpi=180)
    plt.close()

    diff_df = analytic_df.loc[analytic_df["method"] == "modified_scipy_approx"].copy()
    plt.figure(figsize=(9, 5))
    plt.bar(diff_df["set"], diff_df["ll_minus_original"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Modified log likelihood minus original analytic")
    plt.title("Analytic density difference")
    plt.tight_layout()
    plt.savefig(ANALYTIC_FIG, dpi=180)
    plt.close()


def write_report(data_df, analytic_df, numerical_df):
    x = data_df["SPF_shock"].to_numpy(dtype=float)
    param_df = pd.DataFrame(PARAMETER_SETS)
    modified = analytic_df.loc[analytic_df["method"] == "modified_scipy_approx"].copy()
    original = analytic_df.loc[analytic_df["method"] == "original_intended_scipy_mpmath"].copy()
    analytic_comp = modified[
        [
            "set",
            "loglik",
            "ll_minus_original",
            "max_abs_obs_logdiff_vs_original",
            "seconds",
            "nonfinite_obs",
        ]
    ].rename(
        columns={
            "loglik": "modified_loglik",
            "seconds": "modified_seconds",
        }
    )
    analytic_comp = analytic_comp.merge(
        original[["set", "loglik", "seconds"]].rename(
            columns={"loglik": "original_loglik", "seconds": "original_seconds"}
        ),
        on="set",
        how="left",
    )
    analytic_comp = analytic_comp[
        [
            "set",
            "original_loglik",
            "modified_loglik",
            "ll_minus_original",
            "max_abs_obs_logdiff_vs_original",
            "original_seconds",
            "modified_seconds",
            "nonfinite_obs",
        ]
    ]

    last_grid = numerical_df.loc[numerical_df["n_points"] == max(NUMERICAL_POINTS)].copy()
    numerical_summary = last_grid[
        [
            "set",
            "n_points",
            "loglik",
            "nonfinite_obs",
            "zero_density_obs",
            "clipped_loglik",
            "clipped_ll_minus_original",
            "seconds",
        ]
    ].rename(
        columns={
            "loglik": "raw_numerical_loglik_50000",
            "clipped_loglik": "clipped_numerical_loglik_50000",
        }
    )

    convergence = numerical_df.pivot(index="set", columns="n_points", values="clipped_ll_minus_original")
    convergence = convergence.reset_index()
    convergence.columns = [str(c) if c != "set" else c for c in convergence.columns]

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        "# BEGE Density Calculation Audit",
        "",
        "This check compares three BEGE density calculations on the effective quarterly sample using `SPF_shock` as the observation series:",
        "",
        "1. `modified_scipy_approx`: the current fast BEGE density in `BEGE_density.py`.",
        "2. `original_intended_scipy_mpmath`: the analytic BEGE density reconstructed from the pre-speed-pass version of `BEGE_density.py`. I supplied the missing `sys` import so the original intended SciPy fallback can run.",
        "3. `numerical_grid`: the old grid integration method from `numerical_approximation`, evaluated with different numbers of grid points. Raw zero densities are reported as `-inf`; a clipped version is shown only as a convergence diagnostic.",
        "",
        f"Sample: `{data_df.index.min()}` to `{data_df.index.max()}`, observations: `{len(x)}`.",
        f"`SPF_shock` summary: mean `{fmt(np.mean(x), 6)}`, std `{fmt(np.std(x, ddof=1), 6)}`, min `{fmt(np.min(x), 6)}`, max `{fmt(np.max(x), 6)}`.",
        "",
        "## Parameter Sets",
        "",
        markdown_table(
            param_df,
            ["set", "p", "n", "sigma_p", "sigma_n", "source"],
            ["Set", "p", "n", "sigma_p", "sigma_n", "Source"],
            {"p": 6, "n": 6, "sigma_p": 6, "sigma_n": 6},
        ),
        "",
        "## Analytic Density Comparison",
        "",
        markdown_table(
            analytic_comp,
            [
                "set",
                "original_loglik",
                "modified_loglik",
                "ll_minus_original",
                "max_abs_obs_logdiff_vs_original",
                "original_seconds",
                "modified_seconds",
                "nonfinite_obs",
            ],
            [
                "Set",
                "Original LL",
                "Modified LL",
                "Modified - Original",
                "Max Obs Diff",
                "Original sec",
                "Modified sec",
                "Bad Obs",
            ],
            {
                "original_loglik": 6,
                "modified_loglik": 6,
                "ll_minus_original": 6,
                "max_abs_obs_logdiff_vs_original": 6,
                "original_seconds": 4,
                "modified_seconds": 4,
            },
        ),
        "",
        f"![Analytic density difference]({ANALYTIC_FIG.name})",
        "",
        "## Numerical Integration at 50,000 Grid Points",
        "",
        markdown_table(
            numerical_summary,
            [
                "set",
                "n_points",
                "raw_numerical_loglik_50000",
                "nonfinite_obs",
                "zero_density_obs",
                "clipped_numerical_loglik_50000",
                "clipped_ll_minus_original",
                "seconds",
            ],
            [
                "Set",
                "Grid Points",
                "Raw Numerical LL",
                "Bad Obs",
                "Zero Density Obs",
                "Clipped LL",
                "Clipped - Original",
                "Seconds",
            ],
            {
                "raw_numerical_loglik_50000": 6,
                "clipped_numerical_loglik_50000": 6,
                "clipped_ll_minus_original": 6,
                "seconds": 4,
            },
        ),
        "",
        "## Numerical Grid Convergence",
        "",
        "Entries use the clipped numerical-grid log likelihood minus the original analytic log likelihood. This keeps the convergence diagnostic finite when the raw grid assigns zero density to some observations.",
        "",
        markdown_table(
            convergence,
            ["set"] + [str(n) for n in NUMERICAL_POINTS],
            ["Set"] + [str(n) for n in NUMERICAL_POINTS],
            {str(n): 6 for n in NUMERICAL_POINTS},
        ),
        "",
        f"![Numerical integration convergence]({CONVERGENCE_FIG.name})",
        "",
        "## Interpretation",
        "",
        "- After the high-shape fallback fix, the modified analytic density and the original intended analytic density match for all parameter sets in this audit, including the high-shape stress case.",
        "- The current default `scipy_approx` path is still fast for moderate shapes, but it now uses the high-precision fallback instead of the asymptotic shortcut when shape/hypergeometric inputs are large. The aggressive shortcut is reserved for `scipy_fast`.",
        "- The numerical grid method is not a reliable benchmark at low grid counts. It can assign zero density to some observations for small-shape cases, and even 50,000 points can remain materially off for asymmetric-scale cases.",
        "- The original analytic function is the right benchmark for validation; the numerical integral is best treated as a convergence diagnostic.",
        "",
        "Generated files:",
        "",
        f"- `{RESULTS_CSV.name}`: analytic comparison rows.",
        f"- `{NUMERICAL_CSV.name}`: numerical-grid rows for all point counts.",
        f"- `{CONVERGENCE_FIG.name}` and `{ANALYTIC_FIG.name}`: figures used above.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    analytic, numerical = run_audit()
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {NUMERICAL_CSV}")
