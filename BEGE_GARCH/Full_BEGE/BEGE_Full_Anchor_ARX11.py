from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BEGE_GARCH.BEGE_Density.BEGE_density import BEGE_log_density
from BEGE_GARCH.BEGE_GARCH import (
    _make_residual_function,
    bege_implied_variance,
    bege_variance_bounds,
    gjr_recursion,
)


PARAM_NAMES = [
    "const",
    "Inflation_lag_1",
    "SPF",
    "p0",
    "n0",
    "rho_p",
    "rho_n",
    "phi_p_plus",
    "phi_p_minus",
    "phi_n_plus",
    "phi_n_minus",
    "sigma_p",
    "sigma_n",
]


TRUE_PARAMS = {
    "const": 0.08,
    "Inflation_lag_1": 0.25,
    "SPF": 0.70,
    "p0": 1.50,
    "n0": 1.30,
    "rho_p": 0.10,
    "rho_n": 0.12,
    "phi_p_plus": 0.24,
    "phi_p_minus": 0.10,
    "phi_n_plus": 0.09,
    "phi_n_minus": 0.23,
    "sigma_p": 0.25,
    "sigma_n": 0.35,
}


@dataclass(frozen=True)
class SyntheticData:
    Y: np.ndarray
    X: dict[str, np.ndarray]
    residuals: np.ndarray
    pseries: np.ndarray
    nseries: np.ndarray
    spf: np.ndarray
    inflation_lag_1: np.ndarray


def pack_params(params: dict[str, float]) -> np.ndarray:
    return np.array([params[name] for name in PARAM_NAMES], dtype=float)


def unpack_vol(theta: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in theta[3:])


def stability_margins(theta: np.ndarray, variance_bound: float) -> dict[str, float]:
    (
        p0,
        n0,
        rho_p,
        rho_n,
        phi_p_plus,
        phi_p_minus,
        phi_n_plus,
        phi_n_minus,
        sigma_p,
        sigma_n,
    ) = unpack_vol(theta)
    return {
        "p_stability_margin": 1.0 - (rho_p + 0.5 * (phi_p_plus + phi_p_minus)),
        "n_stability_margin": 1.0 - (rho_n + 0.5 * (phi_n_plus + phi_n_minus)),
        "unconditional_variance_margin": variance_bound
        - (sigma_p * sigma_p * p0 + sigma_n * sigma_n * n0),
    }


def constraints_ok(theta: np.ndarray, variance_bound: float, floor_eps: float = 1e-8) -> bool:
    if not np.all(np.isfinite(theta)):
        return False
    margins = stability_margins(theta, variance_bound)
    return (
        margins["p_stability_margin"] > floor_eps
        and margins["n_stability_margin"] > floor_eps
        and margins["unconditional_variance_margin"] >= -floor_eps
    )


def simulate_full_bege_arx11(
    n_obs: int,
    burn_in: int,
    seed: int,
    true_params: dict[str, float],
) -> SyntheticData:
    """
    Simulate pi_t from an ARX(1,1) mean with Full BEGE residuals.

    The returned sample is already burn-in trimmed and includes precomputed
    lag columns, so the estimator does not drop any additional observations.
    """
    rng = np.random.default_rng(seed)
    total = int(n_obs) + int(burn_in)
    theta = pack_params(true_params)
    (
        p0,
        n0,
        rho_p,
        rho_n,
        phi_p_plus,
        phi_p_minus,
        phi_n_plus,
        phi_n_minus,
        sigma_p,
        sigma_n,
    ) = unpack_vol(theta)

    denom_p = 1.0 - rho_p - 0.5 * (phi_p_plus + phi_p_minus)
    denom_n = 1.0 - rho_n - 0.5 * (phi_n_plus + phi_n_minus)
    if denom_p <= 0.0 or denom_n <= 0.0:
        raise ValueError("True parameters are not stable.")

    spf_mean = 0.80
    spf_ar = 0.65
    spf_sd = 0.12
    spf = np.empty(total + 1, dtype=float)
    spf[0] = spf_mean
    for t in range(1, total + 1):
        spf[t] = spf_mean + spf_ar * (spf[t - 1] - spf_mean) + rng.normal(0.0, spf_sd)

    pseries = np.empty(total + 1, dtype=float)
    nseries = np.empty(total + 1, dtype=float)
    residuals = np.empty(total + 1, dtype=float)
    inflation = np.empty(total + 1, dtype=float)

    pseries[0] = p0 / denom_p
    nseries[0] = n0 / denom_n
    inflation[0] = (true_params["const"] + true_params["SPF"] * spf_mean) / (
        1.0 - true_params["Inflation_lag_1"]
    )

    for t in range(total + 1):
        if t > 0:
            u_prev = residuals[t - 1]
            u_plus = max(u_prev, 0.0)
            u_minus = min(u_prev, 0.0)
            pseries[t] = (
                p0
                + rho_p * pseries[t - 1]
                + phi_p_plus * u_plus * u_plus / (2.0 * sigma_p * sigma_p)
                + phi_p_minus * u_minus * u_minus / (2.0 * sigma_p * sigma_p)
            )
            nseries[t] = (
                n0
                + rho_n * nseries[t - 1]
                + phi_n_plus * u_plus * u_plus / (2.0 * sigma_n * sigma_n)
                + phi_n_minus * u_minus * u_minus / (2.0 * sigma_n * sigma_n)
            )

        gamma_p = rng.gamma(shape=pseries[t], scale=1.0)
        gamma_n = rng.gamma(shape=nseries[t], scale=1.0)
        residuals[t] = sigma_p * (gamma_p - pseries[t]) - sigma_n * (gamma_n - nseries[t])

        if t > 0:
            inflation[t] = (
                true_params["const"]
                + true_params["Inflation_lag_1"] * inflation[t - 1]
                + true_params["SPF"] * spf[t]
                + residuals[t]
            )

    start = int(burn_in) + 1
    stop = start + int(n_obs)
    Y = inflation[start:stop].copy()
    inflation_lag_1 = inflation[start - 1 : stop - 1].copy()
    spf_trimmed = spf[start:stop].copy()
    X = {
        "Inflation_lag_1": inflation_lag_1,
        "SPF": spf_trimmed,
    }

    return SyntheticData(
        Y=Y,
        X=X,
        residuals=residuals[start:stop].copy(),
        pseries=pseries[start:stop].copy(),
        nseries=nseries[start:stop].copy(),
        spf=spf_trimmed,
        inflation_lag_1=inflation_lag_1,
    )


def parameter_bounds(Y: np.ndarray) -> list[tuple[float, float]]:
    ymin = float(np.min(Y))
    ymax = float(np.max(Y))
    return [
        (ymin, ymax),
        (-0.999, 0.999),
        (-10.0, 10.0),
        (1e-4, 10.0),
        (1e-4, 10.0),
        (1e-5, 0.999),
        (1e-5, 0.999),
        (1e-6, 2.0),
        (1e-6, 2.0),
        (1e-6, 2.0),
        (1e-6, 2.0),
        (1e-4, 2.0),
        (1e-4, 2.0),
    ]


def project_to_bounds(theta: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    out = np.array(theta, dtype=float, copy=True)
    for idx, (lo, hi) in enumerate(bounds):
        out[idx] = min(max(out[idx], lo + 1e-12), hi - 1e-12)
    return out


def draw_near_start(
    center: np.ndarray,
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
    jitter_scale: float,
    variance_bound: float,
) -> np.ndarray:
    width = float(jitter_scale) * np.maximum(1.0, np.abs(center))
    for shrink in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        candidate = center + rng.uniform(-width * shrink, width * shrink)
        candidate = project_to_bounds(candidate, bounds)
        if constraints_ok(candidate, variance_bound):
            return candidate
    return project_to_bounds(center, bounds)


def estimate_full_bege_arx11(
    Y: np.ndarray,
    X: dict[str, np.ndarray],
    center_params: np.ndarray,
    n_starts: int,
    jitter_scale: float,
    seed: int,
    maxiter: int,
    tol: float,
    density_hyperu_method: str,
    variance_bound: float,
    enforce_variance_bounds: bool,
    optimizer_method: str,
    include_center_start: bool,
    print_summary: bool,
) -> dict[str, object]:
    bounds = parameter_bounds(Y)
    residual_function = _make_residual_function(Y, X, "ARX(1,1)")
    rng = np.random.default_rng(seed)
    big_penalty = 1e12

    def objective(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if not constraints_ok(theta, variance_bound):
            return big_penalty

        residuals = residual_function(theta[:3])
        (
            p0,
            n0,
            rho_p,
            rho_n,
            phi_p_plus,
            phi_p_minus,
            phi_n_plus,
            phi_n_minus,
            sigma_p,
            sigma_n,
        ) = unpack_vol(theta)

        pseries = gjr_recursion(residuals, (p0, rho_p, phi_p_plus, phi_p_minus), sigma_p)
        nseries = gjr_recursion(residuals, (n0, rho_n, phi_n_plus, phi_n_minus), sigma_n)
        if (
            not np.all(np.isfinite(pseries))
            or not np.all(np.isfinite(nseries))
            or np.any(pseries <= 0.0)
            or np.any(nseries <= 0.0)
        ):
            return big_penalty

        cond_var = bege_implied_variance(pseries, nseries, sigma_p, sigma_n)
        if not np.all(np.isfinite(cond_var)) or np.any(cond_var <= 0.0):
            return big_penalty

        if enforce_variance_bounds:
            lower, upper, _ = bege_variance_bounds(residuals)
            if np.any(cond_var < lower) or np.any(cond_var > upper):
                return big_penalty

        ll = BEGE_log_density(
            residuals,
            pseries,
            nseries,
            sigma_p,
            sigma_n,
            hyperu_method=density_hyperu_method,
        )
        value = -float(np.sum(ll))
        return value if np.isfinite(value) else big_penalty

    def p_stability(theta: np.ndarray) -> float:
        return stability_margins(theta, variance_bound)["p_stability_margin"] - 1e-8

    def n_stability(theta: np.ndarray) -> float:
        return stability_margins(theta, variance_bound)["n_stability_margin"] - 1e-8

    def uncond_variance(theta: np.ndarray) -> float:
        return stability_margins(theta, variance_bound)["unconditional_variance_margin"]

    constraints = [
        {"type": "ineq", "fun": p_stability},
        {"type": "ineq", "fun": n_stability},
        {"type": "ineq", "fun": uncond_variance},
    ]

    starts = []
    if include_center_start:
        starts.append(project_to_bounds(center_params, bounds))
    starts.extend(
        draw_near_start(center_params, bounds, rng, jitter_scale, variance_bound)
        for _ in range(max(0, int(n_starts) - len(starts)))
    )
    if not starts:
        raise ValueError("n_starts must leave at least one optimizer start.")

    best_opt = None
    best_fun = np.inf
    for idx, start in enumerate(starts, start=1):
        use_slsqp = optimizer_method == "SLSQP"
        opt = minimize(
            objective,
            start,
            method=optimizer_method,
            bounds=bounds,
            constraints=constraints if use_slsqp else (),
            options={"maxiter": int(maxiter), "ftol": float(tol), "disp": False},
        )
        if print_summary:
            print(
                f"start {idx:02d}/{len(starts):02d}: "
                f"success={opt.success} negloglik={float(opt.fun):.6f}"
            )
        if (
            np.isfinite(opt.fun)
            and opt.fun < big_penalty
            and (
                best_opt is None
                or (bool(opt.success) and not bool(best_opt.success))
                or (bool(opt.success) == bool(best_opt.success) and opt.fun < best_fun)
            )
        ):
            best_opt = opt
            best_fun = float(opt.fun)

    if best_opt is None:
        raise RuntimeError("All synthetic Full BEGE starts failed.")

    params = np.asarray(best_opt.x, dtype=float)
    residuals = residual_function(params[:3])
    (
        p0,
        n0,
        rho_p,
        rho_n,
        phi_p_plus,
        phi_p_minus,
        phi_n_plus,
        phi_n_minus,
        sigma_p,
        sigma_n,
    ) = unpack_vol(params)
    pseries = gjr_recursion(residuals, (p0, rho_p, phi_p_plus, phi_p_minus), sigma_p)
    nseries = gjr_recursion(residuals, (n0, rho_n, phi_n_plus, phi_n_minus), sigma_n)
    cond_var = bege_implied_variance(pseries, nseries, sigma_p, sigma_n)
    n_obs = int(Y.shape[0])
    loglik = -best_fun

    return {
        "opt": best_opt,
        "params": params,
        "loglik": loglik,
        "AIC": 2 * len(params) - 2 * loglik,
        "BIC": np.log(n_obs) * len(params) - 2 * loglik,
        "residuals": residuals,
        "pseries": pseries,
        "nseries": nseries,
        "cond_var": cond_var,
        "bounds": bounds,
        "optimizer_method": optimizer_method,
        "include_center_start": include_center_start,
    }


def evaluate_loglik(
    Y: np.ndarray,
    X: dict[str, np.ndarray],
    theta: np.ndarray,
    density_hyperu_method: str,
) -> float:
    residual_function = _make_residual_function(Y, X, "ARX(1,1)")
    residuals = residual_function(theta[:3])
    (
        p0,
        n0,
        rho_p,
        rho_n,
        phi_p_plus,
        phi_p_minus,
        phi_n_plus,
        phi_n_minus,
        sigma_p,
        sigma_n,
    ) = unpack_vol(theta)
    pseries = gjr_recursion(residuals, (p0, rho_p, phi_p_plus, phi_p_minus), sigma_p)
    nseries = gjr_recursion(residuals, (n0, rho_n, phi_n_plus, phi_n_minus), sigma_n)
    ll = BEGE_log_density(
        residuals,
        pseries,
        nseries,
        sigma_p,
        sigma_n,
        hyperu_method=density_hyperu_method,
    )
    return float(np.sum(ll))


def build_results_frame(theta_true: np.ndarray, theta_hat: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "parameter": PARAM_NAMES,
            "true": theta_true,
            "estimate": theta_hat,
        }
    )
    df["error"] = df["estimate"] - df["true"]
    df["abs_error"] = np.abs(df["error"])
    df["relative_abs_error"] = df["abs_error"] / np.maximum(np.abs(df["true"]), 1e-12)
    return df


def format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def write_markdown_report(
    path: Path,
    csv_path: Path,
    results: pd.DataFrame,
    fit: dict[str, object],
    theta_true: np.ndarray,
    true_loglik: float,
    n_obs: int,
    burn_in: int,
    seed: int,
    n_starts: int,
    jitter_scale: float,
    include_center_start: bool,
    variance_bound: float,
    enforce_variance_bounds: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        csv_display_path = csv_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        csv_display_path = csv_path
    margins = stability_margins(theta_true, variance_bound)
    estimate_margins = stability_margins(np.asarray(fit["params"], dtype=float), variance_bound)
    max_abs_error = float(results["abs_error"].max())
    max_rel_error = float(results["relative_abs_error"].max())
    opt = fit["opt"]
    lower, upper, _ = bege_variance_bounds(np.asarray(fit["residuals"], dtype=float))
    cond_var = np.asarray(fit["cond_var"], dtype=float)
    estimate_variance_bounds_ok = bool(
        np.all(np.isfinite(cond_var))
        and np.all(cond_var >= lower)
        and np.all(cond_var <= upper)
    )

    table = results.copy()
    for col in ["true", "estimate", "error", "abs_error", "relative_abs_error"]:
        table[col] = table[col].map(lambda value: format_float(value, 8))

    lines = [
        "```{raw:typst}",
        "#set page(margin: auto)",
        "```",
        "",
        "# Full BEGE Synthetic Recovery Check",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "This report is produced by `BEGE_GARCH/Full_BEGE/BEGE_Full_Anchor_ARX11.py`.",
        "",
        "## Simulation Design",
        "",
        "The synthetic sample uses the ARX(1,1) mean process",
        "",
        "$$",
        "\\pi_t = c + \\rho_1 \\pi_{t-1} + \\phi_1 SPF_t + u_t.",
        "$$",
        "",
        "The residual is generated from the Full BEGE recursion",
        "",
        "$$",
        "u_t = \\sigma_p (G_{p,t} - p_t) - \\sigma_n (G_{n,t} - n_t),",
        "$$",
        "",
        "where `G_{p,t}` is drawn from `Gamma(shape=p_t, scale=1)` and "
        "`G_{n,t}` is drawn independently from `Gamma(shape=n_t, scale=1)`. "
        "The centered gamma draws have conditional mean zero, so the residual "
        "definition remains `u_t = pi_t - hat(pi)_t`.",
        "",
        "The shape states follow",
        "",
        "$$",
        "\\begin{aligned}",
        "p_t &= p_0 + \\rho_p p_{t-1}"
        " + \\frac{\\phi_p^+}{2\\sigma_p^2}(u_{t-1}^+)^2"
        " + \\frac{\\phi_p^-}{2\\sigma_p^2}(u_{t-1}^-)^2,\\\\",
        "n_t &= n_0 + \\rho_n n_{t-1}"
        " + \\frac{\\phi_n^+}{2\\sigma_n^2}(u_{t-1}^+)^2"
        " + \\frac{\\phi_n^-}{2\\sigma_n^2}(u_{t-1}^-)^2.",
        "\\end{aligned}",
        "$$",
        "",
        f"The simulation draws `{n_obs}` observations after a burn-in of `{burn_in}` observations. "
        f"The random seed is `{seed}`. The SPF process is an exogenous stationary AR(1) process "
        "with mean `0.80`, autoregressive coefficient `0.65`, and innovation standard deviation `0.12`.",
        "",
        "## Estimation Setup",
        "",
        f"The estimator maximizes the same Full BEGE log likelihood used in the project code. "
        f"It runs `{n_starts}` `{fit.get('optimizer_method', 'unknown')}` starts centered at "
        f"the true parameter vector with jitter scale "
        f"`{jitter_scale}`. Exact true-parameter start included: `{include_center_start}`. "
        f"Stability constraints and the unconditional variance bound "
        f"`sigma_p^2 p0 + sigma_n^2 n0 <= {variance_bound}` are imposed by the objective "
        "feasibility screen.",
        "",
        f"EWMA implied-variance bounds are {'enforced' if enforce_variance_bounds else 'not enforced'} "
        "during this synthetic recovery run. The final estimate is still rechecked against those bounds.",
        "",
        "The true parameters were chosen away from the optimizer bounds and with comfortable "
        "stability margins:",
        "",
        f"- True p-process stability margin: `{format_float(margins['p_stability_margin'])}`",
        f"- True n-process stability margin: `{format_float(margins['n_stability_margin'])}`",
        f"- True unconditional variance margin: `{format_float(margins['unconditional_variance_margin'])}`",
        "",
        "## Estimation Results",
        "",
        table.to_markdown(index=False),
        "",
        "## Likelihood and Diagnostics",
        "",
        f"- Optimizer success: `{bool(opt.success)}`",
        f"- Optimizer message: `{opt.message}`",
        f"- Log likelihood at estimate: `{format_float(fit['loglik'])}`",
        f"- Log likelihood at true parameters: `{format_float(true_loglik)}`",
        f"- AIC at estimate: `{format_float(fit['AIC'])}`",
        f"- BIC at estimate: `{format_float(fit['BIC'])}`",
        f"- Maximum absolute parameter error: `{format_float(max_abs_error, 8)}`",
        f"- Maximum relative absolute parameter error: `{format_float(max_rel_error, 8)}`",
        f"- Estimated p-process stability margin: `{format_float(estimate_margins['p_stability_margin'])}`",
        f"- Estimated n-process stability margin: `{format_float(estimate_margins['n_stability_margin'])}`",
        f"- Estimated unconditional variance margin: `{format_float(estimate_margins['unconditional_variance_margin'])}`",
        f"- Estimated EWMA implied-variance-bound check: `{estimate_variance_bounds_ok}`",
        f"- Maximum estimated shape state: `{format_float(max(np.max(fit['pseries']), np.max(fit['nseries'])), 6)}`",
        f"- Maximum estimated implied variance: `{format_float(np.max(fit['cond_var']), 6)}`",
        "",
        f"The parameter comparison CSV is written to `{csv_display_path}`.",
        "",
        "A finite random sample does not make the MLE exactly equal to the data-generating "
        "parameters. This run is a recovery check: with a long sample and a well-conditioned "
        "interior parameter vector, the estimate should be close to the true vector and the "
        "sample likelihood at the estimate should be at least as high as the likelihood at the truth.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_report = SCRIPT_DIR / "results" / "synthetic_full_arx11_recovery.md"
    default_csv = SCRIPT_DIR / "results" / "synthetic_full_arx11_recovery.csv"

    parser = argparse.ArgumentParser(
        description="Synthetic ARX(1,1) Full BEGE recovery experiment."
    )
    parser.add_argument("--n-obs", type=int, default=20000, help="Post-burn-in sample length")
    parser.add_argument("--burn-in", type=int, default=5000, help="Simulation burn-in length")
    parser.add_argument("--seed", type=int, default=20261208, help="Simulation seed")
    parser.add_argument("--estimation-seed", type=int, default=20261209, help="Start jitter seed")
    parser.add_argument("--n-starts", type=int, default=3, help="Number of true-centered starts")
    parser.add_argument("--jitter-scale", type=float, default=0.03, help="Near-start jitter scale")
    parser.add_argument("--maxiter", type=int, default=180, help="Optimizer iteration limit")
    parser.add_argument("--tol", type=float, default=1e-9, help="Optimizer tolerance")
    parser.add_argument(
        "--density-hyperu-method",
        choices=["scipy_approx", "scipy_fast", "mpmath"],
        default="scipy_fast",
        help="BEGE density backend",
    )
    parser.add_argument(
        "--optimizer-method",
        choices=["L-BFGS-B", "SLSQP"],
        default="L-BFGS-B",
        help="Optimizer for the synthetic recovery MLE.",
    )
    parser.add_argument(
        "--include-center-start",
        action="store_true",
        help="Include the exact true parameter vector as one optimizer start.",
    )
    parser.add_argument(
        "--variance-bound",
        type=float,
        default=0.75,
        help="Unconditional variance bound used by Full BEGE.",
    )
    parser.add_argument(
        "--enforce-variance-bounds",
        action="store_true",
        help="Impose EWMA implied-variance bounds in the synthetic objective. "
        "The final estimate is rechecked either way.",
    )
    parser.add_argument(
        "--no-variance-bounds",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--report-path", type=Path, default=default_report)
    parser.add_argument("--csv-path", type=Path, default=default_csv)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-start optimizer output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theta_true = pack_params(TRUE_PARAMS)
    if not constraints_ok(theta_true, args.variance_bound):
        raise ValueError("Configured true parameters do not satisfy Full BEGE constraints.")

    data = simulate_full_bege_arx11(
        n_obs=args.n_obs,
        burn_in=args.burn_in,
        seed=args.seed,
        true_params=TRUE_PARAMS,
    )
    enforce_variance_bounds = bool(args.enforce_variance_bounds and not args.no_variance_bounds)
    fit = estimate_full_bege_arx11(
        Y=data.Y,
        X=data.X,
        center_params=theta_true,
        n_starts=args.n_starts,
        jitter_scale=args.jitter_scale,
        seed=args.estimation_seed,
        maxiter=args.maxiter,
        tol=args.tol,
        density_hyperu_method=args.density_hyperu_method,
        variance_bound=args.variance_bound,
        enforce_variance_bounds=enforce_variance_bounds,
        optimizer_method=args.optimizer_method,
        include_center_start=args.include_center_start,
        print_summary=not args.quiet,
    )
    theta_hat = np.asarray(fit["params"], dtype=float)
    results = build_results_frame(theta_true, theta_hat)

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.csv_path, index=False)

    true_loglik = evaluate_loglik(
        data.Y,
        data.X,
        theta_true,
        density_hyperu_method=args.density_hyperu_method,
    )
    write_markdown_report(
        path=args.report_path,
        csv_path=args.csv_path,
        results=results,
        fit=fit,
        theta_true=theta_true,
        true_loglik=true_loglik,
        n_obs=args.n_obs,
        burn_in=args.burn_in,
        seed=args.seed,
        n_starts=args.n_starts,
        jitter_scale=args.jitter_scale,
        include_center_start=args.include_center_start,
        variance_bound=args.variance_bound,
        enforce_variance_bounds=enforce_variance_bounds,
    )

    print(f"Wrote parameter comparison: {args.csv_path}")
    print(f"Wrote markdown report: {args.report_path}")
    print(
        "Max abs error: "
        f"{float(results['abs_error'].max()):.8f}; "
        f"loglik estimate={float(fit['loglik']):.6f}; "
        f"loglik truth={true_loglik:.6f}"
    )


if __name__ == "__main__":
    main()
