import numpy as np
import mpmath as mp
from scipy.integrate import quad, trapezoid
from scipy.special import digamma, hyperu, loggamma
from scipy.stats import gamma

def characteristic_function_scalar(x, p, n, sigma_p, sigma_n, max_subinterval=500):
    stdx = np.sqrt(p * sigma_p**2 + n * sigma_n**2)
    delx = 0.001 * stdx
    fmax = 1 / delx * np.pi

    def integrand(t):
        term1 = -p * ((1j * t * sigma_p) + np.log(1 - (1j * t * sigma_p)))
        term2 = -n * ((-1j * t * sigma_n) + np.log(1 - (-1j * t * sigma_n)))
        phi_t = np.exp(term1 + term2)
        return np.real(np.exp(-1j * t * x) * phi_t)

    begepdf, _ = quad(integrand, 0, fmax, limit=max_subinterval)
    begepdf = begepdf / np.pi
    begepdf_log = np.log(begepdf)
    return begepdf_log


def numerical_approximation(x, p, n, sigma_p, sigma_n, n_points=1000):
    # Grid over ω_p
    stdx = np.sqrt(p * sigma_p**2 + n * sigma_n**2)
    zmin = -5 * stdx
    zmax = 5 * stdx
    zgrid = np.linspace(zmin, zmax, n_points + 1)

    # ω_p = σ_p (γ_p - p)
    gamma_p = zgrid / sigma_p + p
    gamma_n = (zgrid - x) / sigma_n + n  # shift to account for f_n(z - x)

    # Densities with change of variable
    f_p = gamma.pdf(gamma_p, p) / sigma_p
    f_n = gamma.pdf(gamma_n, n) / sigma_n

    # Valid gamma argument range
    valid = (gamma_p > 0) & (gamma_n > 0)
    z_valid = zgrid[valid]
    integrand = f_p[valid] * f_n[valid]

    # Trapezoidal integration
    density = trapezoid(integrand, z_valid)

    # Log-likelihood
    log_likelihood = np.log(density)
    return log_likelihood


mp.mp.dps = 35

HYPERU_TINY = np.finfo(np.float64).tiny
HYPERU_MPMATH_MAXTERMS = 1000
# BadGood rescore diagnostics show that a pointwise max(p, n) cutoff is too
# late and too discontinuous: stored job-side failures appeared with max shape
# below 180 but total shape near 200.  Use total-shape guarding plus an
# exact-vs-saddlepoint disagreement check instead of a hard max-shape cap.
SADDLEPOINT_SHAPE_THRESHOLD = 180.0
SADDLEPOINT_BLEND_TOTAL_SHAPE_LOWER = 50.0
SADDLEPOINT_BLEND_TOTAL_SHAPE_UPPER = 80.0
SADDLEPOINT_GUARD_TOTAL_SHAPE = 40.0
SADDLEPOINT_EXACT_DIFF_TOL = 2.0
NORMAL_LIMIT_TOTAL_SHAPE = 500.0
NORMAL_LIMIT_SKEWNESS = 0.03
NORMAL_LIMIT_EXCESS_KURTOSIS = 0.03


def _smoothstep(value, lower, upper):
    if upper <= lower:
        return np.where(value >= upper, 1.0, 0.0)
    scaled = np.clip((value - lower) / (upper - lower), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _log_hyperu_large_z_approximation(a, b, z):
    return -a * np.log(z) + a * (a + 1 - b) / z


def _log_hyperu_small_z_approximation(a, b, z):
    if b > 1:
        return loggamma(b - 1) - loggamma(a) + (1 - b) * np.log(z)
    if b < 1:
        return loggamma(1 - b) - loggamma(a - b + 1)

    euler_gamma = 0.5772156649015329
    leading_term = -np.log(z) - digamma(a) - 2 * euler_gamma
    if leading_term <= 0:
        return np.nan
    return np.log(leading_term) - loggamma(a)


def _log_hyperu_approximation(a, b, z):
    if z < 1e-8:
        return _log_hyperu_small_z_approximation(a, b, z)
    return _log_hyperu_large_z_approximation(a, b, z)


def _log_hyperu_mpmath_scalar(a, b, z):
    a_mp = mp.mpf(float(a))
    b_mp = mp.mpf(float(b))
    z_mp = mp.mpf(float(z))
    try:
        result = mp.hyperu(a_mp, b_mp, z_mp, maxterms=HYPERU_MPMATH_MAXTERMS)
    except Exception:
        return _log_hyperu_approximation(a, b, z)
    if result <= 0:
        return _log_hyperu_approximation(a, b, z)
    result_log = mp.log(result)
    if not mp.isfinite(result_log):
        return _log_hyperu_approximation(a, b, z)
    return float(result_log)


def _log_hyperu_helper_scalar(a, b, z, hyperu_method='scipy_approx'):
    """
    Calculate hypergeometric U function using mpmath for higher precision.
    Scalar fallback used by the array implementation below.
    """
    try:
        if hyperu_method == 'mpmath':
            return _log_hyperu_mpmath_scalar(a, b, z)

        near_integer_b = np.isclose(b, np.round(b), rtol=0.0, atol=1e-6)
        if hyperu_method == 'scipy_approx' and (a > 50.0 or b >= 40.0 or near_integer_b):
            return _log_hyperu_mpmath_scalar(a, b, z)

        result = hyperu(a, b, z)
        if result > HYPERU_TINY and np.isfinite(result):
            return np.log(result)

        if hyperu_method == 'scipy_fast':
            return _log_hyperu_approximation(a, b, z)

        return _log_hyperu_mpmath_scalar(a, b, z)
    except Exception:
        return _log_hyperu_approximation(a, b, z)


def log_hyperu_helper(a, b, z, hyperu_method='scipy_approx'):
    """
    Vectorized log U(a, b, z).

    The estimation loop calls this hundreds of thousands of times.  Use SciPy's
    array implementation first, and only pay for scalar mpmath evaluations on
    entries where SciPy underflows or returns a non-finite value.
    """
    a, b, z = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    scalar_output = a.ndim == 0
    out = np.full(a.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(z) & (z > 0)

    if hyperu_method == 'mpmath':
        flat_a = a.ravel()
        flat_b = b.ravel()
        flat_z = z.ravel()
        flat_out = out.ravel()
        for idx in np.flatnonzero(valid.ravel()):
            flat_out[idx] = _log_hyperu_helper_scalar(
                flat_a[idx], flat_b[idx], flat_z[idx], hyperu_method
            )
        return float(out) if scalar_output else out

    valid_idx = np.flatnonzero(valid.ravel())
    if valid_idx.size:
        flat_a = a.ravel()
        flat_b = b.ravel()
        flat_z = z.ravel()
        flat_out = out.ravel()
        scipy_idx = valid_idx

        if hyperu_method == 'scipy_fast':
            force_approx = (
                (flat_a[valid_idx] > 50.0)
                | (flat_b[valid_idx] > 100.0)
                | (flat_z[valid_idx] > 80.0)
            )
            approx_idx = valid_idx[force_approx]
            for idx in approx_idx:
                flat_out[idx] = _log_hyperu_approximation(flat_a[idx], flat_b[idx], flat_z[idx])
            scipy_idx = valid_idx[~force_approx]
        elif hyperu_method == 'scipy_approx':
            near_integer_b = np.isclose(flat_b[valid_idx], np.round(flat_b[valid_idx]), rtol=0.0, atol=1e-6)
            force_mpmath = (flat_a[valid_idx] > 50.0) | (flat_b[valid_idx] >= 40.0) | near_integer_b
            mpmath_idx = valid_idx[force_mpmath]
            for idx in mpmath_idx:
                flat_out[idx] = _log_hyperu_mpmath_scalar(flat_a[idx], flat_b[idx], flat_z[idx])
            scipy_idx = valid_idx[~force_mpmath]

        try:
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                values = hyperu(flat_a[scipy_idx], flat_b[scipy_idx], flat_z[scipy_idx])
                good = np.isfinite(values) & (values > HYPERU_TINY)
                flat_out[scipy_idx[good]] = np.log(values[good])
        except Exception:
            good = np.zeros(scipy_idx.size, dtype=bool)

        bad_idx = scipy_idx[~good]
        for idx in bad_idx:
            flat_out[idx] = _log_hyperu_helper_scalar(
                flat_a[idx], flat_b[idx], flat_z[idx], hyperu_method
            )

    return float(out) if scalar_output else out


def _bege_normal_log_density(x, p, n, sigma_p, sigma_n):
    variance = p * sigma_p * sigma_p + n * sigma_n * sigma_n
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return -0.5 * (np.log(2.0 * np.pi * variance) + (x * x) / variance)


def _bege_standardized_skewness_excess_kurtosis(p, n, sigma_p, sigma_n):
    variance = p * sigma_p * sigma_p + n * sigma_n * sigma_n
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        skewness = 2.0 * (p * sigma_p**3 - n * sigma_n**3) / np.power(variance, 1.5)
        excess_kurtosis = 6.0 * (p * sigma_p**4 + n * sigma_n**4) / (variance * variance)
    return skewness, excess_kurtosis


def _bege_saddlepoint_log_density(x, p, n, sigma_p, sigma_n):
    """
    Saddlepoint log density for
        X = sigma_p * (Gamma(p, 1) - p) - sigma_n * (Gamma(n, 1) - n).

    The closed-form BEGE density is fragile when the recursive shapes are
    large because it evaluates a confluent hypergeometric U term with large,
    nearly cancelling log components. The cumulant-generating function is
    simple and gives a stable high-shape approximation.
    """
    x, p, n, sigma_p, sigma_n = np.broadcast_arrays(x, p, n, sigma_p, sigma_n)
    out = np.full(x.shape, np.nan, dtype=np.float64)

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
    if not np.any(valid):
        return out

    xv = x[valid]
    pv = p[valid]
    nv = n[valid]
    spv = sigma_p[valid]
    snv = sigma_n[valid]

    lo = -1.0 / snv
    hi = 1.0 / spv
    eps = np.sqrt(np.finfo(np.float64).eps)
    lower = lo + eps * np.maximum(1.0, np.abs(lo))
    upper = hi - eps * np.maximum(1.0, np.abs(hi))

    variance = pv * spv * spv + nv * snv * snv
    t = np.clip(xv / np.maximum(variance, HYPERU_TINY), lower, upper)
    bracket_lo = lower.copy()
    bracket_hi = upper.copy()

    converged = np.zeros_like(xv, dtype=bool)
    for _ in range(80):
        denom_p = 1.0 - spv * t
        denom_n = 1.0 + snv * t
        kp = pv * spv / denom_p - nv * snv / denom_n - pv * spv + nv * snv
        kpp = pv * spv * spv / (denom_p * denom_p) + nv * snv * snv / (denom_n * denom_n)
        diff = kp - xv

        converged_now = np.abs(diff) <= 1e-10 * (1.0 + np.abs(xv))
        converged |= converged_now
        if np.all(converged):
            break

        move_right = diff < 0.0
        bracket_lo = np.where(move_right, t, bracket_lo)
        bracket_hi = np.where(move_right, bracket_hi, t)

        newton = t - diff / kpp
        midpoint = 0.5 * (bracket_lo + bracket_hi)
        use_midpoint = (~np.isfinite(newton)) | (newton <= bracket_lo) | (newton >= bracket_hi)
        t = np.where(converged, t, np.where(use_midpoint, midpoint, newton))
        t = np.clip(t, lower, upper)

    denom_p = 1.0 - spv * t
    denom_n = 1.0 + snv * t
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        k_val = (
            -pv * np.log1p(-spv * t)
            - nv * np.log1p(snv * t)
            - pv * spv * t
            + nv * snv * t
        )
        kpp = pv * spv * spv / (denom_p * denom_p) + nv * snv * snv / (denom_n * denom_n)
        saddle = k_val - t * xv - 0.5 * np.log(2.0 * np.pi * kpp)

    normal = _bege_normal_log_density(xv, pv, nv, spv, snv)
    saddle = np.where(np.isfinite(saddle), saddle, normal)
    out[valid] = saddle
    return out


def BEGE_log_density(
    x,
    p,
    n,
    sigma_p,
    sigma_n,
    hyperu_method='scipy_approx',
    saddlepoint_shape_threshold=SADDLEPOINT_SHAPE_THRESHOLD,
):
    """
    Compute the BEGE log density for a vector of parameters.

    :param x: Array of shape (k,) or scalar
    :param p: Array of shape (k,) or scalar
    :param n: Array of shape (k,) or scalar
    :param sigma_p: scalar
    :param sigma_n: scalar
    :return: log density series as numpy array
    """
    x, p, n, sigma_p, sigma_n = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
        np.asarray(n, dtype=np.float64),
        np.asarray(sigma_p, dtype=np.float64),
        np.asarray(sigma_n, dtype=np.float64),
    )

    x = np.atleast_1d(x)
    p = np.atleast_1d(p)
    n = np.atleast_1d(n)
    sigma_p = np.atleast_1d(sigma_p)
    sigma_n = np.atleast_1d(sigma_n)

    x = x.astype(np.float64, copy=False)
    p = p.astype(np.float64, copy=False)
    n = n.astype(np.float64, copy=False)
    sigma_p = sigma_p.astype(np.float64, copy=False)
    sigma_n = sigma_n.astype(np.float64, copy=False)

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
    if saddlepoint_shape_threshold is None or hyperu_method == 'mpmath':
        use_saddlepoint = np.zeros_like(valid, dtype=bool)
    else:
        use_saddlepoint = valid & (np.maximum(p, n) >= float(saddlepoint_shape_threshold))
    exact_valid = valid & ~use_saddlepoint

    k_omega_p = p
    k_omega_n = n
    theta_omega_p = sigma_p
    theta_omega_n = sigma_n
    omega_p_underscore = -k_omega_p * theta_omega_p
    omega_n_underscore = -k_omega_n * theta_omega_n
    theta_tilde = (1 / theta_omega_p + 1 / theta_omega_n)
    k = 0.5 * (k_omega_n - k_omega_p)
    m = 0.5 * (k_omega_n + k_omega_p - 1)
    z = (omega_p_underscore - x - omega_n_underscore) * theta_tilde

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        A_1_log = -loggamma(k_omega_p) - loggamma(k_omega_n) - k_omega_p * np.log(theta_omega_p) - k_omega_n * np.log(theta_omega_n)
        A_2_log = omega_p_underscore / theta_omega_p + omega_n_underscore / theta_omega_n
        A_3_log = x / theta_omega_n
    
    # Masks
    branch_gap = omega_p_underscore - x - omega_n_underscore
    cond1 = exact_valid & (branch_gap > 0)
    cond2 = exact_valid & (branch_gap < 0)
    cond3 = exact_valid & (branch_gap == 0)

    # Initialize result arrays
    A_4 = np.zeros_like(x, dtype=np.float64)
    A_5 = np.zeros_like(x, dtype=np.float64)
    A_6 = np.zeros_like(x, dtype=np.float64)
    A_7 = np.zeros_like(x, dtype=np.float64)
    A_8 = np.zeros_like(x, dtype=np.float64)
    W_log = np.zeros_like(x, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Fill using cond1
        A_4[cond1] = -omega_p_underscore[cond1] * theta_tilde[cond1]
        A_5[cond1] = k_omega_p[cond1] * np.log(1 / theta_tilde[cond1])
        A_6[cond1] = (k_omega_n[cond1] - 1) * np.log(branch_gap[cond1])
        A_7[cond1] = loggamma(0.5 - k[cond1] + m[cond1])
        A_8[cond1] = z[cond1] / 2 - k[cond1] * np.log(z[cond1])
        W_log[cond1] = -z[cond1]/2 + (m[cond1] + 0.5) * np.log(z[cond1]) + log_hyperu_helper(0.5 - k[cond1] + m[cond1], 1 + 2 * m[cond1], z[cond1], hyperu_method)

        # Fill using cond2
        A_4[cond2] = -(x[cond2] + omega_n_underscore[cond2]) * theta_tilde[cond2]
        A_5[cond2] = k_omega_n[cond2] * np.log(1 / theta_tilde[cond2])
        A_6[cond2] = (k_omega_p[cond2] - 1) * np.log(-branch_gap[cond2])
        A_7[cond2] = loggamma(0.5 + k[cond2] + m[cond2])
        A_8[cond2] = -z[cond2] / 2 + k[cond2] * np.log(-z[cond2])
        W_log[cond2] = z[cond2] / 2 + (m[cond2] + 0.5) * np.log(-z[cond2]) + log_hyperu_helper(0.5 + k[cond2] + m[cond2], 1 + 2 * m[cond2], -z[cond2], hyperu_method)

    # Final result
    result = A_1_log + A_2_log + A_3_log + A_4 + A_5 + A_6 + A_7 + A_8 + W_log
    branch_shape = k_omega_p + k_omega_n
    finite_branch = cond3 & (branch_shape > 1)
    singular_branch = cond3 & (branch_shape <= 1)

    # At the branch point x = -p*sigma_p + n*sigma_n, the density has a
    # finite limit only when p + n > 1; otherwise the BEGE density is singular.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result[finite_branch] = (
            loggamma(branch_shape[finite_branch] - 1)
            - loggamma(k_omega_p[finite_branch])
            - loggamma(k_omega_n[finite_branch])
            - k_omega_p[finite_branch] * np.log(theta_omega_p[finite_branch])
            - k_omega_n[finite_branch] * np.log(theta_omega_n[finite_branch])
            - (branch_shape[finite_branch] - 1) * np.log(theta_tilde[finite_branch])
        )
    result[singular_branch] = np.inf
    if hyperu_method != 'mpmath':
        total_shape = k_omega_p + k_omega_n
        blend_weight = _smoothstep(
            total_shape,
            SADDLEPOINT_BLEND_TOTAL_SHAPE_LOWER,
            SADDLEPOINT_BLEND_TOTAL_SHAPE_UPPER,
        )
        guard_candidate = valid & (total_shape >= SADDLEPOINT_GUARD_TOTAL_SHAPE)
        needs_saddlepoint = (
            use_saddlepoint
            | (valid & (blend_weight > 0.0))
            | guard_candidate
            | (valid & ~np.isfinite(result))
        )

        if np.any(needs_saddlepoint):
            saddle = np.full_like(result, np.nan, dtype=np.float64)
            saddle[needs_saddlepoint] = _bege_saddlepoint_log_density(
                x[needs_saddlepoint],
                p[needs_saddlepoint],
                n[needs_saddlepoint],
                sigma_p[needs_saddlepoint],
                sigma_n[needs_saddlepoint],
            )

            finite_exact = np.isfinite(result)
            finite_saddle = np.isfinite(saddle)
            large_disagreement = (
                guard_candidate
                & finite_exact
                & finite_saddle
                & (np.abs(result - saddle) > SADDLEPOINT_EXACT_DIFF_TOL)
            )

            result[valid & ~finite_exact & finite_saddle] = saddle[valid & ~finite_exact & finite_saddle]
            result[large_disagreement] = saddle[large_disagreement]
            result[use_saddlepoint & finite_saddle] = saddle[use_saddlepoint & finite_saddle]

            blend = valid & finite_saddle & (blend_weight > 0.0) & ~large_disagreement & ~use_saddlepoint
            if np.any(blend):
                full_saddle_blend = blend & (blend_weight >= 1.0)
                partial_blend = blend & ~full_saddle_blend
                result[full_saddle_blend] = saddle[full_saddle_blend]
                w = blend_weight[partial_blend]
                exact_part = result[partial_blend]
                saddle_part = saddle[partial_blend]
                result[partial_blend] = np.logaddexp(
                    np.log1p(-w) + exact_part,
                    np.log(w) + saddle_part,
                )

        skewness, excess_kurtosis = _bege_standardized_skewness_excess_kurtosis(
            p,
            n,
            sigma_p,
            sigma_n,
        )
        normal_limit = (
            valid
            & (total_shape >= NORMAL_LIMIT_TOTAL_SHAPE)
            & np.isfinite(skewness)
            & np.isfinite(excess_kurtosis)
            & (np.abs(skewness) <= NORMAL_LIMIT_SKEWNESS)
            & (excess_kurtosis <= NORMAL_LIMIT_EXCESS_KURTOSIS)
        )
        if np.any(normal_limit):
            result[normal_limit] = _bege_normal_log_density(
                x[normal_limit],
                p[normal_limit],
                n[normal_limit],
                sigma_p[normal_limit],
                sigma_n[normal_limit],
            )
    elif np.any(use_saddlepoint):
        result[use_saddlepoint] = _bege_saddlepoint_log_density(
            x[use_saddlepoint],
            p[use_saddlepoint],
            n[use_saddlepoint],
            sigma_p[use_saddlepoint],
            sigma_n[use_saddlepoint],
        )
    result[~valid] = np.nan
    
    return result
