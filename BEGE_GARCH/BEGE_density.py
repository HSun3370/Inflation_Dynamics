import numpy as np
import mpmath as mp
from scipy.integrate import quad
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
    density = np.trapz(integrand, z_valid)

    # Log-likelihood
    log_likelihood = np.log(density)
    return log_likelihood


mp.dps = 25

HYPERU_INTEGER_B_TOL = 1e-6


def _log_hyperu_helper_scalar(a, b, z, hyperu_method='scipy'):
    """
    Calculate hypergeometric U function using mpmath for higher precision.
    Vectorized version that can handle array inputs.
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
        if hyperu_method == 'mpmath' or b >= 40 or near_integer_b:
            return compute_with_mpmath()
        return compute_with_scipy()
    except Exception:
        return compute_approximation()


log_hyperu_helper = np.vectorize(_log_hyperu_helper_scalar, otypes=[np.float64])
            

def BEGE_log_density(x, p, n, sigma_p, sigma_n, hyperu_method='scipy'):
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
    cond1 = valid & (branch_gap > 0)
    cond2 = valid & (branch_gap < 0)
    cond3 = valid & (branch_gap == 0)

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
    result[~valid] = np.nan
    
    return result
