import numpy as np
from scipy.integrate import quad

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


from scipy.stats import gamma

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
    density = np.trapezoid(integrand, z_valid)

    # Log-likelihood
    log_likelihood = np.log(density)
    return log_likelihood


from scipy.special import loggamma
from scipy.special import hyperu
import numpy as np
import sys
import mpmath as mp

mp.dps = 25

@np.vectorize(otypes=[np.float64])
def log_hyperu_helper(a, b, z, hyperu_method='scipy'):
    """
    Calculate hypergeometric U function using mpmath for higher precision.
    Vectorized version that can handle array inputs.
    """
    def compute_approximation():
        return -a * np.log(z) + a*(a+1-b)/z
        
    def compute_with_mpmath():
        a_mp = mp.mpf(float(a))
        b_mp = mp.mpf(float(b))
        z_mp = mp.mpf(float(z))
        result_float = float(mp.hyperu(a_mp, b_mp, z_mp))
        
        if result_float <= sys.float_info.min or result_float == float('inf'):
            return compute_approximation()
        return np.log(result_float)

    def compute_with_scipy():
        result = hyperu(a, b, z)
        if result <= sys.float_info.min:
            return compute_approximation()
        if np.isnan(result):
            try:
                return compute_with_mpmath()
            except:
                return result
        return np.log(result)

    try:
        if hyperu_method == 'mpmath' or b >= 40:
            return compute_with_mpmath()
        return compute_with_scipy()
    except Exception as e:
        print(f"Error in hyperu calculation for a={a}, b={b}, z={z}")
        print(f"log(hyperu_scipy): {np.log(hyperu(a, b, z))}")
        print(f"approximated log(hyperu): {compute_approximation()}")
        return hyperu(a, b, z)
            

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
    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    sigma_p = np.asarray(sigma_p, dtype=np.float64)
    sigma_n = np.asarray(sigma_n, dtype=np.float64)

    # Ensure x, p, n have the same shape
    if x.ndim == 0:
        x = np.array([x])
    if p.ndim == 0:
        p = np.array([p])
    if n.ndim == 0:
        n = np.array([n])

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

    A_1_log = -loggamma(k_omega_p) - loggamma(k_omega_n) - k_omega_p * np.log(theta_omega_p) - k_omega_n * np.log(theta_omega_n)
    A_2_log = omega_p_underscore / theta_omega_p + omega_n_underscore / theta_omega_n
    A_3_log = x / theta_omega_n
    
    # Masks
    cond1 = omega_p_underscore > x + omega_n_underscore
    cond2 = omega_p_underscore < x + omega_n_underscore
    cond3 = omega_p_underscore == x + omega_n_underscore

    # Initialize result arrays
    A_4 = np.zeros_like(x, dtype=np.float64)
    A_5 = np.zeros_like(x, dtype=np.float64)
    A_6 = np.zeros_like(x, dtype=np.float64)
    A_7 = np.zeros_like(x, dtype=np.float64)
    A_8 = np.zeros_like(x, dtype=np.float64)
    W_log = np.zeros_like(x, dtype=np.float64)

    # Fill using cond1
    A_4[cond1] = -omega_p_underscore[cond1] * theta_tilde
    A_5[cond1] = k_omega_p[cond1] * np.log(1 / theta_tilde)
    A_6[cond1] = (k_omega_n[cond1] - 1) * np.log(omega_p_underscore[cond1] - x[cond1] - omega_n_underscore[cond1])
    A_7[cond1] = loggamma(0.5 - k[cond1] + m[cond1])  # assume k, m are scalars or broadcastable
    A_8[cond1] = z[cond1] / 2 - k[cond1] * np.log(z[cond1])
    W_log[cond1] = -z[cond1]/2 + (m[cond1] + 0.5) * np.log(z[cond1]) + log_hyperu_helper(0.5 - k[cond1] + m[cond1], 1 + 2 * m[cond1], z[cond1], hyperu_method)

    # Fill using cond2
    A_4[cond2] = -(x[cond2] + omega_n_underscore[cond2]) * theta_tilde
    A_5[cond2] = k_omega_n[cond2] * np.log(1 / theta_tilde)
    A_6[cond2] = (k_omega_p[cond2] - 1) * np.log(x[cond2] + omega_n_underscore[cond2] - omega_p_underscore[cond2])
    A_7[cond2] = loggamma(0.5 + k[cond2] + m[cond2])
    A_8[cond2] = -z[cond2] / 2 + k[cond2] * np.log(-z[cond2])
    W_log[cond2] = z[cond2] / 2 + (m[cond2] + 0.5) * np.log(-z[cond2]) + log_hyperu_helper(0.5 + k[cond2] + m[cond2], 1 + 2 * m[cond2], -z[cond2], hyperu_method)

    # Final result
    result = A_1_log + A_2_log + A_3_log + A_4 + A_5 + A_6 + A_7 + A_8 + W_log
    result[cond3] = np.nan
    
    return result