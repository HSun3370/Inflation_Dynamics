
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize 
from statsmodels.tools.numdiff import approx_fprime,approx_hess
from scipy.optimize import differential_evolution, minimize
import statsmodels.api as sm
import importlib 
from scipy.stats import gamma
import matplotlib.pyplot as plt
from itertools import product
try:
    from arch import arch_model
    from arch.univariate import ARX
except ImportError:
    arch_model = None
    ARX = None
from scipy.integrate import quad
from BEGE_GARCH.BEGE_Density.BEGE_density import *
from joblib import Parallel, delayed

# mean model
def _to_1d_float_array(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-d array-like object.")
    return arr


def _coerce_mean_inputs(Y, X):
    """
    Normalize mean-model inputs.
    Supports:
      1) legacy usage: Y as inflation, X as SPF series
      2) structured usage: X as dict/DataFrame with precomputed lag columns
    """
    y = _to_1d_float_array(Y, "Y")
    if X is None:
        return y, {}

    if isinstance(X, pd.DataFrame):
        raw = {k: X[k].to_numpy(dtype=float) for k in X.columns}
    elif isinstance(X, dict):
        raw = {k: np.asarray(v, dtype=float) for k, v in X.items()}
    else:
        raw = {"SPF": _to_1d_float_array(X, "X")}

    alias = {
        "Forecasted inflation": "SPF",
        "SPF_t": "SPF",
        "Inflation.Lag(1)": "Inflation_lag_1",
        "Inflation.Lag(2)": "Inflation_lag_2",
        "SPF.lag(1)": "SPF_lag_1",
    }

    data = {}
    for key, values in raw.items():
        canon = alias.get(key, key)
        arr = _to_1d_float_array(values, canon)
        if arr.shape[0] != y.shape[0]:
            raise ValueError(f"Length mismatch between Y ({y.shape[0]}) and {canon} ({arr.shape[0]}).")
        data[canon] = arr

    return y, data


def mean_const(Y, X, params):
    y, _ = _coerce_mean_inputs(Y, X)
    return y


def mean_ARX11(Y, X, params):
    c, rho_1, phi_1 = params
    y, data = _coerce_mean_inputs(Y, X)

    if {"Inflation_lag_1", "SPF"}.issubset(data):
        y_pred = c + rho_1 * data["Inflation_lag_1"] + phi_1 * data["SPF"]
    elif "SPF" in data:
        # Legacy fallback when lag columns are not supplied.
        y_pred = np.empty_like(y)
        y_pred[0] = c + phi_1 * data["SPF"][0]
        y_pred[1:] = c + rho_1 * y[:-1] + phi_1 * data["SPF"][1:]
    else:
        raise ValueError("ARX(1,1) requires SPF in X.")

    return y - y_pred


def mean_ARX21(Y, X, params):
    c, rho_1, rho_2, phi_1 = params
    y, data = _coerce_mean_inputs(Y, X)

    if {"Inflation_lag_1", "Inflation_lag_2", "SPF"}.issubset(data):
        y_pred = (
            c
            + rho_1 * data["Inflation_lag_1"]
            + rho_2 * data["Inflation_lag_2"]
            + phi_1 * data["SPF"]
        )
    elif "SPF" in data:
        # Legacy fallback when lag columns are not supplied.
        y_pred = np.empty_like(y)
        y_pred[0] = c + phi_1 * data["SPF"][0]
        y_pred[1] = c + rho_1 * y[0] + phi_1 * data["SPF"][1]
        y_pred[2:] = c + rho_1 * y[1:-1] + rho_2 * y[:-2] + phi_1 * data["SPF"][2:]
    else:
        raise ValueError("ARX(2,1) requires SPF in X.")

    return y - y_pred


def mean_ARX22(Y, X, params):
    c, rho_1, rho_2, phi_1, phi_2 = params
    y, data = _coerce_mean_inputs(Y, X)

    if {"Inflation_lag_1", "Inflation_lag_2", "SPF", "SPF_lag_1"}.issubset(data):
        y_pred = (
            c
            + rho_1 * data["Inflation_lag_1"]
            + rho_2 * data["Inflation_lag_2"]
            + phi_1 * data["SPF"]
            + phi_2 * data["SPF_lag_1"]
        )
    elif "SPF" in data:
        # Legacy fallback when lag columns are not supplied.
        y_pred = np.empty_like(y)
        y_pred[0] = c + phi_1 * data["SPF"][0]
        y_pred[1] = c + rho_1 * y[0] + phi_1 * data["SPF"][1] + phi_2 * data["SPF"][0]
        y_pred[2:] = (
            c
            + rho_1 * y[1:-1]
            + rho_2 * y[:-2]
            + phi_1 * data["SPF"][2:]
            + phi_2 * data["SPF"][1:-1]
        )
    else:
        raise ValueError("ARX(2,2) requires SPF in X.")

    return y - y_pred


def _mean_x_len(X):
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        return len(X)
    if isinstance(X, dict):
        return min((len(v) for v in X.values()), default=None)
    return len(X)


def _slice_mean_x(X, n):
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        return X.iloc[:n].copy()
    if isinstance(X, dict):
        return {k: np.asarray(v, dtype=float)[:n] for k, v in X.items()}
    return np.asarray(X, dtype=float)[:n]


def _align_mean_sample(Y, X):
    y = np.asarray(Y, dtype=float)
    x_len = _mean_x_len(X)
    if x_len is None:
        return y, None
    n = min(len(y), x_len)
    return y[:n], _slice_mean_x(X, n)


def _make_residual_function(Y, X, mean_type):
    """
    Pre-normalize mean inputs once so likelihood objectives do not rebuild
    dicts/DataFrames and lag arrays on every optimizer evaluation.
    """
    y, data = _coerce_mean_inputs(Y, X)

    if mean_type == 'constant':
        return lambda params: y

    if mean_type == 'ARX(1,1)':
        if {"Inflation_lag_1", "SPF"}.issubset(data):
            inflation_lag_1 = data["Inflation_lag_1"]
            spf = data["SPF"]

            def _resids(params):
                c, rho_1, phi_1 = params
                return y - (c + rho_1 * inflation_lag_1 + phi_1 * spf)
            return _resids

        if "SPF" in data:
            spf = data["SPF"]

            def _resids(params):
                c, rho_1, phi_1 = params
                y_pred = np.empty_like(y)
                y_pred[0] = c + phi_1 * spf[0]
                y_pred[1:] = c + rho_1 * y[:-1] + phi_1 * spf[1:]
                return y - y_pred
            return _resids

    if mean_type == 'ARX(2,1)':
        if {"Inflation_lag_1", "Inflation_lag_2", "SPF"}.issubset(data):
            inflation_lag_1 = data["Inflation_lag_1"]
            inflation_lag_2 = data["Inflation_lag_2"]
            spf = data["SPF"]

            def _resids(params):
                c, rho_1, rho_2, phi_1 = params
                return y - (
                    c
                    + rho_1 * inflation_lag_1
                    + rho_2 * inflation_lag_2
                    + phi_1 * spf
                )
            return _resids

        if "SPF" in data:
            spf = data["SPF"]

            def _resids(params):
                c, rho_1, rho_2, phi_1 = params
                y_pred = np.empty_like(y)
                y_pred[0] = c + phi_1 * spf[0]
                y_pred[1] = c + rho_1 * y[0] + phi_1 * spf[1]
                y_pred[2:] = c + rho_1 * y[1:-1] + rho_2 * y[:-2] + phi_1 * spf[2:]
                return y - y_pred
            return _resids

    if mean_type == 'ARX(2,2)':
        if {"Inflation_lag_1", "Inflation_lag_2", "SPF", "SPF_lag_1"}.issubset(data):
            inflation_lag_1 = data["Inflation_lag_1"]
            inflation_lag_2 = data["Inflation_lag_2"]
            spf = data["SPF"]
            spf_lag_1 = data["SPF_lag_1"]

            def _resids(params):
                c, rho_1, rho_2, phi_1, phi_2 = params
                return y - (
                    c
                    + rho_1 * inflation_lag_1
                    + rho_2 * inflation_lag_2
                    + phi_1 * spf
                    + phi_2 * spf_lag_1
                )
            return _resids

        if "SPF" in data:
            spf = data["SPF"]

            def _resids(params):
                c, rho_1, rho_2, phi_1, phi_2 = params
                y_pred = np.empty_like(y)
                y_pred[0] = c + phi_1 * spf[0]
                y_pred[1] = c + rho_1 * y[0] + phi_1 * spf[1] + phi_2 * spf[0]
                y_pred[2:] = (
                    c
                    + rho_1 * y[1:-1]
                    + rho_2 * y[:-2]
                    + phi_1 * spf[2:]
                    + phi_2 * spf[1:-1]
                )
                return y - y_pred
            return _resids

    raise ValueError(f"{mean_type} has incompatible X inputs.")


# def gjr_recursion(resids, params,sigma):
#     '''
#     input
#     -----------
#     resids: array of residuals u_t from mean model
#     cont: p_o or n_o
#     rho: AR parameters
#     phi_p: positive MA paramters
#     phi_n: Negative MA parameters
    
#     Return
#     -------------------
#     sprocess: shape process
#     ''' 
#     cont, rho, phi_p, phi_n = params
#     t1 = 10e-5
#     n = len(resids)
#     sprocess = np.zeros(n)
#     backcast = max(cont / (1 - rho - (phi_p + phi_n) / 2), t1) 
#     sprocess[0] = max(backcast, t1)
    
#     for t in range(1, n):
#         sprocess[t] = cont + rho * sprocess[t-1] + ((phi_p if resids[t-1] > 0 else phi_n) * resids[t-1] ** 2 / (2 * sigma ** 2))
#         sprocess[t] = max(sprocess[t], t1)
    
#     return sprocess







try:
    from numba import njit
except Exception:
    def njit(*decorator_args, **decorator_kwargs):
        def _decorator(func):
            return func
        return _decorator


@njit(cache=True, fastmath=True, nogil=True)
def _gjr_recursion_numba_core(r, cont, rho, phi_p, phi_n, sigma):
    """
    Numba-compiled core. r must be float64 and contiguous.
    """
    n = r.shape[0]
    s = np.empty(n, dtype=np.float64)

    # constants & precomputations
    t1 = 1e-4                  # == 10e-5 in your code
    inv_den = 1.0 / (2.0 * sigma * sigma)

    # backcast (guard against near-explosive denominator)
    denom = 1.0 - rho - 0.5 * (phi_p + phi_n)
    backcast = t1 if denom <= 1e-12 else cont / denom
    if backcast < t1:
        backcast = t1
    s[0] = backcast

    # main recursion
    for t in range(1, n):
        # choose phi by sign of r[t-1]
        phi = phi_p if r[t-1] > 0.0 else phi_n
        incr = (r[t-1] * r[t-1]) * phi * inv_den
        val = cont + rho * s[t-1] + incr
        s[t] = val if val > t1 else t1

    return s

def gjr_recursion(resids, params, sigma):
    """
    Drop-in replacement for your original function.
    Uses the Numba-compiled core above.
    """
    cont, rho, phi_p, phi_n = params
    r = np.ascontiguousarray(resids, dtype=np.float64)
    return _gjr_recursion_numba_core(r, float(cont), float(rho), float(phi_p), float(phi_n), float(sigma))


BEGE_VARIANCE_EWMA_LAMBDA = 0.94
BEGE_VARIANCE_EWMA_TAU = 75


@njit(cache=True, nogil=True)
def _bege_variance_bounds_ok_core(r, pseries, nseries, sigma_p, sigma_n, lam, tau):
    n_obs = r.shape[0]
    if n_obs == 0 or pseries.shape[0] != n_obs or nseries.shape[0] != n_obs:
        return False

    sum_r = 0.0
    max_resid_sq = 0.0
    for i in range(n_obs):
        ri = r[i]
        if not np.isfinite(ri):
            return False
        sum_r += ri
        sq_i = ri * ri
        if sq_i > max_resid_sq:
            max_resid_sq = sq_i

    mean_r = sum_r / n_obs
    sample_var = 0.0
    for i in range(n_obs):
        diff = r[i] - mean_r
        sample_var += diff * diff
    sample_var /= n_obs

    tau_eff = tau
    if tau_eff < 1:
        tau_eff = 1
    if tau_eff > n_obs:
        tau_eff = n_obs

    weight = 1.0
    weight_sum = 0.0
    weighted_sq_sum = 0.0
    for i in range(tau_eff):
        weighted_sq_sum += weight * r[i] * r[i]
        weight_sum += weight
        weight *= lam
    initial_value = weighted_sq_sum / weight_sum

    tiny = 2.2250738585072014e-308
    if (not np.isfinite(initial_value)) or initial_value <= 0.0:
        initial_value = sample_var if sample_var > 0.0 else tiny

    lower_floor = sample_var / 1e8
    upper_cap = 1e7 * (1.0 + max_resid_sq)
    min_upper = 1.0 + max_resid_sq
    one_minus_lam = 1.0 - lam
    ewma = initial_value if initial_value > tiny else tiny

    sigp2 = sigma_p * sigma_p
    sign2 = sigma_n * sigma_n
    for t in range(n_obs):
        if t > 0:
            ewma = lam * ewma + one_minus_lam * r[t - 1] * r[t - 1]
            if (not np.isfinite(ewma)) or ewma <= 0.0:
                ewma = tiny

        pt = pseries[t]
        nt = nseries[t]
        if (not np.isfinite(pt)) or (not np.isfinite(nt)):
            return False

        cond_var = sigp2 * pt + sign2 * nt
        if not np.isfinite(cond_var):
            return False

        lower = ewma / 1e6
        if lower < lower_floor:
            lower = lower_floor

        upper = ewma * 1e6
        if upper > upper_cap:
            upper = upper_cap
        if upper < min_upper:
            upper = min_upper
        if upper < lower:
            upper = lower

        if cond_var < lower or cond_var > upper:
            return False

    return True


def _resolve_start_n_jobs(n_starts, start_n_jobs=None):
    if start_n_jobs is None:
        raw = os.environ.get("BEGE_START_N_JOBS") or os.environ.get("SLURM_CPUS_PER_TASK")
    else:
        raw = start_n_jobs
    try:
        n_jobs = int(raw)
    except (TypeError, ValueError):
        n_jobs = 1
    n_jobs = max(1, n_jobs)
    return min(n_jobs, max(1, int(n_starts)))


def _run_multistart_minimize(
    start_values,
    objective,
    *,
    method,
    bounds,
    constraints=None,
    options=None,
    start_n_jobs=None,
):
    start_values = [np.asarray(init, dtype=float) for init in start_values]
    n_jobs = _resolve_start_n_jobs(len(start_values), start_n_jobs=start_n_jobs)
    constraints_arg = () if constraints is None else constraints
    options_arg = {} if options is None else dict(options)

    def _run_one(init):
        try:
            opt = minimize(
                objective,
                init,
                method=method,
                bounds=bounds,
                constraints=constraints_arg,
                options=options_arg,
            )
            return opt, None
        except Exception as exc:
            return None, exc

    if n_jobs == 1 or len(start_values) <= 1:
        return [_run_one(init) for init in start_values]

    backend = os.environ.get("BEGE_START_BACKEND", "loky")
    return Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_run_one)(init) for init in start_values
    )


def bege_implied_variance(pseries, nseries, sigma_p, sigma_n):
    pseries = np.asarray(pseries, dtype=float)
    nseries = np.asarray(nseries, dtype=float)
    return float(sigma_p) * float(sigma_p) * pseries + float(sigma_n) * float(sigma_n) * nseries


def bege_variance_bounds(
    resids,
    lam=BEGE_VARIANCE_EWMA_LAMBDA,
    tau=BEGE_VARIANCE_EWMA_TAU,
):
    """
    Arch-style EWMA variance bounds used to screen BEGE implied variance paths.
    """
    r = np.asarray(resids, dtype=float).reshape(-1)
    if r.size == 0:
        raise ValueError("resids must contain at least one observation.")
    if not np.all(np.isfinite(r)):
        raise ValueError("resids must be finite to compute variance bounds.")

    sq = r * r
    n_obs = r.size
    tau_eff = int(min(max(1, tau), n_obs))
    weights = float(lam) ** np.arange(tau_eff, dtype=float)
    weights = weights / np.sum(weights)

    initial_value = float(np.dot(weights, sq[:tau_eff]))
    sample_var = float(np.var(r))
    tiny = np.finfo(float).tiny
    if (not np.isfinite(initial_value)) or initial_value <= 0.0:
        initial_value = sample_var if sample_var > 0.0 else tiny

    ewma = np.empty(n_obs, dtype=float)
    ewma[0] = max(initial_value, tiny)
    one_minus_lam = 1.0 - float(lam)
    for t in range(1, n_obs):
        ewma[t] = float(lam) * ewma[t - 1] + one_minus_lam * sq[t - 1]
        if (not np.isfinite(ewma[t])) or ewma[t] <= 0.0:
            ewma[t] = tiny

    max_resid_sq = float(np.max(sq))
    lower_floor = sample_var / 1e8
    upper_cap = 1e7 * (1.0 + max_resid_sq)
    min_upper = 1.0 + max_resid_sq

    lower = np.maximum(ewma / 1e6, lower_floor)
    upper = np.minimum(ewma * 1e6, upper_cap)
    upper = np.maximum(upper, min_upper)
    upper = np.maximum(upper, lower)
    return lower, upper, ewma


def bege_variance_bounds_ok(
    resids,
    pseries,
    nseries,
    sigma_p,
    sigma_n,
    *,
    return_details=False,
):
    if not return_details:
        r = np.ascontiguousarray(resids, dtype=np.float64).reshape(-1)
        p = np.ascontiguousarray(pseries, dtype=np.float64).reshape(-1)
        n = np.ascontiguousarray(nseries, dtype=np.float64).reshape(-1)
        return bool(
            _bege_variance_bounds_ok_core(
                r,
                p,
                n,
                float(sigma_p),
                float(sigma_n),
                float(BEGE_VARIANCE_EWMA_LAMBDA),
                int(BEGE_VARIANCE_EWMA_TAU),
            )
        )

    lower, upper, ewma = bege_variance_bounds(resids)
    cond_var = bege_implied_variance(pseries, nseries, sigma_p, sigma_n)
    finite = (
        np.all(np.isfinite(cond_var))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
    )
    ok = bool(finite and np.all(cond_var >= lower) and np.all(cond_var <= upper))
    if not return_details:
        return ok
    return {
        "ok": ok,
        "cond_var": cond_var,
        "lower": lower,
        "upper": upper,
        "ewma": ewma,
        "min_margin": float(np.min(cond_var - lower)) if finite else np.nan,
        "max_margin": float(np.min(upper - cond_var)) if finite else np.nan,
    }


def _optimizer_result_eligible(opt):
    return (
        bool(getattr(opt, "success", False))
        and np.isfinite(getattr(opt, "fun", np.nan))
        and np.all(np.isfinite(getattr(opt, "x", np.array([np.nan]))))
    )



def loglikedgam_constant(resids, p, n, sigma_p, sigma_n, hyperu_method='scipy_approx'):
    resids = np.asarray(resids)
    p = np.full_like(resids, p, dtype=np.double)
    n = np.full_like(resids, n, dtype=np.double)
    return BEGE_log_density(resids, p, n, sigma_p, sigma_n, hyperu_method=hyperu_method)



 




def BEGE_GARCH(
    Y,
    X=None,
    mean_type='constant',
    init_value=None,
    compute_se=False,
    density_hyperu_method='scipy_approx',
):
    Y, X = _align_mean_sample(Y, X)
    # mean model specification
    # initial values of mean model are the estimated values from previous GARCH model
    if mean_type=='constant':
        mean_model = mean_const
        num_param_mean = 0
        sv_mean=[] 
        bounds_mean=[]
        name_mean = []
    elif mean_type=='ARX(1,1)':
        mean_model = mean_ARX11
        num_param_mean = 3
        sv_mean=[0.0752,0.1650,0.8198]
        bounds_mean=[(min(Y),max(Y)),(-0.999,0.999),(None,None)]
        name_mean = ['constant\t','Inflation.Lag(1)','SPF\t\t']
    elif mean_type=='ARX(2,1)':
        num_param_mean = 4
        mean_model = mean_ARX21
        sv_mean=[0.0753,0.1743,0.0574,0.7483]
        bounds_mean=[(min(Y),max(Y)),(-1.999,1.999),(-0.999,0.999),(None,None)]
        name_mean = ['constant\t','Inflation.Lag(1)','Inflation.Lag(2)','SPF\t\t']
    elif mean_type=='ARX(2,2)':
        mean_model = mean_ARX22
        num_param_mean = 5
        sv_mean=[0.08940,0.2081,0.0455,0.5584,0.1288]
        bounds_mean=[(min(Y),max(Y)),(-1.999,1.999),(-0.999,0.999),(None,None),(None,None)]
        name_mean = ['constant\t','Inflation.Lag(1)','Inflation.Lag(2)','SPF\t\t','SPF.lag(1)\t']
 
    residual_function = _make_residual_function(Y, X, mean_type)

    # log likelihood function
    def loglikelihood_bege(params,individual=False):
        """
        Parameter order:
        -----------------
           Mean model                (num_param_mean)
        + Positive GJR shape process  (4)
        + Negative GJR shape process   (4)
        + standard deviation p&n      (2)
        """
        param_mean=params[:num_param_mean]
        param_p= params[num_param_mean:num_param_mean+4]
        param_n= params[num_param_mean+4:num_param_mean+8]
        param_tp=params[ num_param_mean+8]
        param_tn=params[ num_param_mean+9]
        
        resids = residual_function(param_mean)
        pseries =  gjr_recursion(resids, param_p,param_tp )
        nseries = gjr_recursion(resids, param_n,param_tn )
        if individual == False:
            return -np.sum(BEGE_log_density(
                resids, pseries, nseries, param_tp, param_tn,
                hyperu_method=density_hyperu_method,
            ))
        return -BEGE_log_density(
            resids, pseries, nseries, param_tp, param_tn,
            hyperu_method=density_hyperu_method,
        )
    
    # starting_value
    if init_value is None:
        starting_values= sv_mean + [ 1,0.3,0.5,0.7,1,0.3,0.5,0.7,0.4,1 ]
    else:
        starting_values= init_value
    bounds =  bounds_mean+ [ (1e-5,None), (1e-5,0.999),(1e-5,1.999), (1e-5,1.999) ]+[ (1e-5,None), (1e-5,0.999),(1e-5,1.999),  (1e-5,1.999) ]+[(1e-5,10*np.std(Y))]+[(1e-5,10*np.std(Y))]
    
    constraints = [
        {'type': 'ineq', 'fun': lambda params: 2- 2*params[num_param_mean] - params[num_param_mean+1]- params[num_param_mean+2] },   #GJR Garch Constraints
        {'type': 'ineq', 'fun': lambda params:  2- 2*params[num_param_mean+4] - params[num_param_mean+5]- params[num_param_mean+6]}    
    ]
    opt = minimize(
        loglikelihood_bege,
        starting_values,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )
    print(opt.message)
    params = opt.x
    loglikelihood = -1.0 * opt.fun
    k = len(params)
    n = Y.shape[0]
    AIC = 2 * k - 2 * loglikelihood
    BIC = np.log(n) * k - 2 * loglikelihood

    print("AIC:", AIC)
    print("BIC:", BIC)
    
    if compute_se:
        hess = approx_hess(params,loglikelihood_bege, np.sqrt(np.finfo(float).eps)) 
        
        # Calculate the covariance matrix (inverse of the Hessian)
        inv_hess = np.linalg.pinv(hess)
        
        kwargs = {'individual':True}
        scores = approx_fprime(params,loglikelihood_bege, np.sqrt(np.finfo(float).eps), kwargs=kwargs)
        #score_cov = np.cov(scores.T)
        if scores.shape[1] != len(params):
            scores = scores.T

        score_cov = scores.T @ scores
        covariance_matrix = inv_hess.dot(score_cov).dot(inv_hess) 
        std_bege = np.sqrt(np.clip(np.diag(covariance_matrix), a_min=0.0, a_max=None))
    else:
        std_bege = np.full(k, np.nan, dtype=float)
 
    print('-------------------------------------------------------')
    name_bege  = name_mean+['Good Envir Const','Good Envir AR.1 ','Good Envir phi^+','Good Envir phi^-','Bad Envir Const ','Bad Envir AR.1  ','Bad Envir phi^+ ','Bad Envir phi^- ','Good Envir sigma','Bad Envir sigma ']
    print('Estimated Parameterss:')
    print('-------------------------------------------------------')
    for i in range(k):
        if compute_se:
            print(f"{name_bege[i]}: {params[i]:.3f},Std:{std_bege[i]:.3f},t-value:{params[i]/std_bege[i]:.3f}")
        else:
            print(f"{name_bege[i]}: {params[i]:.3f}")
    return opt


# def BEGE_Constant(Y, X=None, mean_type='constant'):
#     '''
#     grid search on gamma distribution parameters
#     '''
#     if mean_type == 'constant':
#         mean_model = mean_const
#         num_param_mean = 0
#         sv_mean = [] 
#         bounds_mean = []
#         name_mean = []
#     elif mean_type == 'ARX(1,1)':
#         mean_model = mean_ARX11
#         num_param_mean = 3
#         sv_mean = [0.0752, 0.1650, 0.8198]
#         bounds_mean = [(min(Y), max(Y)), (-0.999, 0.999), (None, None)]
#         name_mean = ['constant\t', 'Inflation.Lag(1)', 'SPF\t\t']
#     elif mean_type == 'ARX(2,1)':
#         num_param_mean = 4
#         mean_model = mean_ARX21
#         sv_mean = [0.0753, 0.1743, 0.0574, 0.7483]
#         bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (None, None)]
#         name_mean = ['constant\t', 'Inflation.Lag(1)', 'Inflation.Lag(2)', 'SPF\t\t']
#     elif mean_type == 'ARX(2,2)':
#         mean_model = mean_ARX22
#         num_param_mean = 5
#         sv_mean = [0.08940, 0.2081, 0.0455, 0.5584, 0.1288]
#         bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (None, None), (None, None)]
#         name_mean = ['constant\t', 'Inflation.Lag(1)', 'Inflation.Lag(2)', 'SPF\t\t', 'SPF.lag(1)\t']

#     # log likelihood function
#     def loglikelihood_bege(params, individual=False):
#         """
#         Parameter order:
#         -----------------
#            Mean model                (num_param_mean)
#         + Positive GJR shape   (1)
#         + Negative GJR shape    (1)
#         + standard deviation p&n      (2)
#         """
#         param_mean = params[:num_param_mean]
#         param_p = params[num_param_mean]
#         param_n = params[num_param_mean + 1]
#         param_tp = params[num_param_mean + 2]
#         param_tn = params[num_param_mean + 3]
        
#         resids = mean_model(Y, X, param_mean)
#         if individual == False:
#             return -np.sum(loglikedgam_constant(resids, param_p, param_n, param_tp, param_tn))
#         return -loglikedgam_constant(resids, param_p, param_n, param_tp, param_tn)
    
#     # Grid search ranges
#     grid_param_p = np.linspace(1, 6, 4)
#     grid_param_n = np.linspace(1, 6, 4)
#     grid_param_tp = np.linspace(0.1, 3, 4)
#     grid_param_tn = np.linspace(0.1, 5, 4)
    
#     best_opt = None
#     best_aic = np.inf
    
#     for param_p in grid_param_p:
#         for param_n in grid_param_n:
#             for param_tp in grid_param_tp:
#                 for param_tn in grid_param_tn:
#                     # Starting values
#                     sv_mean_perturbed = [sv + np.random.normal() for sv in sv_mean]
#                     starting_values = sv_mean_perturbed + [param_p, param_n, param_tp, param_tn]
#                     bounds = bounds_mean + [(1e-5, None), (1e-5, None), (1e-5, 10 * np.std(Y)), (1e-5, 10 * np.std(Y))]
                    
#                     opt = minimize(
#                         loglikelihood_bege,
#                         starting_values,
#                         method='SLSQP',
#                         bounds=bounds,
#                         options={'maxiter': 300}
#                     )
                    
#                     if opt.success:
#                         params = opt.x
#                         loglikelihood = -1.0 * opt.fun
#                         k = len(params)
#                         n = Y.shape[0]
#                         AIC = 2 * k - 2 * loglikelihood
                        
#                         if AIC < best_aic:
#                             best_aic = AIC
#                             best_opt = opt
    
#     opt = best_opt
#     params = opt.x
#     loglikelihood = -1.0 * opt.fun
#     k = len(params)
#     n = Y.shape[0]
#     BIC = np.log(n) * k - 2 * loglikelihood

#     print(opt.message)
#     print("AIC:", best_aic)
#     print("BIC:", BIC)
    
#     hess = approx_hess(params, loglikelihood_bege, np.sqrt(np.finfo(float).eps)) / n
    
#     # Calculate the covariance matrix (inverse of the Hessian)
#     inv_hess = np.linalg.inv(hess)
    
#     kwargs = {'individual': True}
#     scores = approx_fprime(params, loglikelihood_bege, np.sqrt(np.finfo(float).eps), kwargs=kwargs)
#     if scores.shape[1] != len(params):
#         scores = scores.T

#     # Step 5: Calculate the outer product of the score vectors and sum them
#     score_cov = np.zeros((len(params), len(params)))
#     for i in range(n):
#         score_vec = scores[i, :].reshape(-1, 1)  # column vector
#         score_cov += score_vec.dot(score_vec.T)
#     score = score_cov / n
#     covariance_matrix = inv_hess.dot(score_cov).dot(inv_hess) 
#     std_bege = np.sqrt(np.diag(covariance_matrix))
 
#     print('-------------------------------------------------------')
#     name_bege  = name_mean + ['Good Envir P   ', 'Bad Envir N    ', 'Good Envir sigma', 'Bad Envir sigma ']
#     print('Estimated Parameters:')
#     print('-------------------------------------------------------')
#     for i in range(k):
#         print(f"{name_bege[i]}: {params[i]:.3f}, Std: {std_bege[i]:.3f}, t-value: {params[i] / std_bege[i]:.3f}")
    
#     return opt




# def BEGE_Constant(Y, X=None, mean_type='constant', n_samples=20, n_jobs=4, random_state=123):
#     '''
#     Grid-search on gamma distribution parameters by:
#       1. Sampling n_samples points from the full 4-D grid,
#       2. Running each optimization in parallel,
#       3. Picking the solution with the lowest AIC.
#     '''
#     #––– 1) set up your mean model and starting/bounds for its params
#     if mean_type == 'constant':
#         mean_model = mean_const
#         num_param_mean = 0
#         sv_mean = [] 
#         bounds_mean = []
#         name_mean = []
#     elif mean_type == 'ARX(1,1)':
#         mean_model = mean_ARX11
#         num_param_mean = 3
#         sv_mean = [0.0752, 0.1650, 0.8198]
#         bounds_mean = [(min(Y), max(Y)), (-0.999, 0.999), (None, None)]
#         name_mean = ['constant\t', 'Inflation.Lag(1)', 'SPF\t']
#     elif mean_type == 'ARX(2,1)':
#         num_param_mean = 4
#         mean_model = mean_ARX21
#         sv_mean = [0.0753, 0.1743, 0.0574, 0.7483]
#         bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (None, None)]
#         name_mean = ['constant\t', 'Inflation.Lag(1)', 'Inflation.Lag(2)', 'SPF\t']
#     elif mean_type == 'ARX(2,2)':
#         mean_model = mean_ARX22
#         num_param_mean = 5
#         sv_mean = [0.08940, 0.2081, 0.0455, 0.5584, 0.1288]
#         bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (None, None), (None, None)]
#         name_mean = ['constant\t', 'Inflation.Lag(1)', 'Inflation.Lag(2)', 'SPF\t', 'SPF.lag(1)\t']




#     #––– 2) define the log-likelihood wrapper
#     def loglikelihood_bege(params, individual=False):
#         pm = params[:num_param_mean]
#         p, n, tp, tn = params[num_param_mean:]
#         resids = mean_model(Y, X, pm)
#         ll   = loglikedgam_constant(resids, p, n, tp, tn)
#         return -np.sum(ll) if not individual else -ll

#     #––– 3) sample 10 random grid points from the full 4-D grid
#     grid_p  = np.linspace(1,   6, 4)
#     grid_n  = np.linspace(1,   6, 4)
#     grid_tp = np.linspace(0.1, 3, 4)
#     grid_tn = np.linspace(0.1, 5, 4)
#     all_pts = np.array(np.meshgrid(grid_p, grid_n, grid_tp, grid_tn)).T.reshape(-1,4)
    
#     rng = np.random.RandomState(random_state)
#     idx = rng.choice(len(all_pts), size=n_samples, replace=False)
#     samples = all_pts[idx]

#     #––– 4) optimizer for one sample
#     def _optimize_one(pt):
#         p0_perturbed = [sv + rng.normal() for sv in sv_mean]
#         start_vals   = p0_perturbed + list(pt)
#         bounds = bounds_mean + [
#             (1e-5, None), (1e-5, None),
#             (1e-5, 10*np.std(Y)), (1e-5, 10*np.std(Y))
#         ]
#         out = minimize(
#             loglikelihood_bege,
#             start_vals,
#             method='SLSQP',
#             bounds=bounds,
#             options={'maxiter': 300}
#         )
#         if not out.success:
#             return None
#         k   = len(out.x)
#         ll  = -out.fun
#         AIC = 2*k - 2*ll
#         return (AIC, out)

#     #––– 5) parallel execution
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(_optimize_one)(pt) for pt in samples
#     )
#     # filter out failures
#     results = [r for r in results if r is not None]
#     if not results:
#         raise RuntimeError("All optimizations failed.")

#     #––– 6) pick best
#     best_aic, best_opt = min(results, key=lambda x: x[0])
#     opt = best_opt
#     params = opt.x
#     loglik = -opt.fun
#     k, n   = len(params), Y.shape[0]
#     BIC    = np.log(n)*k - 2*loglik

#     #––– 7) report
#     print(opt.message)
#     print(f"AIC: {best_aic:.3f}")
#     print(f"BIC: {BIC:.3f}")

#     #––– 8) compute standard errors and t-stats
#     hess = approx_hess(params, loglikelihood_bege, np.sqrt(np.finfo(float).eps)) / n
#     inv_h = np.linalg.inv(hess)
#     scores = approx_fprime(
#         params, loglikelihood_bege, np.sqrt(np.finfo(float).eps),
#         kwargs={'individual': True}
#     )
#     if scores.shape[1] != k:
#         scores = scores.T
#     score_cov = sum(
#         scores[i].reshape(-1,1) @ scores[i].reshape(1,-1)
#         for i in range(n)
#     ) / n
#     cov_mat = inv_h @ score_cov @ inv_h
#     std_err = np.sqrt(np.diag(cov_mat))

#     # names for display
#     name_bege = name_mean + [
#         'Good Envir P', 'Bad Envir N', 'Good Envir σ', 'Bad Envir σ'
#     ]
#     print('-'*55)
#     print("Estimated Parameters:")
#     print('-'*55)
#     for i in range(k):
#         tstat = params[i] / std_err[i]
#         print(f"{name_bege[i]:<20}: {params[i]:.4f}  (SE: {std_err[i]:.4f}, t={tstat:.2f})")

#     return opt





# import numpy as np
# from scipy.optimize import differential_evolution, minimize
# from joblib import Parallel, delayed

# def BEGE_Constant(Y, X=None, mean_type='constant',
#                   de_niter=60, de_popsize=15,
#                   refine_maxiter=300):
#     """
#     1) Global search on (p, n, t_p, t_n) via Differential Evolution
#     2) Local refine on all params (mean + p,n,t_p,t_n) via SLSQP
#     3) Print AIC, BIC, SE's and t‐stats
#     """
#     # ——— 1) pick your mean model
#     if mean_type == 'constant':
#         mean_model, num_m = mean_const, 0
#         sv_mean, bounds_mean, names_mean = [], [], []
#     elif mean_type == 'ARX(1,1)':
#         mean_model, num_m = mean_ARX11, 3
#         sv_mean = [0.0752, 0.1650, 0.8198]
#         bounds_mean = [(min(Y), max(Y)), (-0.999, 0.999), (None, None)]
#         names_mean = ['const', 'Infl(1)', 'SPF']
#     elif mean_type == 'ARX(2,1)':
#         mean_model, num_m = mean_ARX21, 4
#         sv_mean = [0.0753, 0.1743, 0.0574, 0.7483]
#         bounds_mean = [
#             (min(Y), max(Y)),
#             (-1.999, 1.999),
#             (-0.999, 0.999),
#             (None, None)
#         ]
#         names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
#     else:  # ARX(2,2)
#         mean_model, num_m = mean_ARX22, 5
#         sv_mean = [0.0894, 0.2081, 0.0455, 0.5584, 0.1288]
#         bounds_mean = [
#             (min(Y), max(Y)),
#             (-1.999, 1.999),
#             (-0.999, 0.999),
#             (None, None),
#             (None, None)
#         ]
#         names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']

#     # ——— 2) define the 4-dim DE objective (only GJR params; fix mean at sv_mean)
#     def _de_obj(theta):
#         p, n, tp, tn = theta
#         res = mean_model(Y, X, sv_mean)
#         return -np.sum(loglikedgam_constant(res, p, n, tp, tn))

#     # ——— 3) run Differential Evolution
#     de_bounds = [(1,   6),    # p
#                  (1,   6),    # n
#                  (0.1, 3),    # t_p
#                  (0.1, 5)]    # t_n

#     de = differential_evolution(
#         _de_obj,
#         bounds=de_bounds,
#         maxiter=de_niter,
#         popsize=de_popsize,
#         tol=1e-6,
#         disp=True
#     )
#     print("DE:", de.message)
#     best_de = de.x

#     # ——— 4) build full starting vector for SLSQP
#     #    lightly perturb your mean params
#     rng = np.random.RandomState(0)
#     sv0 = [sv + rng.normal(scale=0.01) for sv in sv_mean]
#     x0  = sv0 + list(best_de)

#     # full bounds (mean + gjr)
#     full_bounds = (
#         bounds_mean
#         + [(1e-5, None), (1e-5, None),
#            (1e-5, 10*np.std(Y)), (1e-5, 10*np.std(Y))]
#     )

#     # ——— 5) define full‐dim loglik for refine
#     def _full_obj(params):
#         pm = params[:num_m]
#         p, n, tp, tn = params[num_m:]
#         resid = mean_model(Y, X, pm)
#         return -np.sum(loglikedgam_constant(resid, p, n, tp, tn))

#     opt = minimize(
#         _full_obj,
#         x0,
#         method='SLSQP',
#         bounds=full_bounds,
#         options={'maxiter': refine_maxiter}
#     )
#     print("Refine:", opt.message)

#     # ——— 6) AIC, BIC
#     params = opt.x
#     ll     = -opt.fun
#     k, N   = len(params), len(Y)
#     AIC    = 2*k - 2*ll
#     BIC    = np.log(N)*k - 2*ll
#     print(f"AIC: {AIC:.3f}, BIC: {BIC:.3f}")

#     # ——— 7) SE’s via Hessian/outer‐product of scores
#     hess = approx_hess(params, _full_obj, np.sqrt(np.finfo(float).eps)) / N
#     invh = np.linalg.inv(hess)

#     scores = approx_fprime(
#         params, _full_obj, np.sqrt(np.finfo(float).eps),args=(),   
#         kwargs={'individual': True}
#     )
#     if scores.shape[1] != k:
#         scores = scores.T

#     S = sum(
#         scores[i].reshape(-1,1) @ scores[i].reshape(1,-1)
#         for i in range(N)
#     ) / N
#     cov = invh @ S @ invh
#     se  = np.sqrt(np.diag(cov))

#     # ——— 8) print estimates
#     names = names_mean + ['GJR p', 'GJR n', 'σ₊', 'σ₋']
#     print("-"*50)
#     print("Parameter Estimates")
#     print("-"*50)
#     for i, nm in enumerate(names):
#         tstat = params[i] / se[i]
#         print(f"{nm:15s}: {params[i]:.4f} (SE {se[i]:.4f}, t={tstat:.2f})")

#     return opt

 
from scipy.optimize import differential_evolution, minimize

def BEGE_Constant_DE(Y, X=None, mean_type='constant',
                  de_niter=60, de_popsize=15, refine_maxiter=300,
                  compute_se=False, density_hyperu_method='scipy_approx'):
    """
    1) Global search on all parameters (mean process + distributional) via Differential Evolution
    2) Optional local refine via SLSQP
    3) Print AIC, BIC, SE's and t‐stats
    """
    Y, X = _align_mean_sample(Y, X)

    # ——— 1) select mean model
    if mean_type == 'constant':
        mean_model, num_m = mean_const, 0
        sv_mean, bounds_mean, names_mean = [], [], []
    elif mean_type == 'ARX(1,1)':
        mean_model, num_m = mean_ARX11, 3
        sv_mean = [0.0752, 0.1650, 0.8198]
        bounds_mean = [(min(Y), max(Y)), (-0.999, 0.999), (-10, 10)]
        names_mean = ['const', 'Infl(1)', 'SPF']
    elif mean_type == 'ARX(2,1)':
        mean_model, num_m = mean_ARX21, 4
        sv_mean = [0.0753, 0.1743, 0.0574, 0.7483]
        bounds_mean = [
            (min(Y), max(Y)),
            (-1.999, 1.999),
            (-0.999, 0.999),
            (-10, 10)
        ]
        names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
    else:  # ARX(2,2)
        mean_model, num_m = mean_ARX22, 5
        sv_mean = [0.0894, 0.2081, 0.0455, 0.5584, 0.1288]
        bounds_mean = [
            (min(Y), max(Y)),
            (-1.999, 1.999),
            (-0.999, 0.999),
            (-10, 10),
            (-10, 10)
        ]
        names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']

    residual_function = _make_residual_function(Y, X, mean_type)

    # ——— 2) full objective for DE & refine
    def full_obj(params):
        pm = params[:num_m]
        p, n, tp, tn = params[num_m:]
        resid = residual_function(pm)
        return -np.sum(loglikedgam_constant(
            resid, p, n, tp, tn, hyperu_method=density_hyperu_method,
        ))

    # ——— 3) per-observation loglik for score
    def ind_loglik(params):
        pm = params[:num_m]
        p, n, tp, tn = params[num_m:]
        resid = residual_function(pm)
        return -loglikedgam_constant(
            resid, p, n, tp, tn, hyperu_method=density_hyperu_method,
        )

    # ——— 4) build bounds
    dist_bounds = [
        (0.1,   10),    # p
        (0.1,  10),    # n
        (0.05, 2),    # t_p
        (0.05, 2)     # t_n
    ]
    full_bounds = bounds_mean + dist_bounds

    # ——— 5) run DE across all parameters
    de = differential_evolution(
        full_obj,
        bounds=full_bounds,
        maxiter=de_niter,
        popsize=de_popsize,
        tol=1e-6,
        disp=False
    )
    print("DE finished:", de.message)
    params_de = de.x

    # ——— 6) refine with SLSQP
    opt = minimize(
        full_obj,
        params_de,
        method='SLSQP',
        bounds=full_bounds,
        options={'maxiter': refine_maxiter}
    )
    print("Refinement:", opt.message)

    # ——— 7) AIC, BIC
    params = opt.x
    ll     = -opt.fun
    k, N   = len(params), len(Y)
    AIC    = 2*k - 2*ll
    BIC    = np.log(N)*k - 2*ll
    print(f"AIC: {AIC:.3f}, BIC: {BIC:.3f}")

    if compute_se:
        # ——— 8) Hessian and SEs
        hess = approx_hess(params, full_obj, np.sqrt(np.finfo(float).eps)) / N
        invh = np.linalg.pinv(hess)

        scores = approx_fprime(params, ind_loglik, np.sqrt(np.finfo(float).eps))
        if scores.shape[0] != N and scores.shape[1] == N:
            scores = scores.T
        S = scores.T @ scores / N
        cov = invh @ S @ invh
        se  = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    else:
        se = np.full(k, np.nan, dtype=float)

    names = names_mean + ['GJR p', 'GJR n', 'σ₊', 'σ₋']
    print("-"*50)
    print("Parameter Estimates")
    print("-"*50)
    for i, nm in enumerate(names):
        if compute_se:
            tstat = params[i] / se[i]
            print(f"{nm:15s}: {params[i]:.4f} (SE {se[i]:.4f}, t={tstat:.2f})")
        else:
            print(f"{nm:15s}: {params[i]:.4f}")

    return opt


def BEGE_Constant_MLE(Y, X=None, mean_type='ARX(2,2)',
                      n_starts=20, maxiter=1500, tol=1e-8, random_state=None,
                      compute_se=False, density_hyperu_method='scipy_approx',
                      enforce_variance_bounds=True, start_n_jobs=None):
    """
    BEGE GARCH MLE with random initialization.
    Mean params drawn from manually set ranges (last estimation results).
    Volatility params drawn from bounds.
    """

    rng = np.random.default_rng(random_state)
    Y, X = _align_mean_sample(Y, X)
    N = len(Y)
    big_penalty = 1e12
    big_vec_penalty = 1e6
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1.")

    # -------- 1) Select mean model and bounds --------
    if mean_type == 'constant':
        mean_model, num_m = mean_const, 0
        bounds_mean, names_mean = [], []
        # No mean params, skip range
        mean_param_ranges = []
    elif mean_type == 'ARX(1,1)':
        mean_model, num_m = mean_ARX11, 3
        bounds_mean = [(min(Y), max(Y)), (-0.999, 0.999), (-10, 10)]
        names_mean = ['c', 'rho_1', 'phi_1']
        # OLS +/- 2*SE from MeanProcess/README.md and MeanProcess/OLS_Results.md
        mean_param_ranges = [
            (0.0824 - 2*0.086, 0.0824 + 2*0.086),   # c
            (0.3005 - 2*0.112, 0.3005 + 2*0.112),   # rho_1
            (0.7337 - 2*0.167, 0.7337 + 2*0.167)    # phi_1
        ]
    elif mean_type == 'ARX(2,1)':
        mean_model, num_m = mean_ARX21, 4
        bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)]
        names_mean = ['c', 'rho_1', 'rho_2', 'phi_1']
        mean_param_ranges = [
            (0.0897 - 2*0.086, 0.0897 + 2*0.086),   # c
            (0.2892 - 2*0.098, 0.2892 + 2*0.098),   # rho_1
            (0.0834 - 2*0.126, 0.0834 + 2*0.126),   # rho_2
            (0.6385 - 2*0.251, 0.6385 + 2*0.251)    # phi_1
        ]
    elif mean_type == 'ARX(2,2)':
        mean_model, num_m = mean_ARX22, 5
        bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)]
        names_mean = ['c', 'rho_1', 'rho_2', 'phi_1', 'phi_2']
        mean_param_ranges = [
            (0.0856 - 2*0.086, 0.0856 + 2*0.086),   # c
            (0.2992 - 2*0.106, 0.2992 + 2*0.106),   # rho_1
            (0.0914 - 2*0.125, 0.0914 + 2*0.125),   # rho_2
            (0.4084 - 2*0.433, 0.4084 + 2*0.433),   # phi_1
            (0.2136 - 2*0.297, 0.2136 + 2*0.297)    # phi_2
        ]
    else:
        raise ValueError("Invalid mean_type")

    # -------- 2) Volatility bounds --------
    dist_bounds = [
        (0.1,   10),    # p
        (0.1,  10),    # n
        (0.05, 2),    # t_p
        (0.05, 2)     # t_n
    ]
    names_dist = ['GJR p', 'GJR n', 'σ₊', 'σ₋']
    full_bounds = bounds_mean + dist_bounds
    residual_function = _make_residual_function(Y, X, mean_type)

    # -------- 3) Objective (negative log-likelihood) --------
    def full_obj(params):
        pm = params[:num_m]
        p, n, tp, tn = params[num_m:]
        resid = residual_function(pm)
        pseries = np.full_like(resid, float(p), dtype=float)
        nseries = np.full_like(resid, float(n), dtype=float)
        if enforce_variance_bounds and not bege_variance_bounds_ok(resid, pseries, nseries, tp, tn):
            return float(big_penalty)
        val = -float(np.sum(loglikedgam_constant(
            resid, p, n, tp, tn, hyperu_method=density_hyperu_method,
        )))
        return val if np.isfinite(val) else float(big_penalty)

    def ind_loglik(params):
        pm = params[:num_m]
        p, n, tp, tn = params[num_m:]
        resid = residual_function(pm)
        pseries = np.full_like(resid, float(p), dtype=float)
        nseries = np.full_like(resid, float(n), dtype=float)
        if enforce_variance_bounds and not bege_variance_bounds_ok(resid, pseries, nseries, tp, tn):
            return np.full(N, float(big_vec_penalty), dtype=float)
        return -loglikedgam_constant(
            resid, p, n, tp, tn, hyperu_method=density_hyperu_method,
        )

    # -------- 4) Sampling helpers --------
    def sample_mean_params():
        vals = []
        for (low, high), (b_low, b_high) in zip(mean_param_ranges, bounds_mean):
            lo = max(low, b_low if b_low is not None else low)
            hi = min(high, b_high if b_high is not None else high)
            vals.append(rng.uniform(lo, hi))
        return np.array(vals, dtype=float)

    def sample_dist_params():
        return np.array([rng.uniform(a, b) for (a, b) in dist_bounds], dtype=float)

    # -------- 5) Multi-start MLE --------
    start_values = [
        np.concatenate([sample_mean_params(), sample_dist_params()])
        for _ in range(int(n_starts))
    ]
    results = _run_multistart_minimize(
        start_values,
        full_obj,
        method='L-BFGS-B',
        bounds=full_bounds,
        options={'maxiter': maxiter, 'ftol': tol},
        start_n_jobs=start_n_jobs,
    )

    best_fun = np.inf
    best_opt = None
    last_error = None
    for opt, exc in results:
        if exc is not None:
            last_error = exc
            continue
        if _optimizer_result_eligible(opt) and opt.fun < best_fun:
            best_fun = opt.fun
            best_opt = opt

    if best_opt is None or (not np.isfinite(best_fun)) or best_fun >= big_penalty:
        detail = ""
        if last_error is not None:
            detail = f" Last error: {type(last_error).__name__}: {last_error}"
        raise RuntimeError(f"All optimization runs failed to converge to a finite eligible objective value.{detail}")

    # -------- 6) Post-estimation --------
    params = best_opt.x
    ll = -best_opt.fun
    k = len(params)
    AIC = 2*k - 2*ll
    BIC = np.log(N)*k - 2*ll

    if compute_se:
        # Hessian & robust SE
        hess = approx_hess(params, full_obj, np.sqrt(np.finfo(float).eps)) / N
        try:
            invh = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            invh = np.linalg.pinv(hess)

        scores = approx_fprime(params, ind_loglik, np.sqrt(np.finfo(float).eps))
        if scores.shape[0] != N and scores.shape[1] == N:
            scores = scores.T
        if scores.shape[0] != N:
            raise RuntimeError(f"Unexpected score shape {scores.shape}; expected first dimension {N}.")
        S = scores.T @ scores / N
        cov = invh @ S @ invh
        se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    else:
        se = np.full(k, np.nan, dtype=float)

    # # Print results
    # names = names_mean + names_dist
    # print("-"*50)
    # print("Parameter Estimates (MLE)")
    # print("-"*50)
    # for nm, val, err in zip(names, params, se):
    #     print(f"{nm:15s}: {val:.4f} (SE {err:.4f}, t={val/err:.2f})")
    # print(f"AIC: {AIC:.3f}, BIC: {BIC:.3f}")

    return {
        'opt': best_opt,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'compute_se': bool(compute_se),
    }



def BEGE_Symmetric_MLE(Y, X=None, mean_type='ARX(2,2)',
                       n_starts=20, maxiter=500, tol=1e-8, random_state=None,
                       compute_se=False, density_hyperu_method='scipy_approx',
                       enforce_variance_bounds=True, start_n_jobs=None):
    """
    Symmetric-volatility BEGE MLE (multi-start, no explicit constraints).
    - One shared GJR shape process s_t for BOTH p_t and n_t (symmetry).
    - One shared scale parameter sigma: t_p = t_n = sigma.
    - Recursion uses *your* gjr_recursion exactly.

    Params (order):
      [ mean params (num_m) ] + [ cont_s, rho_s, phi_p, phi_n, sigma ]

    Returns:
      dict with opt result, params, se, AIC, BIC, loglik
    """
    rng = np.random.default_rng(random_state)
    Y, X = _align_mean_sample(Y, X)
    N = len(Y)
    big_penalty = 1e12
    big_vec_penalty = 1e6

    # ---------- 1) Mean model selection & ranges (copied from your MLE style) ----------
    if mean_type == 'constant':
        mean_model, num_m = mean_const, 0
        bounds_mean, names_mean = [], []
        mean_param_ranges = []
    elif mean_type == 'ARX(1,1)':
        mean_model, num_m = mean_ARX11, 3
        bounds_mean = [(min(Y), max(Y)), (-0.999, 0.999), (-10, 10)]
        names_mean = ['const', 'Infl(1)', 'SPF']
        mean_param_ranges = [
            (0.0792 - 2*0.087, 0.0792 + 2*0.087),
            (0.2793 - 2*0.073, 0.2793 + 2*0.073),
            (0.6661 - 2*0.156, 0.6661 + 2*0.156),
        ]
    elif mean_type == 'ARX(2,1)':
        mean_model, num_m = mean_ARX21, 4
        bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)]
        names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
        mean_param_ranges = [
            (0.0792 - 2*0.087, 0.0792 + 2*0.087),
            (0.2793 - 2*0.073, 0.2793 + 2*0.073),
            (0.0728 - 2*0.080, 0.0728 + 2*0.080),
            (0.6661 - 2*0.156, 0.6661 + 2*0.156),
        ]
    elif mean_type == 'ARX(2,2)':
        mean_model, num_m = mean_ARX22, 5
        bounds_mean = [(min(Y), max(Y)), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)]
        names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']
        mean_param_ranges = [
            (0.0761 - 2*0.087, 0.0761 + 2*0.087),
            (0.2880 - 2*0.076, 0.2880 + 2*0.076),
            (0.0799 - 2*0.082, 0.0799 + 2*0.082),
            (0.4720 - 2*0.459, 0.4720 + 2*0.459),
            (0.1793 - 2*0.399, 0.1793 + 2*0.399),
        ]
    else:
        raise ValueError("Invalid mean_type")

    # ---------- 2) Symmetric volatility bounds ----------
    # Keep broad but sane bounds; no explicit constraints, but avoid blow-ups via a soft penalty.
    # (cont_s, rho_s, phi_p, phi_n, sigma)
    sym_bounds = [
        (0.0, 10),             # cont_s
        (0.0, 1.0),            # rho_s
        (0.0, 2.0),            # phi_p
        (0.0, 2.0),            # phi_n
        (1e-5, 2),  # sigma (shared tp=tn)
    ]
    names_sym = ['Sym Cont', 'Sym AR(1)', 'Sym phi⁺', 'Sym phi⁻', 'Shared sigma']
    full_bounds = bounds_mean + sym_bounds
    residual_function = _make_residual_function(Y, X, mean_type)

    # ---------- 3) Helpers ----------
    def sample_mean_params():
        return np.array([rng.uniform(low, high) for (low, high) in mean_param_ranges], dtype=float)

    def sample_sym_params():
        vals = np.array([rng.uniform(a if a is not None else 1e-5,
                                     b if b is not None else 1.0) for (a, b) in sym_bounds], dtype=float)
        return vals

    # quick stability guard used only to avoid pathological starts / evaluations
    def _stable_tuple(cont, rho, phi_p, phi_n):
        # denominator in your backcast: 1 - rho - (phi_p + phi_n)/2
        return (1.0 - rho - 0.5*(phi_p + phi_n)) > 1e-6

    # ---------- 4) Objective (neg log-likelihood) ----------
    # Uses your gjr_recursion; symmetry enforced by using SAME s_t and SAME sigma for p & n.
    def full_obj(params):
        pm = params[:num_m]
        cont, rho, phi_p, phi_n, sigma = params[num_m:]
        # soft penalty to skip explosive regions without formal constraints
        if not _stable_tuple(cont, rho, phi_p, phi_n):
            return float(big_penalty)

        resid = residual_function(pm)
        sseries = gjr_recursion(resid, (cont, rho, phi_p, phi_n), sigma)  # your function
        if (
            not np.all(np.isfinite(sseries))
            or (enforce_variance_bounds and not bege_variance_bounds_ok(resid, sseries, sseries, sigma, sigma))
        ):
            return float(big_penalty)
        # symmetric: p_t = n_t = s_t, tp = tn = sigma
        ll = BEGE_log_density(
            resid, sseries, sseries, sigma, sigma,
            hyperu_method=density_hyperu_method,
        )
        # negative total log-likelihood
        val = -float(np.sum(ll))
        return val if np.isfinite(val) else float(big_penalty)

    def ind_loglik(params):
        pm = params[:num_m]
        cont, rho, phi_p, phi_n, sigma = params[num_m:]
        if not _stable_tuple(cont, rho, phi_p, phi_n):
            # return a large vector so OPG stays finite
            return np.full(N, float(big_vec_penalty), dtype=float)
        resid = residual_function(pm)
        sseries = gjr_recursion(resid, (cont, rho, phi_p, phi_n), sigma)
        if (
            not np.all(np.isfinite(sseries))
            or (enforce_variance_bounds and not bege_variance_bounds_ok(resid, sseries, sseries, sigma, sigma))
        ):
            return np.full(N, float(big_vec_penalty), dtype=float)
        return -BEGE_log_density(
            resid, sseries, sseries, sigma, sigma,
            hyperu_method=density_hyperu_method,
        )  # per-observation neg ll

    # ---------- 5) Multi-start MLE ----------
    start_values = []
    for _ in range(int(n_starts)):
        if num_m > 0:
            init_mean = sample_mean_params()
        else:
            init_mean = np.array([], dtype=float)

        # resample symmetric params until the quick stability guard is satisfied
        for _tries in range(50):
            init_sym = sample_sym_params()
            c_, r_, pp_, pn_, sg_ = init_sym
            if _stable_tuple(c_, r_, pp_, pn_):
                break
        else:
            # if we couldn't find a stable draw quickly, just clamp to a mild stable default
            init_sym = np.array([1.0, 0.3, 0.5, 0.5, max(0.4, 0.1*np.std(Y))], dtype=float)

        start_values.append(np.concatenate([init_mean, init_sym]))

    results = _run_multistart_minimize(
        start_values,
        full_obj,
        method='L-BFGS-B',
        bounds=full_bounds,
        options={'maxiter': maxiter, 'ftol': tol},
        start_n_jobs=start_n_jobs,
    )

    best_fun = np.inf
    best_opt = None
    last_error = None
    for opt, exc in results:
        if exc is not None:
            last_error = exc
            continue
        if _optimizer_result_eligible(opt) and opt.fun < best_fun:
            best_fun = opt.fun
            best_opt = opt

    if best_opt is None or (not np.isfinite(best_fun)) or best_fun >= big_penalty:
        detail = ""
        if last_error is not None:
            detail = f" Last error: {type(last_error).__name__}: {last_error}"
        raise RuntimeError(f"All starts failed to converge to a finite eligible objective value.{detail}")

    # ---------- 6) Post-estimation ----------
    params = best_opt.x
    ll = -best_opt.fun
    k = len(params)
    AIC = 2*k - 2*ll
    BIC = np.log(N)*k - 2*ll

    if compute_se:
        # Hessian & robust SE (like your MLE)
        hess = approx_hess(params, full_obj, np.sqrt(np.finfo(float).eps)) / N
        invh = np.linalg.pinv(hess)

        scores = approx_fprime(params, ind_loglik, np.sqrt(np.finfo(float).eps))
        if scores.ndim == 1:
            scores = scores[:, None]
        if scores.shape[0] != N and scores.shape[1] == N:
            scores = scores.T
        S = scores.T @ scores / N
        cov = invh @ S @ invh
        se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    else:
        se = np.full(k, np.nan, dtype=float)

    # Print results
    # names = names_mean + names_sym
    # print("-"*50)
    # print("Parameter Estimates (Symmetric Volatility MLE)")
    # print("-"*50)
    # for nm, val, err in zip(names, params, se):
    #     tval = val/err if err > 0 else np.nan
    #     print(f"{nm:15s}: {val:.4f} (SE {err:.4f}, t={tval:.2f})")
    # print(f"AIC: {AIC:.3f}, BIC: {BIC:.3f}")

    return {
        'opt': best_opt,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'compute_se': bool(compute_se),
    }


def BEGE_AsymSharedGJR_MLE(
    Y, X=None, mean_type='ARX(1,1)',
    n_starts=50, maxiter=800, tol=1e-8, random_state=None,
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.0, 10.0),
    rho_bounds=(0.0, 1.0),
    phi_bounds=(0.0, 2.0),  # kept for optimizer bounds; sampling ignores these for phis (see below)
    floor_eps=1e-6,
    print_summary=True,
    big_penalty=1e12,
    big_vec_penalty=1e6,
    compute_se=False,
    density_hyperu_method='scipy_approx',
    enforce_variance_bounds=True,
    start_n_jobs=None,
):
    """
    BEGE with asymmetric constants and scales, shared GJR coefficients.

    MODS:
      • Stability guard: rho + 0.5*(phi+ + phi-) < 1.
      • Sampling: sample all params from bounds except phip/2 and phin/2.
        Sample beta_p := phip/2 ~ U[0, 1 - rho - floor_eps], beta_n similarly,
        then set phip = 2*beta_p, phin = 2*beta_n.

    Returns dict:
      {'opt','params','se','AIC','BIC','loglik','names'}
    """
    import numpy as np
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_hess

    rng = np.random.default_rng(random_state)
    Y, X = _align_mean_sample(Y, X)
    N_obs = int(Y.shape[0])

    # ---------- mean spec ----------
    def _get_mean_spec(Y, mean_type):
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        if mean_type == 'constant':
            mean_model, num_m = mean_const, 0
            bounds_mean, names_mean, ranges = [], [], []
        elif mean_type == 'ARX(1,1)':
            mean_model, num_m = mean_ARX11, 3
            bounds_mean = [(ymin, ymax), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'SPF']
            ranges = [
                (0.0720 - 2*0.086, 0.0720 + 2*0.086),
                (0.2881 - 2*0.073, 0.2881 + 2*0.073),
                (0.7508 - 2*0.125, 0.7508 + 2*0.125)
            ]
        elif mean_type == 'ARX(2,1)':
            mean_model, num_m = mean_ARX21, 4
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
            ranges = [
                (0.0792 - 2*0.087, 0.0792 + 2*0.087),
                (0.2793 - 2*0.073, 0.2793 + 2*0.073),
                (0.0728 - 2*0.080, 0.0728 + 2*0.080),
                (0.6661 - 2*0.156, 0.6661 + 2*0.156),
            ]
        elif mean_type == 'ARX(2,2)':
            mean_model, num_m = mean_ARX22, 5
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']
            ranges = [
                (0.0761 - 2*0.087, 0.0761 + 2*0.087),
                (0.2880 - 2*0.076, 0.2880 + 2*0.076),
                (0.0799 - 2*0.082, 0.0799 + 2*0.082),
                (0.4720 - 2*0.459, 0.4720 + 2*0.459),
                (0.1793 - 2*0.399, 0.1793 + 2*0.399),
            ]
        else:
            raise ValueError("Invalid mean_type")
        return mean_model, num_m, bounds_mean, names_mean, ranges

    mean_model, num_m, bounds_mean, names_mean, mean_ranges = _get_mean_spec(Y, mean_type)
    residual_function = _make_residual_function(Y, X, mean_type)

    # ---------- bounds & names ----------
    (sig_lo, sig_hi) = sigma_bounds
    (p0_lo, p0_hi) = p0n0_bounds
    (rho_lo, rho_hi) = rho_bounds
    (phi_lo, phi_hi) = phi_bounds  # used for optimizer only; sampling ignores for phis

    bounds_vol = [
        (p0_lo,  p0_hi),   # p0
        (p0_lo,  p0_hi),   # n0
        (rho_lo, rho_hi),  # rho (shared)
        (phi_lo, phi_hi),  # phi+
        (phi_lo, phi_hi),  # phi-
        (sig_lo, sig_hi),  # sigma_p
        (sig_lo, sig_hi),  # sigma_n
    ]
    names_vol = ['p0', 'n0', 'rho', 'phi⁺', 'phi⁻', 'σ₊', 'σ₋']

    bounds_full = bounds_mean + bounds_vol
    names_full = names_mean + names_vol
    k = len(bounds_full)

    # ---------- constraints ----------
    def _unpack_vol(theta):
        p0, n0, rho, phip, phin, sigp, sign = theta[num_m:]
        return p0, n0, rho, phip, phin, sigp, sign

    def _constraints_ok(p0, n0, rho, phip, phin, sigp, sign):
        vals = np.asarray([p0, n0, rho, phip, phin, sigp, sign], dtype=float)
        if not np.all(np.isfinite(vals)):
            return False
        if rho + 0.5 * (phip + phin) >= 1.0 - floor_eps:
            return False
        return True

    def constr_rho_phi(theta):
        p0, n0, rho, phip, phin, sigp, sign = _unpack_vol(theta)
        return 1.0 - floor_eps - (rho + 0.5 * (phip + phin))

    constraints = [
        {'type': 'ineq', 'fun': constr_rho_phi},
    ]

    # ---------- objectives ----------
    def _negloglik(params):
        pm = params[:num_m]
        p0, n0, rho, phip, phin, sigp, sign = params[num_m:]
        if not _constraints_ok(p0, n0, rho, phip, phin, sigp, sign):
            return float(big_penalty)
        res = residual_function(pm)

        # Recursions
        pseries = gjr_recursion(res, (float(p0), float(rho), float(phip), float(phin)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho), float(phip), float(phin)), float(sign))

        if (not np.all(np.isfinite(pseries)) or
            not np.all(np.isfinite(nseries))):
            return float(big_penalty)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return float(big_penalty)

        ll = BEGE_log_density(
            res, pseries, nseries, float(sigp), float(sign),
            hyperu_method=density_hyperu_method,
        )
        val = -float(np.sum(ll))
        # Guard against NaNs/Infs in likelihood too
        if not np.isfinite(val):
            return float(big_penalty)
        return val

    def _ind_negloglik_vec(params):
        """
        Per-observation NEG log-likelihood: always returns shape (N_obs,).
        """
        pm = params[:num_m]
        p0, n0, rho, phip, phin, sigp, sign = params[num_m:]
        if not _constraints_ok(p0, n0, rho, phip, phin, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)
        res = residual_function(pm)

        pseries = gjr_recursion(res, (float(p0), float(rho), float(phip), float(phin)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho), float(phip), float(phin)), float(sign))

        if (not np.all(np.isfinite(pseries)) or
            not np.all(np.isfinite(nseries))):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        v = -BEGE_log_density(
            res, pseries, nseries, float(sigp), float(sign),
            hyperu_method=density_hyperu_method,
        )
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.shape[0] != N_obs:
            v = np.full(N_obs, float(v.ravel()[0]))
        # Replace any bad values by penalty (robustness)
        if not np.all(np.isfinite(v)):
            v = np.full(N_obs, float(big_vec_penalty))
        return v

    # ---------- robust numerical derivatives ----------
    def _project_to_bounds(x, bounds):
        out = np.array(x, float)
        tiny = 1e-10
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None: out[j] = max(out[j], lo + tiny)
            if hi is not None: out[j] = min(out[j], hi - tiny)
        return out

    def _central_diff_scores(theta, f_per_obs, bounds, rel=1e-4, absmin=1e-6):
        """
        Jacobian (N_obs x k): row i = score_i(theta).
        Uses parameter-scaled central differences and bounds projection.
        """
        theta = np.asarray(theta, float)
        k = theta.size
        f0 = f_per_obs(theta)                    # (N_obs,)
        J = np.empty((N_obs, k), float)

        # parameter-scaled steps
        h = np.maximum(absmin, rel * np.maximum(1.0, np.abs(theta)))

        for j in range(k):
            th_p = theta.copy(); th_p[j] += h[j]
            th_m = theta.copy(); th_m[j] -= h[j]
            th_p = _project_to_bounds(th_p, bounds)
            th_m = _project_to_bounds(th_m, bounds)

            fp = f_per_obs(th_p); fp = np.asarray(fp, float).reshape(-1)
            fm = f_per_obs(th_m); fm = np.asarray(fm, float).reshape(-1)

            if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
            if fm.shape[0] != N_obs: fm = np.full(N_obs, float(fm.ravel()[0]))

            denom = float(th_p[j] - th_m[j])
            if denom == 0.0:
                th_p = theta.copy(); th_p[j] += h[j]
                th_p = _project_to_bounds(th_p, bounds)
                fp = f_per_obs(th_p); fp = np.asarray(fp, float).reshape(-1)
                if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
                fm = f0
                denom = float(th_p[j] - theta[j])

            J[:, j] = (fp - fm) / denom

        return J

    # ---------- sampling ----------
    def _sample_mean():
        if num_m == 0:
            return np.array([], dtype=float)
        return np.array([rng.uniform(a, b) for (a, b) in mean_ranges], dtype=float)

    def _sample_vol_safe(
        rng,
        p0_bounds, rho_bounds, phi_bounds, sigma_bounds,
        floor_eps=1e-6,
        max_tries=200
    ):
        """
        Return a *stable* parameter vector [p0, n0, rho, phip, phin, sigp, sign]
        under the guard:
           1 - rho - max(phin/2, phip/2) > floor_eps

        Sampling rule:
          - rho ~ U[rho_lo, rho_hi]
          - sigp, sign ~ U[sigma_lo, sigma_hi]
          - p0, n0 ~ U[p0_lo, p0_hi]
          - beta_p := phip/2 ~ U[0, 1 - rho - floor_eps]
            beta_n := phin/2 ~ U[0, 1 - rho - floor_eps]
            phip = 2*beta_p, phin = 2*beta_n
          - Require denom = 1 - rho - 0.5*(phip + phin) > floor_eps
        """
        p0_lo, p0_hi = p0_bounds
        rho_lo, rho_hi = rho_bounds
        phi_lo, phi_hi = phi_bounds
        sig_lo, sig_hi = sigma_bounds

        rho_upper = min(rho_hi, 1.0 - floor_eps - phi_lo)
        if rho_upper <= rho_lo:
            raise ValueError("rho_bounds and phi_bounds leave no feasible stable Symmetric BEGE starts.")

        for _ in range(max_tries):
            rho  = rng.uniform(rho_lo, rho_upper)
            sigp = rng.uniform(sig_lo, sig_hi)
            sign = rng.uniform(sig_lo, sig_hi)
            p0 = rng.uniform(p0_lo, p0_hi)
            n0 = rng.uniform(p0_lo, p0_hi)

            cap = min(0.5 * phi_hi, 1.0 - rho - floor_eps)
            if cap <= 0:
                continue

            beta_p = rng.uniform(0.0, cap)
            beta_n = rng.uniform(0.0, cap)
            phip = 2.0 * beta_p
            phin = 2.0 * beta_n

            denom = 1.0 - rho - 0.5*(phip + phin)
            if denom <= floor_eps:
                continue

            vals = np.array([p0, n0, rho, phip, phin, sigp, sign], dtype=float)
            if _constraints_ok(*vals):
                return vals

        # Conservative fallback if repeated failures
        rho  = 0.5
        sigp = min(max(0.3, sig_lo), sig_hi)
        sign = min(max(0.3, sig_lo), sig_hi)
        beta = 0.25 * (1.0 - rho - floor_eps) if (1.0 - rho - floor_eps) > 0 else 0.1
        phip = 2.0 * beta
        phin = 2.0 * beta
        denom = 1.0 - rho - 0.5*(phip + phin)
        if denom <= floor_eps:
            denom = floor_eps + 1e-4
        p_bar = 1.0; n_bar = 1.0
        p0 = float(np.clip(denom * p_bar, p0_lo + 1e-8, p0_hi - 1e-8))
        n0 = float(np.clip(denom * n_bar, p0_lo + 1e-8, p0_hi - 1e-8))
        return np.array([p0, n0, rho, phip, phin, sigp, sign], dtype=float)

    def _sample_vol():
        return _sample_vol_safe(
            rng,
            p0_bounds=(p0_lo, p0_hi),
            rho_bounds=(rho_lo, rho_hi),
            phi_bounds=(phi_lo, phi_hi),
            sigma_bounds=(sig_lo, sig_hi),
            floor_eps=floor_eps,
            max_tries=200
        )

    # ---------- optimize (multi-start SLSQP) ----------
    start_values = [
        np.concatenate([_sample_mean(), _sample_vol()])
        for _ in range(int(n_starts))
    ]
    results = _run_multistart_minimize(
        start_values,
        _negloglik,
        method='SLSQP',
        bounds=bounds_full,
        constraints=constraints,
        options={'maxiter': int(maxiter), 'ftol': float(tol)},
        start_n_jobs=start_n_jobs,
    )

    best, best_fun = None, np.inf
    last_error = None
    for opt, exc in results:
        if exc is not None:
            last_error = exc
            continue
        if (
            _optimizer_result_eligible(opt)
            and (
                best is None
                or opt.fun < best_fun
            )
        ):
            best_fun = opt.fun
            best = opt

    if best is None or (not np.isfinite(best_fun)) or best_fun >= big_penalty:
        detail = ""
        if last_error is not None:
            detail = f" Last error: {type(last_error).__name__}: {last_error}"
        raise RuntimeError(f"All starts failed to converge to a finite eligible objective value.{detail}")

    params = best.x
    ll = -best.fun
    k_params = len(params)
    AIC = 2*k_params - 2*ll
    BIC = np.log(N_obs)*k_params - 2*ll

    # ---------- SEs ----------
    def _sym(A): return 0.5 * (A + A.T)

    def _safe_inv_with_ridge(A, ridge0=1e-8, max_tries=6):
        A = _sym(A)
        I = np.eye(A.shape[0])
        ridge = float(ridge0)
        for _ in range(max_tries):
            try:
                return np.linalg.inv(A + ridge * I), ridge, False
            except np.linalg.LinAlgError:
                ridge *= 10.0
        return np.linalg.pinv(A), ridge, True

    se = np.full(k_params, np.nan, dtype=float)
    used_ridge = 0.0
    used_pseudo = False
    used_opg_fallback = False

    if compute_se:
        # Larger epsilon for piecewise-smooth objective
        H = approx_hess(params, _negloglik, epsilon=1e-5)
        H = _sym(H)
        H_inv, used_ridge, used_pseudo = _safe_inv_with_ridge(H)

        # Per-observation scores (N_obs x k)
        scores = _central_diff_scores(params, _ind_negloglik_vec, bounds_full, rel=1e-4, absmin=1e-6)
        OPG = scores.T @ scores

        # Observed-info fallback if OPG is tiny/ill-conditioned
        opg_scale = np.linalg.norm(OPG) / max(1, OPG.size)
        if (not np.isfinite(opg_scale)) or (opg_scale < 1e-8):
            cov = H_inv.copy()
            used_opg_fallback = True
        else:
            cov = H_inv @ _sym(OPG) @ H_inv
            used_opg_fallback = False

        cov = _sym(cov)
        # PSD projection
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        cov = (V * w) @ V.T
        se = np.sqrt(np.diag(cov))

    # ---------- summary ----------
    if compute_se and used_pseudo:
        print("[warn] Hessian singular; using pseudoinverse for covariance.")
    elif compute_se and used_ridge > 1e-8:
        print(f"[warn] Hessian near-singular; used ridge λ={used_ridge:.1e}.")
    if compute_se and used_opg_fallback:
        print("[warn] OPG nearly zero/ill-conditioned; using observed-information (H^{-1}) for covariance.")

    if print_summary:
        print("\n" + "-"*68)
        print("BEGE (Asymmetric constants & sigmas; shared GJR coefficients)")
        print("-"*68)
        if compute_se:
            print(f"{'Parameter':<20}{'Estimate':>14}{'Std. Err.':>14}{'t-Stat':>14}")
        else:
            print(f"{'Parameter':<20}{'Estimate':>14}")
        print("-"*68)
        if compute_se:
            for nm, val, err in zip(names_full, params, se):
                t = np.nan if err <= 0 else (val/err)
                print(f"{nm:<20}{val:>14.6f}{err:>14.6f}{t:>14.3f}")
        else:
            for nm, val in zip(names_full, params):
                print(f"{nm:<20}{val:>14.6f}")
        print("-"*68)
        print(f"{'LogLik':<20}{ll:>14.6f}")
        print(f"{'AIC':<20}{AIC:>14.6f}")
        print(f"{'BIC':<20}{BIC:>14.6f}")
        print("-"*68)

    return {
        'opt': best,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'names': names_full,
        'compute_se': bool(compute_se),
    }


def ID_GARCH(
    Y, X=None, mean_type='ARX(1,1)',
    n_starts=50, maxiter=800, tol=1e-8, random_state=None,
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.0, 10.0),
    rho_bounds=(0.0, 1.0),
    phi_bounds=(0.0, 2.0),
    floor_eps=1e-6,
    print_summary=True,
    compute_se=False,
    initial_params=None,
    enforce_constraints=True,
    density_hyperu_method='scipy_approx',
    big_penalty=1e12,
    big_vec_penalty=1e6,
    enforce_variance_bounds=True,
    start_n_jobs=None,
):
    """
    Inflation/Deflation BEGE-GJR.

    Parameter order:
      [mean params] +
      [p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigma_p, sigma_n]

    By default the likelihood enforces the documented persistence-plus-loading
    bounds.
    The optimizer uses a SciPy/asymptotic BEGE-density fallback by default to
    avoid slow high-precision calls at rejected trial points.
    """
    import numpy as np
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_hess
    from types import SimpleNamespace

    rng = np.random.default_rng(random_state)
    Y = np.asarray(Y, dtype=float)
    N_obs = int(Y.shape[0])

    def _x_len(x):
        if x is None:
            return len(Y)
        if isinstance(x, pd.DataFrame):
            return len(x)
        if isinstance(x, dict):
            return min(len(v) for v in x.values()) if x else len(Y)
        return len(x)

    def _slice_x(x, n):
        if x is None:
            return None
        if isinstance(x, pd.DataFrame):
            return x.iloc[:n].copy()
        if isinstance(x, dict):
            return {k: np.asarray(v, dtype=float)[:n] for k, v in x.items()}
        return np.asarray(x, dtype=float)[:n]

    if X is not None:
        n = min(len(Y), _x_len(X))
        Y = Y[:n]
        X = _slice_x(X, n)
        N_obs = int(n)

    def _make_residual_function(y, x, model):
        y, data = _coerce_mean_inputs(y, x)

        if model == 'constant':
            return lambda params: y

        if model == 'ARX(1,1)':
            if {"Inflation_lag_1", "SPF"}.issubset(data):
                inflation_lag_1 = data["Inflation_lag_1"]
                spf = data["SPF"]

                def _resids(params):
                    c, rho_1, phi_1 = params
                    return y - (c + rho_1 * inflation_lag_1 + phi_1 * spf)
                return _resids

            if "SPF" in data:
                spf = data["SPF"]

                def _resids(params):
                    c, rho_1, phi_1 = params
                    y_pred = np.empty_like(y)
                    y_pred[0] = c + phi_1 * spf[0]
                    y_pred[1:] = c + rho_1 * y[:-1] + phi_1 * spf[1:]
                    return y - y_pred
                return _resids

        if model == 'ARX(2,1)':
            if {"Inflation_lag_1", "Inflation_lag_2", "SPF"}.issubset(data):
                inflation_lag_1 = data["Inflation_lag_1"]
                inflation_lag_2 = data["Inflation_lag_2"]
                spf = data["SPF"]

                def _resids(params):
                    c, rho_1, rho_2, phi_1 = params
                    return y - (
                        c
                        + rho_1 * inflation_lag_1
                        + rho_2 * inflation_lag_2
                        + phi_1 * spf
                    )
                return _resids

            if "SPF" in data:
                spf = data["SPF"]

                def _resids(params):
                    c, rho_1, rho_2, phi_1 = params
                    y_pred = np.empty_like(y)
                    y_pred[0] = c + phi_1 * spf[0]
                    y_pred[1] = c + rho_1 * y[0] + phi_1 * spf[1]
                    y_pred[2:] = c + rho_1 * y[1:-1] + rho_2 * y[:-2] + phi_1 * spf[2:]
                    return y - y_pred
                return _resids

        if model == 'ARX(2,2)':
            if {"Inflation_lag_1", "Inflation_lag_2", "SPF", "SPF_lag_1"}.issubset(data):
                inflation_lag_1 = data["Inflation_lag_1"]
                inflation_lag_2 = data["Inflation_lag_2"]
                spf = data["SPF"]
                spf_lag_1 = data["SPF_lag_1"]

                def _resids(params):
                    c, rho_1, rho_2, phi_1, phi_2 = params
                    return y - (
                        c
                        + rho_1 * inflation_lag_1
                        + rho_2 * inflation_lag_2
                        + phi_1 * spf
                        + phi_2 * spf_lag_1
                    )
                return _resids

            if "SPF" in data:
                spf = data["SPF"]

                def _resids(params):
                    c, rho_1, rho_2, phi_1, phi_2 = params
                    y_pred = np.empty_like(y)
                    y_pred[0] = c + phi_1 * spf[0]
                    y_pred[1] = c + rho_1 * y[0] + phi_1 * spf[1] + phi_2 * spf[0]
                    y_pred[2:] = (
                        c
                        + rho_1 * y[1:-1]
                        + rho_2 * y[:-2]
                        + phi_1 * spf[2:]
                        + phi_2 * spf[1:-1]
                    )
                    return y - y_pred
                return _resids

        raise ValueError(f"{model} has incompatible X inputs.")

    def _get_mean_spec(y, model):
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if model == 'constant':
            return mean_const, 0, [], [], []
        if model == 'ARX(1,1)':
            return (
                mean_ARX11,
                3,
                [(ymin, ymax), (-0.999, 0.999), (-10, 10)],
                ['c', 'rho_1', 'phi_1'],
                [
                    (0.0824 - 2 * 0.086, 0.0824 + 2 * 0.086),
                    (0.3005 - 2 * 0.112, 0.3005 + 2 * 0.112),
                    (0.7337 - 2 * 0.167, 0.7337 + 2 * 0.167),
                ],
            )
        if model == 'ARX(2,1)':
            return (
                mean_ARX21,
                4,
                [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)],
                ['c', 'rho_1', 'rho_2', 'phi_1'],
                [
                    (0.0897 - 2 * 0.086, 0.0897 + 2 * 0.086),
                    (0.2892 - 2 * 0.098, 0.2892 + 2 * 0.098),
                    (0.0834 - 2 * 0.126, 0.0834 + 2 * 0.126),
                    (0.6385 - 2 * 0.251, 0.6385 + 2 * 0.251),
                ],
            )
        if model == 'ARX(2,2)':
            return (
                mean_ARX22,
                5,
                [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)],
                ['c', 'rho_1', 'rho_2', 'phi_1', 'phi_2'],
                [
                    (0.0856 - 2 * 0.086, 0.0856 + 2 * 0.086),
                    (0.2992 - 2 * 0.106, 0.2992 + 2 * 0.106),
                    (0.0914 - 2 * 0.125, 0.0914 + 2 * 0.125),
                    (0.4084 - 2 * 0.433, 0.4084 + 2 * 0.433),
                    (0.2136 - 2 * 0.297, 0.2136 + 2 * 0.297),
                ],
            )
        raise ValueError("Invalid mean_type")

    mean_model, num_m, bounds_mean, names_mean, mean_ranges = _get_mean_spec(Y, mean_type)
    residual_function = _make_residual_function(Y, X, mean_type)

    sig_lo, sig_hi = sigma_bounds
    p0_lo, p0_hi = p0n0_bounds
    rho_lo, rho_hi = rho_bounds
    phi_lo, phi_hi = phi_bounds

    bounds_vol = [
        (p0_lo, p0_hi),
        (p0_lo, p0_hi),
        (rho_lo, rho_hi),
        (rho_lo, rho_hi),
        (phi_lo, phi_hi),
        (phi_lo, phi_hi),
        (sig_lo, sig_hi),
        (sig_lo, sig_hi),
    ]
    names_vol = ['p0', 'n0', 'rho_p', 'rho_n', 'phi_p_plus', 'phi_n_minus', 'sigma_p', 'sigma_n']
    bounds_full = bounds_mean + bounds_vol
    names_full = names_mean + names_vol

    def _constraints_ok(p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign):
        if not enforce_constraints:
            return True
        if rho_p + 0.5 * phi_p_plus >= 1.0 - floor_eps:
            return False
        if rho_n + 0.5 * phi_n_minus >= 1.0 - floor_eps:
            return False
        return True

    def _series_ok(pseries, nseries):
        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return False
        return True

    def _negloglik(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign = theta[num_m:]
        if not _constraints_ok(p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign):
            return float(big_penalty)

        res = residual_function(pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), 0.0), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), 0.0, float(phi_n_minus)), float(sign))

        if not _series_ok(pseries, nseries):
            return float(big_penalty)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return float(big_penalty)

        ll = BEGE_log_density(
            res,
            pseries,
            nseries,
            float(sigp),
            float(sign),
            hyperu_method=density_hyperu_method,
        )
        val = -float(np.sum(ll))
        return val if np.isfinite(val) else float(big_penalty)

    def _ind_negloglik_vec(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign = theta[num_m:]
        if not _constraints_ok(p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        res = residual_function(pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), 0.0), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), 0.0, float(phi_n_minus)), float(sign))

        if not _series_ok(pseries, nseries):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        v = -BEGE_log_density(
            res,
            pseries,
            nseries,
            float(sigp),
            float(sign),
            hyperu_method=density_hyperu_method,
        )
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.shape[0] != N_obs:
            v = np.full(N_obs, float(v.ravel()[0]))
        if not np.all(np.isfinite(v)):
            v = np.full(N_obs, float(big_vec_penalty))
        return v

    def _sym(A):
        return 0.5 * (A + A.T)

    def _safe_inv_with_ridge(A, ridge0=1e-8, max_tries=6):
        A = _sym(A)
        I = np.eye(A.shape[0])
        ridge = float(ridge0)
        for _ in range(max_tries):
            try:
                return np.linalg.inv(A + ridge * I), ridge, False
            except np.linalg.LinAlgError:
                ridge *= 10.0
        return np.linalg.pinv(A), ridge, True

    def _project_to_bounds(x, bounds):
        out = np.array(x, dtype=float)
        tiny = 1e-10
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None:
                out[j] = max(out[j], lo + tiny)
            if hi is not None:
                out[j] = min(out[j], hi - tiny)
        return out

    def _central_diff_scores(theta, f_per_obs, bounds, rel=1e-4, absmin=1e-6):
        theta = np.asarray(theta, dtype=float)
        k = theta.size
        f0 = f_per_obs(theta)
        J = np.empty((N_obs, k), dtype=float)
        h = np.maximum(absmin, rel * np.maximum(1.0, np.abs(theta)))

        for j in range(k):
            th_p = theta.copy()
            th_m = theta.copy()
            th_p[j] += h[j]
            th_m[j] -= h[j]
            th_p = _project_to_bounds(th_p, bounds)
            th_m = _project_to_bounds(th_m, bounds)

            fp = np.asarray(f_per_obs(th_p), dtype=float).reshape(-1)
            fm = np.asarray(f_per_obs(th_m), dtype=float).reshape(-1)
            if fp.shape[0] != N_obs:
                fp = np.full(N_obs, float(fp.ravel()[0]))
            if fm.shape[0] != N_obs:
                fm = np.full(N_obs, float(fm.ravel()[0]))

            denom = float(th_p[j] - th_m[j])
            if denom == 0.0:
                th_p = theta.copy()
                th_p[j] += h[j]
                th_p = _project_to_bounds(th_p, bounds)
                fp = np.asarray(f_per_obs(th_p), dtype=float).reshape(-1)
                if fp.shape[0] != N_obs:
                    fp = np.full(N_obs, float(fp.ravel()[0]))
                fm = f0
                denom = float(th_p[j] - theta[j])

            J[:, j] = (fp - fm) / denom
        return J

    def _sample_mean():
        if num_m == 0:
            return np.array([], dtype=float)
        vals = []
        for (low, high), (b_low, b_high) in zip(mean_ranges, bounds_mean):
            lo = max(low, b_low if b_low is not None else low)
            hi = min(high, b_high if b_high is not None else high)
            vals.append(rng.uniform(lo, hi))
        return np.array(vals, dtype=float)

    def _sample_vol():
        if not enforce_constraints:
            return np.array(
                [
                    rng.uniform(p0_lo, p0_hi),
                    rng.uniform(p0_lo, p0_hi),
                    rng.uniform(rho_lo, rho_hi),
                    rng.uniform(rho_lo, rho_hi),
                    rng.uniform(phi_lo, phi_hi),
                    rng.uniform(phi_lo, phi_hi),
                    rng.uniform(sig_lo, sig_hi),
                    rng.uniform(sig_lo, sig_hi),
                ],
                dtype=float,
            )

        rho_upper = min(rho_hi, 1.0 - floor_eps - 0.5 * phi_lo)
        if rho_upper <= rho_lo:
            raise ValueError("rho_bounds and phi_bounds leave no feasible stable ID-GARCH starts.")

        for _ in range(500):
            sigp = rng.uniform(sig_lo, sig_hi)
            sign = rng.uniform(sig_lo, sig_hi)
            p0 = rng.uniform(p0_lo, p0_hi)
            n0 = rng.uniform(p0_lo, p0_hi)

            rho_p = rng.uniform(rho_lo, rho_upper)
            phi_p_upper = min(phi_hi, 2.0 * (1.0 - floor_eps - rho_p))
            if phi_p_upper <= phi_lo:
                continue
            phi_p_plus = rng.uniform(phi_lo, phi_p_upper)

            rho_n = rng.uniform(rho_lo, rho_upper)
            phi_n_upper = min(phi_hi, 2.0 * (1.0 - floor_eps - rho_n))
            if phi_n_upper <= phi_lo:
                continue
            phi_n_minus = rng.uniform(phi_lo, phi_n_upper)

            vals = np.array([p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign], dtype=float)
            if _constraints_ok(*vals):
                return vals

        sig_default = min(max(0.4, sig_lo), sig_hi)
        p0_default = min(max(0.5, p0_lo), p0_hi)
        n0_default = min(max(0.5, p0_lo), p0_hi)
        rho_default = min(max(0.3, rho_lo), rho_upper)
        phi_default = min(0.5, phi_hi, 2.0 * (1.0 - floor_eps - rho_default))
        phi_default = max(phi_lo, phi_default)
        return np.array(
            [
                p0_default,
                n0_default,
                rho_default,
                rho_default,
                phi_default,
                phi_default,
                sig_default,
                sig_default,
            ],
            dtype=float,
        )

    best = None
    best_fun = np.inf

    if initial_params is not None and int(n_starts) <= 0:
        params0 = np.asarray(initial_params, dtype=float)
        if params0.shape[0] != len(bounds_full):
            raise ValueError(f"initial_params has length {params0.shape[0]}, expected {len(bounds_full)}.")
        params0 = _project_to_bounds(params0, bounds_full)
        best_fun = _negloglik(params0)
        best = SimpleNamespace(
            x=params0,
            fun=best_fun,
            success=np.isfinite(best_fun) and best_fun < big_penalty,
            status=0,
            message="Evaluated supplied initial_params without re-optimizing.",
        )
    else:
        start_values = []
        if initial_params is not None:
            params0 = np.asarray(initial_params, dtype=float)
            if params0.shape[0] != len(bounds_full):
                raise ValueError(f"initial_params has length {params0.shape[0]}, expected {len(bounds_full)}.")
            start_values.append(_project_to_bounds(params0, bounds_full))
        for _ in range(max(0, int(n_starts))):
            start_values.append(np.concatenate([_sample_mean(), _sample_vol()]))

        if not start_values:
            raise ValueError("Set n_starts >= 1 or provide initial_params.")

        results = _run_multistart_minimize(
            start_values,
            _negloglik,
            method='L-BFGS-B',
            bounds=bounds_full,
            options={'maxiter': int(maxiter), 'ftol': float(tol)},
            start_n_jobs=start_n_jobs,
        )
        for opt, exc in results:
            if exc is not None:
                continue
            if _optimizer_result_eligible(opt) and opt.fun < best_fun:
                best_fun = opt.fun
                best = opt

    if best is None or (not np.isfinite(best_fun)) or best_fun >= big_penalty:
        raise RuntimeError("All starts failed to converge to a finite eligible objective value.")

    params = np.asarray(best.x, dtype=float)
    ll = -float(best.fun)
    AIC = 2 * len(params) - 2 * ll
    BIC = np.log(N_obs) * len(params) - 2 * ll

    se = np.full(len(params), np.nan, dtype=float)
    used_ridge = 0.0
    used_pseudo = False
    used_opg_fallback = False

    if compute_se:
        H = approx_hess(params, _negloglik, epsilon=1e-5)
        H = _sym(H)
        H_inv, used_ridge, used_pseudo = _safe_inv_with_ridge(H)

        scores = _central_diff_scores(params, _ind_negloglik_vec, bounds_full, rel=1e-4, absmin=1e-6)
        OPG = scores.T @ scores

        opg_scale = np.linalg.norm(OPG) / max(1, OPG.size)
        if (not np.isfinite(opg_scale)) or (opg_scale < 1e-8):
            cov = H_inv.copy()
            used_opg_fallback = True
        else:
            cov = H_inv @ _sym(OPG) @ H_inv
            used_opg_fallback = False

        cov = _sym(cov)
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        cov = (V * w) @ V.T
        se = np.sqrt(np.diag(cov))

    if compute_se and used_pseudo:
        print("[warn] Hessian singular; using pseudoinverse for covariance.")
    elif compute_se and used_ridge > 1e-8:
        print(f"[warn] Hessian near-singular; used ridge lambda={used_ridge:.1e}.")
    if compute_se and used_opg_fallback:
        print("[warn] OPG nearly zero/ill-conditioned; using observed-information inverse for covariance.")

    if print_summary:
        print("\n" + "-" * 72)
        print("ID_GARCH (Inflation/Deflation asymmetry)")
        print("-" * 72)
        if compute_se:
            print(f"{'Parameter':<18}{'Estimate':>14}{'Std. Err.':>14}{'t-Stat':>14}")
            print("-" * 72)
            for nm, val, err in zip(names_full, params, se):
                t = np.nan if err <= 0 else (val / err)
                print(f"{nm:<18}{val:>14.6f}{err:>14.6f}{t:>14.3f}")
        else:
            print(f"{'Parameter':<18}{'Estimate':>14}")
            print("-" * 72)
            for nm, val in zip(names_full, params):
                print(f"{nm:<18}{val:>14.6f}")
        print("-" * 72)
        print(f"{'LogLik':<18}{ll:>14.6f}")
        print(f"{'AIC':<18}{AIC:>14.6f}")
        print(f"{'BIC':<18}{BIC:>14.6f}")
        print("-" * 72)

    return {
        'opt': best,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'names': names_full,
        'compute_se': bool(compute_se),
    }



def BEGE_FullGJR_MLE(
    Y, X=None, mean_type='ARX(1,1)',
    n_starts=50, maxiter=800, tol=1e-8, random_state=None,
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.0, 10.0),
    rho_bounds=(0.0, 1.0),
    phi_bounds=(0.0, 2.0),
    floor_eps=1e-6,
    print_summary=True,
    big_penalty=1e12,           # scalar objective penalty
    big_vec_penalty=1e6,        # per-observation penalty (vector version)
    compute_se=False,
    density_hyperu_method='scipy_approx',
    enforce_variance_bounds=True,
    start_n_jobs=None,
):
    """
    BEGE with FULL GJR recursions (separate parameters for p_t and n_t).

    Parameter order:
      [mean params] +
      [p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigma_p, sigma_n]

    Behavior in this version:
      • Stability is enforced for both shape processes:
        rho + 0.5*(phi_plus + phi_minus) < 1.
      • Random starts are sampled from the feasible stability/variance region.

    Returns:
      {'opt','params','se','AIC','BIC','loglik','names'}
    """
    import numpy as np
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_hess

    rng = np.random.default_rng(random_state)
    Y, X = _align_mean_sample(Y, X)
    N_obs = int(Y.shape[0])

    # ---------- mean spec ----------
    def _get_mean_spec(Y, mean_type):
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        if mean_type == 'constant':
            mean_model, num_m = mean_const, 0
            bounds_mean, names_mean, ranges = [], [], []
        elif mean_type == 'ARX(1,1)':
            mean_model, num_m = mean_ARX11, 3
            bounds_mean = [(ymin, ymax), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'SPF']
            ranges = [
                (0.0720 - .2*0.086, 0.0720 + .2*0.086),
                (0.2881 - .2*0.073, 0.2881 + .2*0.073),
                (0.7508 - .2*0.125, 0.7508 + .2*0.125),
            ]
            # ranges = [
            #     (0.0720 - 2*0.086, 0.0720 + 2*0.086),
            #     (0.2881 - 2*0.073, 0.2881 + 2*0.073),
            #     (0.7508 - 2*0.125, 0.7508 + 2*0.125),
            # ]
            
        elif mean_type == 'ARX(2,1)':
            mean_model, num_m = mean_ARX21, 4
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
            ranges = [
                (0.0792 - 2*0.087, 0.0792 + 2*0.087),
                (0.2793 - 2*0.073, 0.2793 + 2*0.073),
                (0.0728 - 2*0.080, 0.0728 + 2*0.080),
                (0.6661 - 2*0.156, 0.6661 + 2*0.156),
            ]
        elif mean_type == 'ARX(2,2)':
            mean_model, num_m = mean_ARX22, 5
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']
            ranges = [
                (0.0761 - 2*0.087, 0.0761 + 2*0.087),
                (0.2880 - 2*0.076, 0.2880 + 2*0.076),
                (0.0799 - 2*0.082, 0.0799 + 2*0.082),
                (0.4720 - 2*0.459, 0.4720 + 2*0.459),
                (0.1793 - 2*0.399, 0.1793 + 2*0.399),
            ]
        else:
            raise ValueError("Invalid mean_type")
        return mean_model, num_m, bounds_mean, names_mean, ranges

    mean_model, num_m, bounds_mean, names_mean, mean_ranges = _get_mean_spec(Y, mean_type)
    residual_function = _make_residual_function(Y, X, mean_type)

    # ---------- bounds & names ----------
    (sig_lo, sig_hi) = sigma_bounds
    (p0_lo,  p0_hi)  = p0n0_bounds
    (rho_lo, rho_hi) = rho_bounds
    (phi_lo, phi_hi) = phi_bounds

    bounds_vol = [
        (p0_lo, p0_hi),     # p0
        (p0_lo, p0_hi),     # n0
        (rho_lo, rho_hi),   # rho_p
        (rho_lo, rho_hi),   # rho_n
        (phi_lo, phi_hi),   # phi_p_plus
        (phi_lo, phi_hi),   # phi_p_minus
        (phi_lo, phi_hi),   # phi_n_plus
        (phi_lo, phi_hi),   # phi_n_minus
        (sig_lo, sig_hi),   # sigma_p
        (sig_lo, sig_hi),   # sigma_n
    ]
    names_vol = ['p0', 'n0', 'rho_p', 'rho_n', 'phi_p⁺', 'phi_p⁻', 'phi_n⁺', 'phi_n⁻', 'σ₊', 'σ₋']

    bounds_full = bounds_mean + bounds_vol
    names_full  = names_mean + names_vol


    ####################################
    ##### Constraints added in Dec 8 2025
    ####################################
    # ---------- nonlinear constraints for SLSQP ----------
    # helper to unpack volatility parameters from full theta
    def _unpack_vol(theta):
        p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, \
        phi_n_plus, phi_n_minus, sigp, sign = theta[num_m:]
        return p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign

    def _constraints_ok(p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus,
                        phi_n_plus, phi_n_minus, sigp, sign):
        vals = np.asarray(
            [p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus,
             phi_n_plus, phi_n_minus, sigp, sign],
            dtype=float,
        )
        if not np.all(np.isfinite(vals)):
            return False
        if rho_p + 0.5 * (phi_p_plus + phi_p_minus) >= 1.0 - floor_eps:
            return False
        if rho_n + 0.5 * (phi_n_plus + phi_n_minus) >= 1.0 - floor_eps:
            return False
        return True

    # 1) rho_p + phi_p⁺/2 + phi_p⁻/2 < 1
    def constr_rho_phi_p(theta):
        p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, \
        phi_n_plus, phi_n_minus, sigp, sign = _unpack_vol(theta)
        return 1.0 - floor_eps - (rho_p + 0.5 * (phi_p_plus + phi_p_minus))

    # 2) rho_n + phi_n⁺/2 + phi_n⁻/2 < 1
    def constr_rho_phi_n(theta):
        p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, \
        phi_n_plus, phi_n_minus, sigp, sign = _unpack_vol(theta)
        return 1.0 - floor_eps - (rho_n + 0.5 * (phi_n_plus + phi_n_minus))

    constraints = [
        {'type': 'ineq', 'fun': constr_rho_phi_p},
        {'type': 'ineq', 'fun': constr_rho_phi_n},
    ]

    
    # ---------- objective (total NEG log-lik) ----------
    def _negloglik(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign = theta[num_m:]
        if not _constraints_ok(
            p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus,
            phi_n_plus, phi_n_minus, sigp, sign,
        ):
            return float(big_penalty)

        res = residual_function(pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), float(phi_p_minus)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), float(phi_n_plus), float(phi_n_minus)), float(sign))

        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return float(big_penalty)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return float(big_penalty)

        ll = BEGE_log_density(
            res, pseries, nseries, float(sigp), float(sign),
            hyperu_method=density_hyperu_method,
        )
        val = -float(np.sum(ll))
        if not np.isfinite(val):
            return float(big_penalty)
        return val

    # ---------- per-obs NEG loglik (vector) ----------
    def _ind_negloglik_vec(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign = theta[num_m:]
        if not _constraints_ok(
            p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus,
            phi_n_plus, phi_n_minus, sigp, sign,
        ):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        res = residual_function(pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), float(phi_p_minus)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), float(phi_n_plus), float(phi_n_minus)), float(sign))

        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        v = -BEGE_log_density(
            res, pseries, nseries, float(sigp), float(sign),
            hyperu_method=density_hyperu_method,
        )
        v = np.asarray(v, float).reshape(-1)
        if v.shape[0] != N_obs:
            v = np.full(N_obs, float(v.ravel()[0]))
        if not np.all(np.isfinite(v)):
            v = np.full(N_obs, float(big_vec_penalty))
        return v

    # ---------- helpers for SEs ----------
    def _sym(A): return 0.5*(A + A.T)

    def _safe_inv_with_ridge(A, ridge0=1e-8, max_tries=6):
        A = _sym(A)
        I = np.eye(A.shape[0])
        ridge = float(ridge0)
        for _ in range(max_tries):
            try:
                return np.linalg.inv(A + ridge*I), ridge, False
            except np.linalg.LinAlgError:
                ridge *= 10.0
        return np.linalg.pinv(A), ridge, True

    def _project_to_bounds(x, bounds):
        out = np.array(x, float); tiny = 1e-10
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None: out[j] = max(out[j], lo + tiny)
            if hi is not None: out[j] = min(out[j], hi - tiny)
        return out

    def _central_diff_scores(theta, f_per_obs, bounds, rel=1e-4, absmin=1e-6):
        """
        Jacobian (N_obs x k): row i = score_i(theta).
        Parameter-scaled central differences with bounds projection.
        """
        theta = np.asarray(theta, float)
        k = theta.size
        f0 = f_per_obs(theta)                   # (N_obs,)
        J  = np.empty((N_obs, k), float)
        h  = np.maximum(absmin, rel * np.maximum(1.0, np.abs(theta)))

        for j in range(k):
            th_p = theta.copy(); th_p[j] += h[j]; th_p = _project_to_bounds(th_p, bounds)
            th_m = theta.copy(); th_m[j] -= h[j]; th_m = _project_to_bounds(th_m, bounds)

            fp = np.asarray(f_per_obs(th_p), float).reshape(-1)
            fm = np.asarray(f_per_obs(th_m), float).reshape(-1)
            if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
            if fm.shape[0] != N_obs: fm = np.full(N_obs, float(fm.ravel()[0]))

            denom = float(th_p[j] - th_m[j])
            if denom == 0.0:
                th_p = theta.copy(); th_p[j] += h[j]; th_p = _project_to_bounds(th_p, bounds)
                fp   = np.asarray(f_per_obs(th_p), float).reshape(-1)
                if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
                fm   = f0
                denom = float(th_p[j] - theta[j])

            J[:, j] = (fp - fm) / denom

        return J

    # ---------- free initializer (no stability) ----------
    def _sample_mean():
        if num_m == 0: return np.array([], float)
        return np.array([rng.uniform(a, b) for (a, b) in mean_ranges], dtype=float)


    def _sample_vol_full_stable(
        rng,
        p0_bounds, rho_bounds, phi_bounds, sigma_bounds,
        floor_eps=1e-6,
        max_tries=200
    ):
        """
        Stability-aware sampling for Full GJR (separate p_t and n_t parameters).
        
        Returns: [p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign]
        
        Sampling rule for each component (p and n):
        - rho_p, rho_n ~ U[rho_lo, rho_hi]
        - sigp, sign ~ U[sigma_lo, sigma_hi]
        - p0, n0 ~ U[p0_lo, p0_hi]
        
        For p component:
            - beta_p_plus := phi_p_plus/2 ~ U[0, 1 - rho_p - floor_eps]
            - beta_p_minus := phi_p_minus/2 ~ U[0, 1 - rho_p - floor_eps]
            - phi_p_plus = 2*beta_p_plus, phi_p_minus = 2*beta_p_minus
            - Require: 1 - rho_p - max(beta_p_plus, beta_p_minus) > floor_eps
        
        For n component:
            - beta_n_plus := phi_n_plus/2 ~ U[0, 1 - rho_n - floor_eps]
            - beta_n_minus := phi_n_minus/2 ~ U[0, 1 - rho_n - floor_eps]
            - phi_n_plus = 2*beta_n_plus, phi_n_minus = 2*beta_n_minus
            - Require: 1 - rho_n - max(beta_n_plus, beta_n_minus) > floor_eps
        """
        import numpy as np
        
        p0_lo, p0_hi = p0_bounds
        rho_lo, rho_hi = rho_bounds
        sig_lo, sig_hi = sigma_bounds
        phi_lo, phi_hi = phi_bounds

        rho_upper = min(rho_hi, 1.0 - floor_eps - phi_lo)
        if rho_upper <= rho_lo:
            raise ValueError("rho_bounds and phi_bounds leave no feasible stable Full BEGE starts.")

        for _ in range(max_tries):
            # Sample rho parameters
            rho_p = rng.uniform(rho_lo, rho_upper)
            rho_n = rng.uniform(rho_lo, rho_upper)
            
            # Sample sigma parameters
            sigp = rng.uniform(sig_lo, sig_hi)
            sign = rng.uniform(sig_lo, sig_hi)
            p0 = rng.uniform(p0_lo, p0_hi)
            n0 = rng.uniform(p0_lo, p0_hi)

            # Sample p component phi parameters
            cap_p = min(0.5 * phi_hi, 1.0 - rho_p - floor_eps)
            if cap_p <= 0:
                continue
                
            beta_p_plus = rng.uniform(0.0, cap_p)
            beta_p_minus = rng.uniform(0.0, cap_p)
            phi_p_plus = 2.0 * beta_p_plus
            phi_p_minus = 2.0 * beta_p_minus
            
            # Check p stability
            if not (1.0 - rho_p - max(beta_p_plus, beta_p_minus) > floor_eps):
                continue
            
            
            # Check p denominator
            denom_p = 1.0 - rho_p - 0.5*(phi_p_plus + phi_p_minus)
            if denom_p <= floor_eps:
                continue

            # Sample n component phi parameters
            cap_n = min(0.5 * phi_hi, 1.0 - rho_n - floor_eps)
            if cap_n <= 0:
                continue
                
            beta_n_plus = rng.uniform(0.0, cap_n)
            beta_n_minus = rng.uniform(0.0, cap_n)
            phi_n_plus = 2.0 * beta_n_plus
            phi_n_minus = 2.0 * beta_n_minus
            
            # Check n stability
            if not (1.0 - rho_n - max(beta_n_plus, beta_n_minus) > floor_eps):
                continue
            
            # Check n denominator
            denom_n = 1.0 - rho_n - 0.5*(phi_n_plus + phi_n_minus)
            if denom_n <= floor_eps:
                continue

            vals = np.array([p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, 
                             phi_n_plus, phi_n_minus, sigp, sign], dtype=float)
            if _constraints_ok(*vals):
                return vals

        # Conservative fallback if repeated failures
        rho_p = 0.5
        rho_n = 0.5
        sigp = min(max(0.3, sig_lo), sig_hi)
        sign = min(max(0.3, sig_lo), sig_hi)
        
        # Conservative beta values for p component
        beta_p = 0.25 * (1.0 - rho_p - floor_eps) if (1.0 - rho_p - floor_eps) > 0 else 0.1
        phi_p_plus = 2.0 * beta_p
        phi_p_minus = 2.0 * beta_p
        
        # Conservative beta values for n component
        beta_n = 0.25 * (1.0 - rho_n - floor_eps) if (1.0 - rho_n - floor_eps) > 0 else 0.1
        phi_n_plus = 2.0 * beta_n
        phi_n_minus = 2.0 * beta_n
        
        # Ensure denominators are valid
        denom_p = 1.0 - rho_p - 0.5*(phi_p_plus + phi_p_minus)
        denom_n = 1.0 - rho_n - 0.5*(phi_n_plus + phi_n_minus)
        if denom_p <= floor_eps:
            denom_p = floor_eps + 1e-4
        if denom_n <= floor_eps:
            denom_n = floor_eps + 1e-4
        
        # Set conservative p0, n0
        p_bar = 1.0
        n_bar = 1.0
        p0 = float(np.clip(denom_p * p_bar, p0_lo + 1e-8, p0_hi - 1e-8))
        n0 = float(np.clip(denom_n * n_bar, p0_lo + 1e-8, p0_hi - 1e-8))
        
        return np.array([p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, 
                        phi_n_plus, phi_n_minus, sigp, sign], dtype=float)


    # Usage: Replace the _sample_vol() function definition with:
    def _sample_vol():
        return _sample_vol_full_stable(
            rng,
            p0_bounds=(p0_lo, p0_hi),
            rho_bounds=(rho_lo, rho_hi),
            phi_bounds=(phi_lo, phi_hi),
            sigma_bounds=(sig_lo, sig_hi),
            floor_eps=floor_eps,
            max_tries=200
        )
    # ---------- optimize (multi-start SLSQP) ----------
    start_values = [
        np.concatenate([_sample_mean(), _sample_vol()])
        for _ in range(int(n_starts))
    ]
    results = _run_multistart_minimize(
        start_values,
        _negloglik,
        method='SLSQP',
        bounds=bounds_full,
        constraints=constraints,
        options={'maxiter': int(maxiter), 'ftol': float(tol)},
        start_n_jobs=start_n_jobs,
    )

    best, best_fun = None, np.inf
    last_error = None
    for opt, exc in results:
        if exc is not None:
            last_error = exc
            continue
        if (
            _optimizer_result_eligible(opt)
            and (
                best is None
                or opt.fun < best_fun
            )
        ):
            best_fun = opt.fun
            best = opt

    if best is None or (not np.isfinite(best_fun)) or best_fun >= big_penalty:
        detail = ""
        if last_error is not None:
            detail = f" Last error: {type(last_error).__name__}: {last_error}"
        raise RuntimeError("All starts failed to converge to a finite eligible objective value. "
                           f"Consider tightening bounds, increasing big_penalty, or scaling data.{detail}")

    params = best.x
    ll     = -best.fun
    AIC    = 2*len(params) - 2*ll
    BIC    = np.log(N_obs)*len(params) - 2*ll

    se = np.full(len(params), np.nan, dtype=float)
    used_ridge = 0.0
    used_pseudo = False
    used_opg_fallback = False

    if compute_se:
        # ---------- SEs ----------
        H = approx_hess(params, _negloglik, epsilon=1e-5)
        H = _sym(H)
        H_inv, used_ridge, used_pseudo = _safe_inv_with_ridge(H)

        scores = _central_diff_scores(params, _ind_negloglik_vec, bounds_full, rel=1e-4, absmin=1e-6)
        OPG    = scores.T @ scores

        opg_scale = np.linalg.norm(OPG) / max(1, OPG.size)
        if (not np.isfinite(opg_scale)) or (opg_scale < 1e-8):
            cov = H_inv.copy()
            used_opg_fallback = True
        else:
            cov = H_inv @ _sym(OPG) @ H_inv
            used_opg_fallback = False

        cov = _sym(cov)
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        cov = (V * w) @ V.T
        se  = np.sqrt(np.diag(cov))

    # ---------- summary ----------
    if compute_se and used_pseudo:
        print("[warn] Hessian singular; using pseudoinverse for covariance.")
    elif compute_se and used_ridge > 1e-8:
        print(f"[warn] Hessian near-singular; used ridge λ={used_ridge:.1e}.")
    if compute_se and used_opg_fallback:
        print("[warn] OPG nearly zero/ill-conditioned; using observed-information (H^{-1}) for covariance.")

    if print_summary:
        print("\n" + "-"*72)
        print("BEGE (Full GJR: separate p_t and n_t) — hard overflow penalty")
        print("-"*72)
        if compute_se:
            print(f"{'Parameter':<18}{'Estimate':>14}{'Std. Err.':>14}{'t-Stat':>14}")
        else:
            print(f"{'Parameter':<18}{'Estimate':>14}")
        print("-"*72)
        if compute_se:
            for nm, val, err in zip(names_full, params, se):
                t = np.nan if err <= 0 else (val/err)
                print(f"{nm:<18}{val:>14.6f}{err:>14.6f}{t:>14.3f}")
        else:
            for nm, val in zip(names_full, params):
                print(f"{nm:<18}{val:>14.6f}")
        print("-"*72)
        print(f"{'LogLik':<18}{ll:>14.6f}")
        print(f"{'AIC':<18}{AIC:>14.6f}")
        print(f"{'BIC':<18}{BIC:>14.6f}")
        print("-"*72)

    return {
        'opt': best,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'names': names_full,
        'compute_se': bool(compute_se),
    }




def _BEGE_OneConstantFullGJR_MLE(
    Y, X=None, mean_type='ARX(1,1)',
    n_starts=50, maxiter=800, tol=1e-8, random_state=None,
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.0, 10.0),
    rho_bounds=(0.0, 1.0),
    phi_bounds=(0.0, 2.0),
    floor_eps=1e-6,
    print_summary=True,
    big_penalty=1e12,
    big_vec_penalty=1e6,
    compute_se=False,
    density_hyperu_method='scipy_approx',
    enforce_variance_bounds=True,
    start_n_jobs=None,
    constant_shape='p',
):
    """
    BEGE with one constant shape process and one full GJR shape process.

    constant_shape='p':
      [mean params] + [p0, n0, rho_n, phi_n_plus, phi_n_minus, sigma_p, sigma_n]
      with p_t = p0 and full GJR dynamics for n_t.

    constant_shape='n':
      [mean params] + [p0, n0, rho_p, phi_p_plus, phi_p_minus, sigma_p, sigma_n]
      with n_t = n0 and full GJR dynamics for p_t.
    """
    import numpy as np
    from statsmodels.tools.numdiff import approx_hess

    if constant_shape not in {'p', 'n'}:
        raise ValueError("constant_shape must be either 'p' or 'n'.")

    rng = np.random.default_rng(random_state)
    Y, X = _align_mean_sample(Y, X)
    N_obs = int(Y.shape[0])

    def _get_mean_spec(Y, mean_type):
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        if mean_type == 'constant':
            mean_model, num_m = mean_const, 0
            bounds_mean, names_mean, ranges = [], [], []
        elif mean_type == 'ARX(1,1)':
            mean_model, num_m = mean_ARX11, 3
            bounds_mean = [(ymin, ymax), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'SPF']
            ranges = [
                (0.0720 - .2*0.086, 0.0720 + .2*0.086),
                (0.2881 - .2*0.073, 0.2881 + .2*0.073),
                (0.7508 - .2*0.125, 0.7508 + .2*0.125),
            ]
        elif mean_type == 'ARX(2,1)':
            mean_model, num_m = mean_ARX21, 4
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
            ranges = [
                (0.0792 - 2*0.087, 0.0792 + 2*0.087),
                (0.2793 - 2*0.073, 0.2793 + 2*0.073),
                (0.0728 - 2*0.080, 0.0728 + 2*0.080),
                (0.6661 - 2*0.156, 0.6661 + 2*0.156),
            ]
        elif mean_type == 'ARX(2,2)':
            mean_model, num_m = mean_ARX22, 5
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']
            ranges = [
                (0.0761 - 2*0.087, 0.0761 + 2*0.087),
                (0.2880 - 2*0.076, 0.2880 + 2*0.076),
                (0.0799 - 2*0.082, 0.0799 + 2*0.082),
                (0.4720 - 2*0.459, 0.4720 + 2*0.459),
                (0.1793 - 2*0.399, 0.1793 + 2*0.399),
            ]
        else:
            raise ValueError("Invalid mean_type")
        return mean_model, num_m, bounds_mean, names_mean, ranges

    mean_model, num_m, bounds_mean, names_mean, mean_ranges = _get_mean_spec(Y, mean_type)
    residual_function = _make_residual_function(Y, X, mean_type)

    (sig_lo, sig_hi) = sigma_bounds
    (p0_lo, p0_hi) = p0n0_bounds
    (rho_lo, rho_hi) = rho_bounds
    (phi_lo, phi_hi) = phi_bounds

    if constant_shape == 'p':
        bounds_vol = [
            (p0_lo, p0_hi),
            (p0_lo, p0_hi),
            (rho_lo, rho_hi),
            (phi_lo, phi_hi),
            (phi_lo, phi_hi),
            (sig_lo, sig_hi),
            (sig_lo, sig_hi),
        ]
        names_vol = ['p0', 'n0', 'rho_n', 'phi_n+', 'phi_n-', 'sigma_p', 'sigma_n']
    else:
        bounds_vol = [
            (p0_lo, p0_hi),
            (p0_lo, p0_hi),
            (rho_lo, rho_hi),
            (phi_lo, phi_hi),
            (phi_lo, phi_hi),
            (sig_lo, sig_hi),
            (sig_lo, sig_hi),
        ]
        names_vol = ['p0', 'n0', 'rho_p', 'phi_p+', 'phi_p-', 'sigma_p', 'sigma_n']

    bounds_full = bounds_mean + bounds_vol
    names_full = names_mean + names_vol

    def _unpack_vol(theta):
        return tuple(theta[num_m:])

    def _constraints_ok(*vol):
        vals = np.asarray(vol, dtype=float)
        if not np.all(np.isfinite(vals)):
            return False
        if vals[0] <= floor_eps or vals[1] <= floor_eps:
            return False
        if vals[-2] <= 0.0 or vals[-1] <= 0.0:
            return False
        rho = vals[2]
        phi_plus = vals[3]
        phi_minus = vals[4]
        return bool(rho + 0.5 * (phi_plus + phi_minus) < 1.0 - floor_eps)

    def constr_shape_positive_p(theta):
        return _unpack_vol(theta)[0] - floor_eps

    def constr_shape_positive_n(theta):
        return _unpack_vol(theta)[1] - floor_eps

    def constr_dynamic_persistence(theta):
        vol = _unpack_vol(theta)
        return 1.0 - floor_eps - (vol[2] + 0.5 * (vol[3] + vol[4]))

    constraints = [
        {'type': 'ineq', 'fun': constr_shape_positive_p},
        {'type': 'ineq', 'fun': constr_shape_positive_n},
        {'type': 'ineq', 'fun': constr_dynamic_persistence},
    ]

    def _shape_paths(theta, res):
        vol = _unpack_vol(theta)
        if constant_shape == 'p':
            p0, n0, rho_n, phi_n_plus, phi_n_minus, sigp, sign = vol
            pseries = np.full_like(res, float(p0), dtype=float)
            nseries = gjr_recursion(
                res,
                (float(n0), float(rho_n), float(phi_n_plus), float(phi_n_minus)),
                float(sign),
            )
        else:
            p0, n0, rho_p, phi_p_plus, phi_p_minus, sigp, sign = vol
            pseries = gjr_recursion(
                res,
                (float(p0), float(rho_p), float(phi_p_plus), float(phi_p_minus)),
                float(sigp),
            )
            nseries = np.full_like(res, float(n0), dtype=float)
        return pseries, nseries, float(sigp), float(sign)

    def _negloglik(theta):
        pm = theta[:num_m]
        vol = _unpack_vol(theta)
        if not _constraints_ok(*vol):
            return float(big_penalty)

        res = residual_function(pm)
        pseries, nseries, sigp, sign = _shape_paths(theta, res)
        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return float(big_penalty)
        if np.any(pseries <= 0.0) or np.any(nseries <= 0.0):
            return float(big_penalty)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return float(big_penalty)

        ll = BEGE_log_density(
            res,
            pseries,
            nseries,
            sigp,
            sign,
            hyperu_method=density_hyperu_method,
        )
        val = -float(np.sum(ll))
        if not np.isfinite(val):
            return float(big_penalty)
        return val

    def _ind_negloglik_vec(theta):
        pm = theta[:num_m]
        vol = _unpack_vol(theta)
        if not _constraints_ok(*vol):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        res = residual_function(pm)
        pseries, nseries, sigp, sign = _shape_paths(theta, res)
        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)
        if np.any(pseries <= 0.0) or np.any(nseries <= 0.0):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        values = -BEGE_log_density(
            res,
            pseries,
            nseries,
            sigp,
            sign,
            hyperu_method=density_hyperu_method,
        )
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != N_obs:
            values = np.full(N_obs, float(values.ravel()[0]))
        if not np.all(np.isfinite(values)):
            values = np.full(N_obs, float(big_vec_penalty), dtype=float)
        return values

    def _sym(A): return 0.5 * (A + A.T)

    def _safe_inv_with_ridge(A, ridge0=1e-8, max_tries=6):
        A = _sym(A)
        I = np.eye(A.shape[0])
        ridge = float(ridge0)
        for _ in range(max_tries):
            try:
                return np.linalg.inv(A + ridge * I), ridge, False
            except np.linalg.LinAlgError:
                ridge *= 10.0
        return np.linalg.pinv(A), ridge, True

    def _project_to_bounds(x, bounds):
        out = np.array(x, float)
        tiny = 1e-10
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None:
                out[j] = max(out[j], lo + tiny)
            if hi is not None:
                out[j] = min(out[j], hi - tiny)
        return out

    def _central_diff_scores(theta, f_per_obs, bounds, rel=1e-4, absmin=1e-6):
        theta = np.asarray(theta, float)
        k = theta.size
        f0 = f_per_obs(theta)
        J = np.empty((N_obs, k), float)
        h = np.maximum(absmin, rel * np.maximum(1.0, np.abs(theta)))
        for j in range(k):
            th_p = theta.copy()
            th_m = theta.copy()
            th_p[j] += h[j]
            th_m[j] -= h[j]
            th_p = _project_to_bounds(th_p, bounds)
            th_m = _project_to_bounds(th_m, bounds)
            fp = np.asarray(f_per_obs(th_p), float).reshape(-1)
            fm = np.asarray(f_per_obs(th_m), float).reshape(-1)
            if fp.shape[0] != N_obs:
                fp = np.full(N_obs, float(fp.ravel()[0]))
            if fm.shape[0] != N_obs:
                fm = np.full(N_obs, float(fm.ravel()[0]))
            denom = float(th_p[j] - th_m[j])
            if denom == 0.0:
                th_p = theta.copy()
                th_p[j] += h[j]
                th_p = _project_to_bounds(th_p, bounds)
                fp = np.asarray(f_per_obs(th_p), float).reshape(-1)
                if fp.shape[0] != N_obs:
                    fp = np.full(N_obs, float(fp.ravel()[0]))
                fm = f0
                denom = float(th_p[j] - theta[j])
            J[:, j] = (fp - fm) / denom
        return J

    def _sample_mean():
        if num_m == 0:
            return np.array([], dtype=float)
        return np.array([rng.uniform(a, b) for (a, b) in mean_ranges], dtype=float)

    def _sample_dynamic_component():
        rho_upper = min(rho_hi, 1.0 - floor_eps - phi_lo)
        if rho_upper <= rho_lo:
            raise ValueError("rho_bounds and phi_bounds leave no feasible stable constant-shape BEGE starts.")

        for _ in range(500):
            rho = rng.uniform(rho_lo, rho_upper)
            cap = min(0.5 * phi_hi, 1.0 - rho - floor_eps)
            if cap <= 0.0:
                continue
            beta_plus = rng.uniform(0.0, cap)
            beta_minus = rng.uniform(0.0, cap)
            phi_plus = 2.0 * beta_plus
            phi_minus = 2.0 * beta_minus
            if rho + 0.5 * (phi_plus + phi_minus) < 1.0 - floor_eps:
                return rho, phi_plus, phi_minus

        rho = min(max(0.3, rho_lo), rho_upper)
        beta = 0.25 * max(1.0 - rho - floor_eps, floor_eps)
        phi = min(phi_hi, 2.0 * beta)
        phi = max(phi_lo, phi)
        return rho, phi, phi

    def _sample_vol():
        sigp = rng.uniform(sig_lo, sig_hi)
        sign = rng.uniform(sig_lo, sig_hi)
        shape_lo = max(p0_lo, floor_eps * 10.0)
        p0 = rng.uniform(shape_lo, p0_hi)
        n0 = rng.uniform(shape_lo, p0_hi)
        rho, phi_plus, phi_minus = _sample_dynamic_component()
        return np.array([p0, n0, rho, phi_plus, phi_minus, sigp, sign], dtype=float)

    start_values = [
        np.concatenate([_sample_mean(), _sample_vol()])
        for _ in range(int(n_starts))
    ]
    results = _run_multistart_minimize(
        start_values,
        _negloglik,
        method='SLSQP',
        bounds=bounds_full,
        constraints=constraints,
        options={'maxiter': int(maxiter), 'ftol': float(tol)},
        start_n_jobs=start_n_jobs,
    )

    best, best_fun = None, np.inf
    last_error = None
    for opt, exc in results:
        if exc is not None:
            last_error = exc
            continue
        if _optimizer_result_eligible(opt) and (best is None or opt.fun < best_fun):
            best_fun = opt.fun
            best = opt

    if best is None or (not np.isfinite(best_fun)) or best_fun >= big_penalty:
        detail = ""
        if last_error is not None:
            detail = f" Last error: {type(last_error).__name__}: {last_error}"
        raise RuntimeError(f"All starts failed to converge to a finite eligible objective value.{detail}")

    params = np.asarray(best.x, dtype=float)
    ll = -float(best.fun)
    AIC = 2 * len(params) - 2 * ll
    BIC = np.log(N_obs) * len(params) - 2 * ll

    se = np.full(len(params), np.nan, dtype=float)
    used_ridge = 0.0
    used_pseudo = False
    used_opg_fallback = False

    if compute_se:
        H = approx_hess(params, _negloglik, epsilon=1e-5)
        H = _sym(H)
        H_inv, used_ridge, used_pseudo = _safe_inv_with_ridge(H)
        scores = _central_diff_scores(params, _ind_negloglik_vec, bounds_full, rel=1e-4, absmin=1e-6)
        OPG = scores.T @ scores
        opg_scale = np.linalg.norm(OPG) / max(1, OPG.size)
        if (not np.isfinite(opg_scale)) or (opg_scale < 1e-8):
            cov = H_inv.copy()
            used_opg_fallback = True
        else:
            cov = H_inv @ _sym(OPG) @ H_inv
        cov = _sym(cov)
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        cov = (V * w) @ V.T
        se = np.sqrt(np.diag(cov))

    if compute_se and used_pseudo:
        print("[warn] Hessian singular; using pseudoinverse for covariance.")
    elif compute_se and used_ridge > 1e-8:
        print(f"[warn] Hessian near-singular; used ridge lambda={used_ridge:.1e}.")
    if compute_se and used_opg_fallback:
        print("[warn] OPG nearly zero/ill-conditioned; using observed-information inverse for covariance.")

    if print_summary:
        title = "BEGE (constant p, full GJR n)" if constant_shape == 'p' else "BEGE (full GJR p, constant n)"
        print("\n" + "-" * 72)
        print(title)
        print("-" * 72)
        if compute_se:
            print(f"{'Parameter':<18}{'Estimate':>14}{'Std. Err.':>14}{'t-Stat':>14}")
        else:
            print(f"{'Parameter':<18}{'Estimate':>14}")
        print("-" * 72)
        for nm, val, err in zip(names_full, params, se):
            if compute_se:
                t = np.nan if err <= 0 else (val / err)
                print(f"{nm:<18}{val:>14.6f}{err:>14.6f}{t:>14.3f}")
            else:
                print(f"{nm:<18}{val:>14.6f}")
        print("-" * 72)
        print(f"{'LogLik':<18}{ll:>14.6f}")
        print(f"{'AIC':<18}{AIC:>14.6f}")
        print(f"{'BIC':<18}{BIC:>14.6f}")
        print("-" * 72)

    return {
        'opt': best,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'names': names_full,
        'compute_se': bool(compute_se),
    }


def BEGE_ConstantPFullGJR_MLE(*args, **kwargs):
    """Full BEGE restriction with constant p_t and full GJR dynamics for n_t."""
    kwargs['constant_shape'] = 'p'
    return _BEGE_OneConstantFullGJR_MLE(*args, **kwargs)


def BEGE_ConstantNFullGJR_MLE(*args, **kwargs):
    """Full BEGE restriction with constant n_t and full GJR dynamics for p_t."""
    kwargs['constant_shape'] = 'n'
    return _BEGE_OneConstantFullGJR_MLE(*args, **kwargs)


def BG_GARCH(
    Y, X=None, mean_type='ARX(1,1)',
    n_starts=50, maxiter=800, tol=1e-8, random_state=None,
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.0, 10.0),
    rho_bounds=(0.0, 1.0),
    phi_bounds=(0.0, 2.0),
    floor_eps=1e-6,
    print_summary=True,
    big_penalty=1e12,
    big_vec_penalty=1e6,
    compute_se=False,
    density_hyperu_method='scipy_approx',
    enforce_variance_bounds=True,
    start_n_jobs=None,
):
    """
    BG_GARCH: Good/Bad volatility with symmetric-in-sign GARCH:
      phi_p+ = phi_p- = phi_p
      phi_n+ = phi_n- = phi_n

    Parameter order:
      [mean params] +
      [p0, n0, rho_p, rho_n, phi_p, phi_n, sigma_p, sigma_n]
    """
    import numpy as np
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_hess

    rng = np.random.default_rng(random_state)
    Y, X = _align_mean_sample(Y, X)
    N_obs = int(Y.shape[0])

    # ---------- mean spec ----------
    def _get_mean_spec(Y, mean_type):
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        if mean_type == 'constant':
            mean_model, num_m = mean_const, 0
            bounds_mean, names_mean, ranges = [], [], []
        elif mean_type == 'ARX(1,1)':
            mean_model, num_m = mean_ARX11, 3
            bounds_mean = [(ymin, ymax), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'SPF']
            ranges = [
                (0.0824 - 2*0.086, 0.0824 + 2*0.086),
                (0.3005 - 2*0.112, 0.3005 + 2*0.112),
                (0.7337 - 2*0.167, 0.7337 + 2*0.167),
            ]
        elif mean_type == 'ARX(2,1)':
            mean_model, num_m = mean_ARX21, 4
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
            ranges = [
                (0.0897 - 2*0.086, 0.0897 + 2*0.086),
                (0.2892 - 2*0.098, 0.2892 + 2*0.098),
                (0.0834 - 2*0.126, 0.0834 + 2*0.126),
                (0.6385 - 2*0.251, 0.6385 + 2*0.251),
            ]
        elif mean_type == 'ARX(2,2)':
            mean_model, num_m = mean_ARX22, 5
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']
            ranges = [
                (0.0856 - 2*0.086, 0.0856 + 2*0.086),
                (0.2992 - 2*0.106, 0.2992 + 2*0.106),
                (0.0914 - 2*0.125, 0.0914 + 2*0.125),
                (0.4084 - 2*0.433, 0.4084 + 2*0.433),
                (0.2136 - 2*0.297, 0.2136 + 2*0.297),
            ]
        else:
            raise ValueError("Invalid mean_type")
        return mean_model, num_m, bounds_mean, names_mean, ranges

    mean_model, num_m, bounds_mean, names_mean, mean_ranges = _get_mean_spec(Y, mean_type)
    residual_function = _make_residual_function(Y, X, mean_type)

    # ---------- bounds & names ----------
    (sig_lo, sig_hi) = sigma_bounds
    (p0_lo,  p0_hi)  = p0n0_bounds
    (rho_lo, rho_hi) = rho_bounds
    (phi_lo, phi_hi) = phi_bounds

    bounds_vol = [
        (p0_lo, p0_hi),     # p0
        (p0_lo, p0_hi),     # n0
        (rho_lo, rho_hi),   # rho_p
        (rho_lo, rho_hi),   # rho_n
        (phi_lo, phi_hi),   # phi_p  (common + and -)
        (phi_lo, phi_hi),   # phi_n  (common + and -)
        (sig_lo, sig_hi),   # sigma_p
        (sig_lo, sig_hi),   # sigma_n
    ]
    names_vol = ['p0', 'n0', 'rho_p', 'rho_n', 'phi_p', 'phi_n', 'σ₊', 'σ₋']

    bounds_full = bounds_mean + bounds_vol
    names_full  = names_mean + names_vol

    # ---------- constraints ----------
    def _unpack_vol(theta):
        p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign = theta[num_m:]
        return p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign

    def _constraints_ok(p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign):
        vals = np.asarray([p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign], dtype=float)
        if not np.all(np.isfinite(vals)):
            return False
        if rho_p + phi_p >= 1.0 - floor_eps:
            return False
        if rho_n + phi_n >= 1.0 - floor_eps:
            return False
        return True

    # rho_p + phi_p < 1
    def constr_rho_phi_p(theta):
        p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign = _unpack_vol(theta)
        return 1.0 - floor_eps - (rho_p + phi_p)

    # rho_n + phi_n < 1
    def constr_rho_phi_n(theta):
        p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign = _unpack_vol(theta)
        return 1.0 - floor_eps - (rho_n + phi_n)

    constraints = [
        {'type': 'ineq', 'fun': constr_rho_phi_p},
        {'type': 'ineq', 'fun': constr_rho_phi_n},
    ]

    # ---------- objective ----------
    def _negloglik(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign = theta[num_m:]
        if not _constraints_ok(p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign):
            return float(big_penalty)

        res = residual_function(pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p), float(phi_p)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), float(phi_n), float(phi_n)), float(sign))

        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return float(big_penalty)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return float(big_penalty)

        ll = BEGE_log_density(
            res, pseries, nseries, float(sigp), float(sign),
            hyperu_method=density_hyperu_method,
        )
        val = -float(np.sum(ll))
        if not np.isfinite(val):
            return float(big_penalty)
        return val

    def _ind_negloglik_vec(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign = theta[num_m:]
        if not _constraints_ok(p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        res = residual_function(pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p), float(phi_p)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), float(phi_n), float(phi_n)), float(sign))

        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)
        if enforce_variance_bounds and not bege_variance_bounds_ok(res, pseries, nseries, sigp, sign):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        v = -BEGE_log_density(
            res, pseries, nseries, float(sigp), float(sign),
            hyperu_method=density_hyperu_method,
        )
        v = np.asarray(v, float).reshape(-1)
        if v.shape[0] != N_obs:
            v = np.full(N_obs, float(v.ravel()[0]))
        if not np.all(np.isfinite(v)):
            v = np.full(N_obs, float(big_vec_penalty))
        return v

    # ---------- helpers ----------
    def _sym(A): return 0.5 * (A + A.T)

    def _safe_inv_with_ridge(A, ridge0=1e-8, max_tries=6):
        A = _sym(A)
        I = np.eye(A.shape[0])
        ridge = float(ridge0)
        for _ in range(max_tries):
            try:
                return np.linalg.inv(A + ridge * I), ridge, False
            except np.linalg.LinAlgError:
                ridge *= 10.0
        return np.linalg.pinv(A), ridge, True

    def _project_to_bounds(x, bounds):
        out = np.array(x, float); tiny = 1e-10
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None: out[j] = max(out[j], lo + tiny)
            if hi is not None: out[j] = min(out[j], hi - tiny)
        return out

    def _central_diff_scores(theta, f_per_obs, bounds, rel=1e-4, absmin=1e-6):
        theta = np.asarray(theta, float)
        k = theta.size
        f0 = f_per_obs(theta)                   # (N_obs,)
        J  = np.empty((N_obs, k), float)
        h  = np.maximum(absmin, rel * np.maximum(1.0, np.abs(theta)))

        for j in range(k):
            th_p = theta.copy(); th_p[j] += h[j]; th_p = _project_to_bounds(th_p, bounds)
            th_m = theta.copy(); th_m[j] -= h[j]; th_m = _project_to_bounds(th_m, bounds)

            fp = np.asarray(f_per_obs(th_p), float).reshape(-1)
            fm = np.asarray(f_per_obs(th_m), float).reshape(-1)
            if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
            if fm.shape[0] != N_obs: fm = np.full(N_obs, float(fm.ravel()[0]))

            denom = float(th_p[j] - th_m[j])
            if denom == 0.0:
                th_p = theta.copy(); th_p[j] += h[j]; th_p = _project_to_bounds(th_p, bounds)
                fp   = np.asarray(f_per_obs(th_p), float).reshape(-1)
                if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
                fm   = f0
                denom = float(th_p[j] - theta[j])

            J[:, j] = (fp - fm) / denom

        return J

    # ---------- initializers ----------
    def _sample_mean():
        if num_m == 0:
            return np.array([], float)
        return np.array([rng.uniform(a, b) for (a, b) in mean_ranges], dtype=float)

    def _sample_vol():
        p0_lo, p0_hi = p0n0_bounds
        rho_lo, rho_hi = rho_bounds
        phi_lo, phi_hi = phi_bounds
        sig_lo, sig_hi = sigma_bounds

        rho_upper = min(rho_hi, 1.0 - floor_eps - phi_lo)
        if rho_upper <= rho_lo:
            raise ValueError("rho_bounds and phi_bounds leave no feasible stable BG-GARCH starts.")

        for _ in range(500):
            sigp = rng.uniform(sig_lo, sig_hi)
            sign = rng.uniform(sig_lo, sig_hi)
            p0 = rng.uniform(p0_lo, p0_hi)
            n0 = rng.uniform(p0_lo, p0_hi)

            rho_p = rng.uniform(rho_lo, rho_hi)
            rho_n = rng.uniform(rho_lo, rho_hi)
            max_phi_p = min(phi_hi, 1.0 - floor_eps - rho_p)
            max_phi_n = min(phi_hi, 1.0 - floor_eps - rho_n)
            if max_phi_p <= phi_lo or max_phi_n <= phi_lo:
                continue

            phi_p = rng.uniform(phi_lo, max_phi_p)
            phi_n = rng.uniform(phi_lo, max_phi_n)

            vals = np.array([p0, n0, rho_p, rho_n, phi_p, phi_n, sigp, sign], dtype=float)
            if _constraints_ok(*vals):
                return vals

        # fallback
        sig_default = min(max(0.5, sig_lo), sig_hi)
        p0_default = min(max(0.5, p0_lo), p0_hi)
        n0_default = min(max(0.5, p0_lo), p0_hi)
        rho_default = min(max(0.3, rho_lo), rho_upper)
        phi_default = min(0.4, phi_hi, 1.0 - floor_eps - rho_default)
        phi_default = max(phi_lo, phi_default)
        return np.array(
            [p0_default, n0_default, rho_default, rho_default, phi_default, phi_default, sig_default, sig_default],
            dtype=float,
        )

    # ---------- optimize ----------
    start_values = [
        np.concatenate([_sample_mean(), _sample_vol()])
        for _ in range(int(n_starts))
    ]
    results = _run_multistart_minimize(
        start_values,
        _negloglik,
        method='SLSQP',
        bounds=bounds_full,
        constraints=constraints,
        options={'maxiter': int(maxiter), 'ftol': float(tol)},
        start_n_jobs=start_n_jobs,
    )

    best, best_fun = None, np.inf
    last_error = None
    for opt, exc in results:
        if exc is not None:
            last_error = exc
            continue
        if _optimizer_result_eligible(opt) and opt.fun < best_fun:
            best_fun = opt.fun
            best = opt

    if best is None or (not np.isfinite(best_fun)) or best_fun >= big_penalty:
        detail = ""
        if last_error is not None:
            detail = f" Last error: {type(last_error).__name__}: {last_error}"
        raise RuntimeError(f"All starts failed to converge to a finite eligible objective value.{detail}")

    params = best.x
    ll     = -best.fun
    AIC    = 2*len(params) - 2*ll
    BIC    = np.log(N_obs)*len(params) - 2*ll

    se = np.full(len(params), np.nan, dtype=float)
    used_ridge = 0.0
    used_pseudo = False
    used_opg_fallback = False

    if compute_se:
        # ---------- SEs ----------
        H = approx_hess(params, _negloglik, epsilon=1e-5)
        H = _sym(H)
        H_inv, used_ridge, used_pseudo = _safe_inv_with_ridge(H)

        scores = _central_diff_scores(params, _ind_negloglik_vec, bounds_full, rel=1e-4, absmin=1e-6)
        OPG    = scores.T @ scores

        opg_scale = np.linalg.norm(OPG) / max(1, OPG.size)
        if (not np.isfinite(opg_scale)) or (opg_scale < 1e-8):
            cov = H_inv.copy()
            used_opg_fallback = True
        else:
            cov = H_inv @ _sym(OPG) @ H_inv
            used_opg_fallback = False

        cov = _sym(cov)
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        cov = (V * w) @ V.T
        se  = np.sqrt(np.diag(cov))

    # ---------- summary ----------
    if compute_se and used_pseudo:
        print("[warn] Hessian singular; using pseudoinverse for covariance.")
    elif compute_se and used_ridge > 1e-8:
        print(f"[warn] Hessian near-singular; used ridge λ={used_ridge:.1e}.")
    if compute_se and used_opg_fallback:
        print("[warn] OPG nearly zero/ill-conditioned; using observed-information (H^{-1}) for covariance.")

    if print_summary:
        print("\n" + "-"*72)
        print("BG_GARCH (sym-in-sign good/bad volatility)")
        print("-"*72)
        if compute_se:
            print(f"{'Parameter':<18}{'Estimate':>14}{'Std. Err.':>14}{'t-Stat':>14}")
        else:
            print(f"{'Parameter':<18}{'Estimate':>14}")
        print("-"*72)
        if compute_se:
            for nm, val, err in zip(names_full, params, se):
                t = np.nan if err <= 0 else (val/err)
                print(f"{nm:<18}{val:>14.6f}{err:>14.6f}{t:>14.3f}")
        else:
            for nm, val in zip(names_full, params):
                print(f"{nm:<18}{val:>14.6f}")
        print("-"*72)
        print(f"{'LogLik':<18}{ll:>14.6f}")
        print(f"{'AIC':<18}{AIC:>14.6f}")
        print(f"{'BIC':<18}{BIC:>14.6f}")
        print("-"*72)

    return {
        'opt': best,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'names': names_full,
        'compute_se': bool(compute_se),
    }





def _ID_GARCH_legacy(
    Y, X=None, mean_type='ARX(1,1)',
    n_starts=50, maxiter=800, tol=1e-8, random_state=None,
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.005, 10.0),
    rho_bounds=(1e-5, 0.999),
    phi_bounds=(1e-5, 1.5),
    floor_eps=1e-6,
    print_summary=True,
    big_penalty=1e12,
    big_vec_penalty=1e6
):
    """
    ID_GARCH: Inflation/Deflation BEGE-GJR:
      phi_p_minus = 0, phi_n_plus = 0.
      Positive residuals affect only 'good' volatility (phi_p_plus),
      negative residuals affect only 'bad' volatility (phi_n_minus).

    Parameter order:
      [mean params] +
      [p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigma_p, sigma_n]
    """
    import numpy as np
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_hess

    rng = np.random.default_rng(random_state)
    Y = np.asarray(Y, dtype=float)
    N_obs = int(Y.shape[0])

    if X is not None:
        n = min(len(Y), len(X))
        Y = Y[:n]
        X = np.asarray(X, dtype=float)[:n]
        N_obs = int(n)

    # ---------- mean spec ----------
    def _get_mean_spec(Y, mean_type):
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        if mean_type == 'constant':
            mean_model, num_m = mean_const, 0
            bounds_mean, names_mean, ranges = [], [], []
        elif mean_type == 'ARX(1,1)':
            mean_model, num_m = mean_ARX11, 3
            bounds_mean = [(ymin, ymax), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'SPF']
            ranges = [
                (0.0720 - 2*0.086, 0.0720 + 2*0.086),
                (0.2881 - 2*0.073, 0.2881 + 2*0.073),
                (0.7508 - 2*0.125, 0.7508 + 2*0.125),
            ]
        elif mean_type == 'ARX(2,1)':
            mean_model, num_m = mean_ARX21, 4
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF']
            ranges = [
                (0.0792 - 2*0.087, 0.0792 + 2*0.087),
                (0.2793 - 2*0.073, 0.2793 + 2*0.073),
                (0.0728 - 2*0.080, 0.0728 + 2*0.080),
                (0.6661 - 2*0.156, 0.6661 + 2*0.156),
            ]
        elif mean_type == 'ARX(2,2)':
            mean_model, num_m = mean_ARX22, 5
            bounds_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)]
            names_mean = ['const', 'Infl(1)', 'Infl(2)', 'SPF', 'SPF.lag(1)']
            ranges = [
                (0.0761 - 2*0.087, 0.0761 + 2*0.087),
                (0.2880 - 2*0.076, 0.2880 + 2*0.076),
                (0.0799 - 2*0.082, 0.0799 + 2*0.082),
                (0.4720 - 2*0.459, 0.4720 + 2*0.459),
                (0.1793 - 2*0.399, 0.1793 + 2*0.399),
            ]
        else:
            raise ValueError("Invalid mean_type")
        return mean_model, num_m, bounds_mean, names_mean, ranges

    mean_model, num_m, bounds_mean, names_mean, mean_ranges = _get_mean_spec(Y, mean_type)

    # ---------- bounds & names ----------
    (sig_lo, sig_hi) = sigma_bounds
    (p0_lo,  p0_hi)  = p0n0_bounds
    (rho_lo, rho_hi) = rho_bounds
    (phi_lo, phi_hi) = phi_bounds

    bounds_vol = [
        (p0_lo, p0_hi),     # p0
        (p0_lo, p0_hi),     # n0
        (rho_lo, rho_hi),   # rho_p
        (rho_lo, rho_hi),   # rho_n
        (phi_lo, phi_hi),   # phi_p_plus
        (phi_lo, phi_hi),   # phi_n_minus
        (sig_lo, sig_hi),   # sigma_p
        (sig_lo, sig_hi),   # sigma_n
    ]
    names_vol = ['p0', 'n0', 'rho_p', 'rho_n', 'phi_p+', 'phi_n-', 'σ₊', 'σ₋']

    bounds_full = bounds_mean + bounds_vol
    names_full  = names_mean + names_vol

    # ---------- constraints ----------
    def _unpack_vol(theta):
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign = theta[num_m:]
        return p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign

    # rho_p + phi_p_plus/2 <= 1
    def constr_rho_phi_p(theta):
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign = _unpack_vol(theta)
        return 1.0 - (rho_p + 0.5 * phi_p_plus)

    # rho_n + phi_n_minus/2 <= 1
    def constr_rho_phi_n(theta):
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign = _unpack_vol(theta)
        return 1.0 - (rho_n + 0.5 * phi_n_minus)

    constraints = [
        {'type': 'ineq', 'fun': constr_rho_phi_p},
        {'type': 'ineq', 'fun': constr_rho_phi_n},
    ]

    # ---------- objective ----------
    def _negloglik(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign = theta[num_m:]

        res = mean_model(Y, X, pm)
        # phi_p_minus = 0, phi_n_plus = 0 by construction:
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), 0.0), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), 0.0, float(phi_n_minus)), float(sign))

        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return float(big_penalty)

        ll = BEGE_log_density(res, pseries, nseries, float(sigp), float(sign))
        val = -float(np.sum(ll))
        if not np.isfinite(val):
            return float(big_penalty)
        return val

    def _ind_negloglik_vec(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign = theta[num_m:]

        res = mean_model(Y, X, pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), 0.0), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), 0.0, float(phi_n_minus)), float(sign))

        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))):
            return np.full(N_obs, float(big_vec_penalty), dtype=float)

        v = -BEGE_log_density(res, pseries, nseries, float(sigp), float(sign))
        v = np.asarray(v, float).reshape(-1)
        if v.shape[0] != N_obs:
            v = np.full(N_obs, float(v.ravel()[0]))
        if not np.all(np.isfinite(v)):
            v = np.full(N_obs, float(big_vec_penalty))
        return v

    # ---------- helpers ----------
    def _sym(A): return 0.5 * (A + A.T)

    def _safe_inv_with_ridge(A, ridge0=1e-8, max_tries=6):
        A = _sym(A)
        I = np.eye(A.shape[0])
        ridge = float(ridge0)
        for _ in range(max_tries):
            try:
                return np.linalg.inv(A + ridge * I), ridge, False
            except np.linalg.LinAlgError:
                ridge *= 10.0
        return np.linalg.pinv(A), ridge, True

    def _project_to_bounds(x, bounds):
        out = np.array(x, float); tiny = 1e-10
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None: out[j] = max(out[j], lo + tiny)
            if hi is not None: out[j] = min(out[j], hi - tiny)
        return out

    def _central_diff_scores(theta, f_per_obs, bounds, rel=1e-4, absmin=1e-6):
        theta = np.asarray(theta, float)
        k = theta.size
        f0 = f_per_obs(theta)
        J  = np.empty((N_obs, k), float)
        h  = np.maximum(absmin, rel * np.maximum(1.0, np.abs(theta)))

        for j in range(k):
            th_p = theta.copy(); th_p[j] += h[j]; th_p = _project_to_bounds(th_p, bounds)
            th_m = theta.copy(); th_m[j] -= h[j]; th_m = _project_to_bounds(th_m, bounds)

            fp = np.asarray(f_per_obs(th_p), float).reshape(-1)
            fm = np.asarray(f_per_obs(th_m), float).reshape(-1)
            if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
            if fm.shape[0] != N_obs: fm = np.full(N_obs, float(fm.ravel()[0]))

            denom = float(th_p[j] - th_m[j])
            if denom == 0.0:
                th_p = theta.copy(); th_p[j] += h[j]; th_p = _project_to_bounds(th_p, bounds)
                fp   = np.asarray(f_per_obs(th_p), float).reshape(-1)
                if fp.shape[0] != N_obs: fp = np.full(N_obs, float(fp.ravel()[0]))
                fm   = f0
                denom = float(th_p[j] - theta[j])

            J[:, j] = (fp - fm) / denom

        return J

    # ---------- initializers ----------
    def _sample_mean():
        if num_m == 0:
            return np.array([], float)
        return np.array([rng.uniform(a, b) for (a, b) in mean_ranges], dtype=float)

    def _sample_vol():
        p0_lo, p0_hi = p0n0_bounds
        rho_lo, rho_hi = rho_bounds
        phi_lo, phi_hi = phi_bounds
        sig_lo, sig_hi = sigma_bounds

        for _ in range(200):
            rho_p = rng.uniform(rho_lo, rho_hi)
            rho_n = rng.uniform(rho_lo, rho_hi)

            max_phi_p = max(phi_lo, 2.0*(1.0 - rho_p - floor_eps))
            max_phi_n = max(phi_lo, 2.0*(1.0 - rho_n - floor_eps))
            max_phi_p = min(max_phi_p, phi_hi)
            max_phi_n = min(max_phi_n, phi_hi)

            phi_p_plus  = rng.uniform(phi_lo, max_phi_p)
            phi_n_minus = rng.uniform(phi_lo, max_phi_n)
            sigp  = rng.uniform(sig_lo, sig_hi)
            sign  = rng.uniform(sig_lo, sig_hi)
            p0    = rng.uniform(p0_lo, p0_hi)
            n0    = rng.uniform(p0_lo, p0_hi)

            return np.array([p0, n0, rho_p, rho_n, phi_p_plus, phi_n_minus, sigp, sign], dtype=float)

        return np.array([1.0, 1.0, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5], dtype=float)

    # ---------- optimize ----------
    best, best_fun = None, np.inf
    for _ in range(int(n_starts)):
        init = np.concatenate([_sample_mean(), _sample_vol()])
        try:
            opt = minimize(
                _negloglik,
                init,
                method='SLSQP',
                bounds=bounds_full,
                constraints=constraints,
                options={'maxiter': int(maxiter), 'ftol': float(tol)}
            )
            if _optimizer_result_eligible(opt) and opt.fun < best_fun:
                best_fun = opt.fun
                best = opt
        except Exception:
            continue

    if best is None:
        raise RuntimeError("All starts failed to converge to a finite eligible objective value.")

    params = best.x
    ll     = -best.fun
    AIC    = 2*len(params) - 2*ll
    BIC    = np.log(N_obs)*len(params) - 2*ll

    # ---------- SEs ----------
    H = approx_hess(params, _negloglik, epsilon=1e-5)
    H = _sym(H)
    H_inv, used_ridge, used_pseudo = _safe_inv_with_ridge(H)

    scores = _central_diff_scores(params, _ind_negloglik_vec, bounds_full, rel=1e-4, absmin=1e-6)
    OPG    = scores.T @ scores

    opg_scale = np.linalg.norm(OPG) / max(1, OPG.size)
    if (not np.isfinite(opg_scale)) or (opg_scale < 1e-8):
        cov = H_inv.copy()
        used_opg_fallback = True
    else:
        cov = H_inv @ _sym(OPG) @ H_inv
        used_opg_fallback = False

    cov = _sym(cov)
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 0.0)
    cov = (V * w) @ V.T
    se  = np.sqrt(np.diag(cov))

    # ---------- summary ----------
    if used_pseudo:
        print("[warn] Hessian singular; using pseudoinverse for covariance.")
    elif used_ridge > 1e-8:
        print(f"[warn] Hessian near-singular; used ridge λ={used_ridge:.1e}.")
    if used_opg_fallback:
        print("[warn] OPG nearly zero/ill-conditioned; using observed-information (H^{-1}) for covariance.")

    if print_summary:
        print("\n" + "-"*72)
        print("ID_GARCH (Inflation/Deflation asymmetry)")
        print("-"*72)
        print(f"{'Parameter':<18}{'Estimate':>14}{'Std. Err.':>14}{'t-Stat':>14}")
        print("-"*72)
        for nm, val, err in zip(names_full, params, se):
            t = np.nan if err <= 0 else (val/err)
            print(f"{nm:<18}{val:>14.6f}{err:>14.6f}{t:>14.3f}")
        print("-"*72)
        print(f"{'LogLik':<18}{ll:>14.6f}")
        print(f"{'AIC':<18}{AIC:>14.6f}")
        print(f"{'BIC':<18}{BIC:>14.6f}")
        print("-"*72)

    return {
        'opt': best,
        'params': params,
        'se': se,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'names': names_full
    }
