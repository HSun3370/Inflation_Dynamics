
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from BEGE_GARCH import *
from BEGE_density import *
import numpy as np
from numpy.random import default_rng
from scipy.stats import gamma as _gamma
from scipy.optimize import minimize
 
from statsmodels.tools.numdiff import approx_hess

import os
  
parser = argparse.ArgumentParser(description="seed sets")

parser.add_argument("--id", type=int, default=1) 
args = parser.parse_args()  # <<< after all arguments are added
seed = args.id  

 
# =========================================================
# 0) TRUE PARAMS (With-Justin)
# =========================================================
# True parameters (example close to your shared set, but fully separate)
vol_true_full = dict(
    p0=1.10,  n0=0.55,
    rho_p=0.40,  rho_n=0.35,
    phi_p_plus=0.26,  phi_p_minus=0.10,
    phi_n_plus=0.22,  phi_n_minus=0.08,
    sigp=0.22,  sign=0.95
)


def simulate_bege_full(
    T,
    mean_type='constant',
    mean_params_true=None,   # for 'constant': {}, for 'ARX(1,1)': {'const':..., 'phi1':..., 'theta1':...}
    vol_params_true=None,    # dict with keys: p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign
    rng_seed=123,
    max_shape_cap=5_000.0
):
    """
    Simulate BEGE with FULL GJR recursions (separate p_t and n_t parameters).

    u_t = sigp*(Gamma(p_t)-p_t) - sign*(Gamma(n_t)-n_t)
    p_t = p0 + rho_p*p_{t-1} + (phi_p_plus/(2*sigp^2))*(u_{t-1}^+)^2 + (phi_p_minus/(2*sigp^2))*(u_{t-1}^-)^2
    n_t = n0 + rho_n*n_{t-1} + (phi_n_plus/(2*sign^2))*(u_{t-1}^+)^2 + (phi_n_minus/(2*sign^2))*(u_{t-1}^-)^2

    Returns:
      Y, X (if needed; else None), dict with pseries, nseries, u, and components
    """
    assert vol_params_true is not None, "vol_params_true must be provided"

    p0   = float(vol_params_true['p0'])
    n0   = float(vol_params_true['n0'])
    rho_p = float(vol_params_true['rho_p'])
    rho_n = float(vol_params_true['rho_n'])
    phi_p_plus = float(vol_params_true['phi_p_plus'])
    phi_p_minus= float(vol_params_true['phi_p_minus'])
    phi_n_plus = float(vol_params_true['phi_n_plus'])
    phi_n_minus= float(vol_params_true['phi_n_minus'])
    sigp = float(vol_params_true['sigp'])
    sign = float(vol_params_true['sign'])

    rng = default_rng(rng_seed)

    # Mean process
    if mean_type == 'constant':
        const = mean_params_true.get('const', 0.0) if mean_params_true else 0.0
        # no X
        def mean_gen(u):
            return const + u
        X = None
    elif mean_type == 'ARX(1,1)':
        # Expect mean_params_true: {'const': c0, 'phi1': φ, 'theta1': θ} and an exogenous series X_t
        assert mean_params_true is not None and 'const' in mean_params_true and 'phi1' in mean_params_true and 'theta1' in mean_params_true and 'X' in mean_params_true
        c0   = float(mean_params_true['const'])
        phi1 = float(mean_params_true['phi1'])
        theta1 = float(mean_params_true['theta1'])
        X    = np.asarray(mean_params_true['X'], float)
        T    = min(T, len(X))
        def mean_gen(u):
            Y = np.empty(T, float)
            # simple ARX(1,1): Y_t = c0 + phi1*Y_{t-1} + theta1*X_{t-1} + u_t
            Y[0] = c0 + u[0]
            for t in range(1, T):
                Y[t] = c0 + phi1*Y[t-1] + theta1*X[t-1] + u[t]
            return Y
    else:
        raise ValueError("mean_type must be 'constant' or 'ARX(1,1)' for this simulator")

    # Allocate
    u = np.zeros(T, dtype=float)
    pseries = np.zeros(T, dtype=float)
    nseries = np.zeros(T, dtype=float)

    # GJR backcasts
    denom_p = max(1e-8, 1.0 - rho_p - 0.5*(phi_p_plus + phi_p_minus))
    denom_n = max(1e-8, 1.0 - rho_n - 0.5*(phi_n_plus + phi_n_minus))
    pseries[0] = max(p0/denom_p, 1e-4)
    nseries[0] = max(n0/denom_n, 1e-4)

    # Initial shock via gamma difference
    gp0 = _gamma.rvs(a=pseries[0], scale=1.0, random_state=rng)
    gn0 = _gamma.rvs(a=nseries[0], scale=1.0, random_state=rng)
    u[0] = sigp*(gp0 - pseries[0]) - sign*(gn0 - nseries[0])

    inv2p = 1.0/(2.0*sigp*sigp)
    inv2n = 1.0/(2.0*sign*sign)

    for t in range(1, T):
        up = max(u[t-1], 0.0)
        un = min(u[t-1], 0.0)

        p_t = p0 + rho_p*pseries[t-1] + phi_p_plus*inv2p*(up**2) + phi_p_minus*inv2p*(un**2)
        n_t = n0 + rho_n*nseries[t-1] + phi_n_plus*inv2n*(up**2) + phi_n_minus*inv2n*(un**2)

        if max_shape_cap is not None:
            p_t = float(np.clip(p_t, 1e-4, max_shape_cap))
            n_t = float(np.clip(n_t, 1e-4, max_shape_cap))
        else:
            p_t = max(p_t, 1e-4)
            n_t = max(n_t, 1e-4)

        pseries[t] = p_t
        nseries[t] = n_t

        gp = _gamma.rvs(a=p_t, scale=1.0, random_state=rng)
        gn = _gamma.rvs(a=n_t, scale=1.0, random_state=rng)
        u[t] = sigp*(gp - p_t) - sign*(gn - n_t)

    Y = mean_gen(u)  # add mean structure
    extras = dict(pseries=pseries, nseries=nseries, u=u)
    return Y, (X if mean_type!='constant' else None), extras

# --- tiny helpers you already effectively have in your code base ---
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
    x = np.array(x, float); tiny = 1e-12
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None: x[j] = max(x[j], lo + tiny)
        if hi is not None: x[j] = min(x[j], hi - tiny)
    return x

def _central_diff_scores(theta, f_per_obs, bounds, rel=1e-4, absmin=1e-6):
    theta = np.asarray(theta, float)
    f0 = f_per_obs(theta)                 # (N,)
    N = f0.size
    k = theta.size
    J = np.empty((N, k), float)
    h = np.maximum(absmin, rel*np.maximum(1.0, np.abs(theta)))
    for j in range(k):
        thp = theta.copy(); thp[j] += h[j]; thp = _project_to_bounds(thp, bounds)
        thm = theta.copy(); thm[j] -= h[j]; thm = _project_to_bounds(thm, bounds)
        fp = np.asarray(f_per_obs(thp), float).reshape(-1); 
        fm = np.asarray(f_per_obs(thm), float).reshape(-1)
        if fp.size != N: fp = np.full(N, float(fp.ravel()[0]))
        if fm.size != N: fm = np.full(N, float(fm.ravel()[0]))
        denom = float(thp[j] - thm[j])
        if denom == 0.0:
            thp = theta.copy(); thp[j] += h[j]; thp = _project_to_bounds(thp, bounds)
            fp = np.asarray(f_per_obs(thp), float).reshape(-1)
            if fp.size != N: fp = np.full(N, float(fp.ravel()[0]))
            fm = f0
            denom = float(thp[j] - theta[j])
        J[:, j] = (fp - fm)/denom
    return J


# =========================
# NEAR-START FULL-GJR MLE
# =========================
def BEGE_FullGJR_MLE_nearstarts(
    Y, X=None, mean_type='constant',
    # --- center for draws (ordered exactly like BEGE_FullGJR_MLE 'names') ---
    center_params=None,           # list/array: [mean..., p0,n0,rho_p,rho_n,phi_p+,phi_p-,phi_n+,phi_n-,sigp,sign]
    # --- narrow perturbations around center ---
    jitter_rel=0.05,              # 5% around center (applied to |center|); see jitter_abs for near-zero centers
    jitter_abs=1e-3,              # absolute epsilon around small centers
    n_starts=100,                  # how many near-starts to try (one will be the exact center)
    maxiter=800, tol=1e-8, random_state=None,
    # --- wide bounds (same as before) ---
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.005, 10.0),
    rho_bounds=(1e-5, 0.999),
    phi_bounds=(1e-5, 1.5),
    # --- hard overflow guard (no stability checks; just cap) ---
    cap_pn=120.0,
    big_penalty=1e12,
    big_vec_penalty=1e6,
    print_summary=True
):
    """
    Full BEGE (separate p_t and n_t) MLE with multi-starts drawn narrowly around a provided center.
    Bounds remain wide; only the initial guesses are 'near' center.
    """
    import numpy as np

    rng = default_rng(random_state)
    Y = np.asarray(Y, float)
    N_obs = Y.size
    if X is not None:
        n = min(len(Y), len(X)); Y = Y[:n]; X = np.asarray(X, float)[:n]; N_obs = n

    # --- mean spec (use your existing mean models; only wiring) ---
    def _get_mean_spec(Y, mean_type):
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        if mean_type == 'constant':
            mean_model, num_m = mean_const, 0
            bounds_mean, names_mean = [], []
        elif mean_type == 'ARX(1,1)':
            mean_model, num_m = mean_ARX11, 3
            bounds_mean, names_mean = [(ymin, ymax), (-0.999, 0.999), (-10, 10)], ['const', 'Infl(1)', 'SPF']
        elif mean_type == 'ARX(2,1)':
            mean_model, num_m = mean_ARX21, 4
            bounds_mean, names_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10)], ['const','Infl(1)','Infl(2)','SPF']
        elif mean_type == 'ARX(2,2)':
            mean_model, num_m = mean_ARX22, 5
            bounds_mean, names_mean = [(ymin, ymax), (-1.999, 1.999), (-0.999, 0.999), (-10, 10), (-10, 10)], ['const','Infl(1)','Infl(2)','SPF','SPF.lag(1)']
        else:
            raise ValueError("Invalid mean_type")
        return mean_model, num_m, bounds_mean, names_mean

    mean_model, num_m, bounds_mean, names_mean = _get_mean_spec(Y, mean_type)

    # --- bounds & names (Full GJR) ---
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
    names_vol  = ['p0','n0','rho_p','rho_n','phi_p⁺','phi_p⁻','phi_n⁺','phi_n⁻','σ₊','σ₋']
    bounds_full = bounds_mean + bounds_vol
    names_full  = names_mean + names_vol

    k = len(bounds_full)

    # --- objective (Justin's BEGE density; no stability check, just cap) ---
    def _negloglik(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign = theta[num_m:]
        res = mean_model(Y, X, pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), float(phi_p_minus)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), float(phi_n_plus), float(phi_n_minus)), float(sign))
        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))) \
           or (np.any(pseries > cap_pn)) or (np.any(nseries > cap_pn)):
            return float(big_penalty)
        ll = BEGE_log_density(res, pseries, nseries, float(sigp), float(sign))
        val = -float(np.sum(ll))
        if not np.isfinite(val): return float(big_penalty)
        return val

    def _ind_negloglik_vec(theta):
        pm = theta[:num_m]
        p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign = theta[num_m:]
        res = mean_model(Y, X, pm)
        pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), float(phi_p_minus)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho_n), float(phi_n_plus), float(phi_n_minus)), float(sign))
        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))) \
           or (np.any(pseries > cap_pn)) or (np.any(nseries > cap_pn)):
            return np.full(N_obs, float(big_vec_penalty), float)
        v = -BEGE_log_density(res, pseries, nseries, float(sigp), float(sign))
        v = np.asarray(v, float).reshape(-1)
        if v.size != N_obs: v = np.full(N_obs, float(v.ravel()[0]))
        if not np.all(np.isfinite(v)): v = np.full(N_obs, float(big_vec_penalty))
        return v

    # --- near-start sampler around center ---
    if center_params is None:
        raise ValueError("Provide center_params (true or benchmark vector) to draw near-starts.")

    center_params = np.asarray(center_params, float)
    if center_params.size != k:
        raise ValueError(f"center_params length {center_params.size} != expected {k}")

    def _one_near_start():
        # add *bounded* uniform noise around center coordinate-wise
        lo = np.array([b[0] for b in bounds_full], float)
        hi = np.array([b[1] for b in bounds_full], float)
        width = np.maximum(jitter_abs, jitter_rel*np.maximum(1.0, np.abs(center_params)))
        # sample δ ~ U[-width, +width]
        delta = (2.0*default_rng().random(center_params.shape) - 1.0)*width
        start = center_params + delta
        # clip to [center-width, center+width] ∩ [bounds]
        lo_near = np.maximum(center_params - width, lo)
        hi_near = np.minimum(center_params + width, hi)
        start = np.minimum(np.maximum(start, lo_near), hi_near)
        # final projection to hard bounds
        return _project_to_bounds(start, bounds_full)

    # --- optimization loop (include exact center as one start) ---
    best, best_fun = None, np.inf

    # start 0 = exact center (projected)
    s0 = _project_to_bounds(center_params, bounds_full)
    try:
        opt0 = minimize(_negloglik, s0, method='L-BFGS-B',
                        bounds=bounds_full, options={'maxiter': int(maxiter), 'ftol': float(tol)})
        if np.isfinite(opt0.fun) and opt0.fun < best_fun:
            best, best_fun = opt0, opt0.fun
    except Exception:
        pass

    for _ in range(int(n_starts)-1):
        init = _one_near_start()
        try:
            opt = minimize(_negloglik, init, method='L-BFGS-B',
                           bounds=bounds_full, options={'maxiter': int(maxiter), 'ftol': float(tol)})
            if np.isfinite(opt.fun) and opt.fun < best_fun:
                best, best_fun = opt, opt.fun
        except Exception:
            continue

    if best is None:
        raise RuntimeError("All near-starts failed. Try loosening jitter_abs/jitter_rel or cap_pn.")

    params = best.x
    ll     = -best.fun
    AIC    = 2*len(params) - 2*ll
    BIC    = np.log(N_obs)*len(params) - 2*ll

    # --- SEs via Hessian and OPG blend (as in your code) ---
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
    w, V = np.linalg.eigh(cov); w = np.maximum(w, 0.0); cov = (V*w) @ V.T
    se = np.sqrt(np.diag(cov))

    if print_summary:
        print("\n" + "-"*68)
        print("BEGE (Asym constants & sigmas; FULL GJR) — near-starts around center")
        print("-"*68)
        print(f"{'Parameter':<20}{'Estimate':>14}{'Std. Err.':>14}{'t-Stat':>14}")
        print("-"*68)
        for nm, val, err in zip(names_full, params, se):
            t = np.nan if err <= 0 else (val/err)
            print(f"{nm:<20}{val:>14.6f}{err:>14.6f}{t:>14.3f}")
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
        'bounds': bounds_full
    }
def evaluate_BEGE_fullGJR(Y, X, mean_type, params_full, cap_pn=120.0):
    """
    Computes LogLik/AIC/BIC for FULL-GJR parameter vector using Justin's BEGE log density,
    with the hard cap rule for p_t, n_t.
    """
    Y = np.asarray(Y, float); N = len(Y)

    if mean_type == 'constant':
        num_m = 0
        def mean_model(Y, X, pm): return Y
    elif mean_type == 'ARX(1,1)':
        num_m = 3
        def mean_model(Y, X, pm):
            c0, phi1, theta1 = pm
            Y = np.asarray(Y, float); X = np.asarray(X, float)
            n = min(len(Y), len(X)); Y = Y[:n]; X = X[:n]
            res = np.empty(n, float); res[0] = Y[0] - c0
            for t in range(1, n):
                res[t] = Y[t] - (c0 + phi1*Y[t-1] + theta1*X[t-1])
            return res
    else:
        raise ValueError("mean_type must be 'constant' or 'ARX(1,1)'")

    theta = np.asarray(params_full, float)
    pm = theta[:num_m]; rest = theta[num_m:]
    p0, n0, rho_p, rho_n, phi_p_plus, phi_p_minus, phi_n_plus, phi_n_minus, sigp, sign = rest
    res = mean_model(Y, X, pm)

    pseries = gjr_recursion(res, (float(p0), float(rho_p), float(phi_p_plus), float(phi_p_minus)), float(sigp))
    nseries = gjr_recursion(res, (float(n0), float(rho_n), float(phi_n_plus), float(phi_n_minus)), float(sign))

    valid = np.all(np.isfinite(pseries)) and np.all(np.isfinite(nseries)) \
            and (np.max(pseries) <= cap_pn) and (np.max(nseries) <= cap_pn)
    if not valid:
        return {'loglik': -np.inf, 'AIC': np.inf, 'BIC': np.inf, 'cap_hit': True}

    ll = BEGE_log_density(res, pseries, nseries, float(sigp), float(sign))
    ll_sum = float(np.sum(ll))
    k = len(theta)
    return {'loglik': ll_sum, 'AIC': 2*k - 2*ll_sum, 'BIC': np.log(N)*k - 2*ll_sum, 'cap_hit': False}
def run_full_bege_nearstart_experiment(
    T=3000,
    mean_type='constant',
    mean_params_true=None,      # {} for constant
    vol_params_true=None,       # dict with full-GJR keys
    # near-start controls
    jitter_rel=0.05, jitter_abs=1e-3, n_starts=25,
    # optimizer controls
    maxiter=800, tol=1e-8, random_state=7,
    # caps for evaluation
    cap_pn_eval=120.0
):
    # 1) simulate data
    Y, X, _ = simulate_bege_full(
        T=T,
        mean_type=mean_type,
        mean_params_true=mean_params_true,
        vol_params_true=vol_params_true,
        rng_seed=123,
        max_shape_cap=5000.0
    )

    # 2) pack true vector in BEGE_FullGJR order
    if mean_type == 'constant':
        theta_true = [
        ] + [
            vol_params_true['p0'], vol_params_true['n0'],
            vol_params_true['rho_p'], vol_params_true['rho_n'],
            vol_params_true['phi_p_plus'], vol_params_true['phi_p_minus'],
            vol_params_true['phi_n_plus'], vol_params_true['phi_n_minus'],
            vol_params_true['sigp'], vol_params_true['sign']
        ]
        names_full = ['p0','n0','rho_p','rho_n','phi_p⁺','phi_p⁻','phi_n⁺','phi_n⁻','σ₊','σ₋']
        true_map = {
            'p0': vol_params_true['p0'],
            'n0': vol_params_true['n0'],
            'rho_p': vol_params_true['rho_p'],
            'rho_n': vol_params_true['rho_n'],
            'phi_p⁺': vol_params_true['phi_p_plus'],
            'phi_p⁻': vol_params_true['phi_p_minus'],
            'phi_n⁺': vol_params_true['phi_n_plus'],
            'phi_n⁻': vol_params_true['phi_n_minus'],
            'σ₊': vol_params_true['sigp'],
            'σ₋': vol_params_true['sign'],
        }
    else:
        theta_true = [
            mean_params_true['const'], mean_params_true['phi1'], mean_params_true['theta1']
        ] + [
            vol_params_true['p0'], vol_params_true['n0'],
            vol_params_true['rho_p'], vol_params_true['rho_n'],
            vol_params_true['phi_p_plus'], vol_params_true['phi_p_minus'],
            vol_params_true['phi_n_plus'], vol_params_true['phi_n_minus'],
            vol_params_true['sigp'], vol_params_true['sign']
        ]
        names_full = ['const','Infl(1)','SPF','p0','n0','rho_p','rho_n','phi_p⁺','phi_p⁻','phi_n⁺','phi_n⁻','σ₊','σ₋']
        true_map = {
            'const': mean_params_true['const'], 'Infl(1)': mean_params_true['phi1'], 'SPF': mean_params_true['theta1'],
            'p0': vol_params_true['p0'], 'n0': vol_params_true['n0'],
            'rho_p': vol_params_true['rho_p'], 'rho_n': vol_params_true['rho_n'],
            'phi_p⁺': vol_params_true['phi_p_plus'], 'phi_p⁻': vol_params_true['phi_p_minus'],
            'phi_n⁺': vol_params_true['phi_n_plus'], 'phi_n⁻': vol_params_true['phi_n_minus'],
            'σ₊': vol_params_true['sigp'], 'σ₋': vol_params_true['sign'],
        }

    # 3) estimate with near-starts around the TRUE vector (mild shocks)
    fit = BEGE_FullGJR_MLE_nearstarts(
        Y, X=X, mean_type=mean_type,
        center_params=theta_true,
        jitter_rel=jitter_rel, jitter_abs=jitter_abs,
        n_starts=n_starts,
        maxiter=maxiter, tol=tol, random_state=random_state,
        print_summary=True
    )

    # 4) evaluate at truth and at estimate (Justin density)
    eval_true = evaluate_BEGE_fullGJR(Y, X, mean_type, theta_true, cap_pn=cap_pn_eval)
    eval_est  = {'loglik': fit['loglik'], 'AIC': fit['AIC'], 'BIC': fit['BIC']}

    # 5) pretty comparison
    print("\n=== True vs Estimated (Full-GJR; near-starts) ===")
    for nm, ev in zip(fit['names'], fit['params']):
        tv = true_map.get(nm, np.nan)
        if np.isfinite(tv):
            print(f"{nm:8s}  true={tv: .6f}   est={ev: .6f}   abs.err={abs(ev-tv): .6f}")
        else:
            print(f"{nm:8s}  est={ev: .6f}")

    print("\n[Estimated]   LogLik: {0:.6f}   AIC: {1:.6f}   BIC: {2:.6f}".format(
        eval_est['loglik'], eval_est['AIC'], eval_est['BIC']
    ))
    print("[True-params] LogLik: {0}   AIC: {1}   BIC: {2}".format(
        f"{eval_true['loglik']:.6f}" if np.isfinite(eval_true['loglik']) else str(eval_true['loglik']),
        f"{eval_true['AIC']:.6f}"    if np.isfinite(eval_true['AIC'])    else str(eval_true['AIC']),
        f"{eval_true['BIC']:.6f}"    if np.isfinite(eval_true['BIC'])    else str(eval_true['BIC'])
    ))

    return {'Y': Y, 'X': X, 'fit': fit, 'eval_true': eval_true}



# --- Run the near-start experiment around the new truth (constant mean) ---
out = run_full_bege_nearstart_experiment(
    T=300,                    # make larger if you want
    mean_type='constant',
    mean_params_true={},       # constant mean
    vol_params_true=vol_true_full,
    # near-starts centered at the numbers above (mildly shocked)
    jitter_rel=0.05,           # ±5% around center (scaled by |center|)
    jitter_abs=1e-3,           # protects very small centers
    n_starts=500,               # more starts -> more robust local search
    # optimizer controls
    maxiter=800, tol=1e-8, random_state=42,
    # evaluation cap for p_t, n_t overflow (no stability checks)
    cap_pn_eval=500.0
)