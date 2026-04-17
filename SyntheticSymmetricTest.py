
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from BEGE_GARCH import BEGE_AsymSharedGJR_MLE, gjr_recursion
from BEGE_density import *
import numpy as np
from numpy.random import default_rng
from scipy.stats import gamma as _gamma
from scipy.optimize import minimize

import os
  
parser = argparse.ArgumentParser(description="seed sets")

parser.add_argument("--id", type=int, default=1) 
args = parser.parse_args()  # <<< after all arguments are added
seed = args.id  

 
# =========================================================
# 0) TRUE PARAMS (With-Justin)
# =========================================================
# vol_true = dict(
#     p0=2.45393007,
#     n0=0.039471529,
#     rho=0.424955083,
#     phi_plus=0.691490105,
#     phi_minus=0.126719492,
#     sig_p=0.153025178,
#     sig_n=1.299725733
# )

vol_true = dict(
    p0   = 1.40,
    n0   = 0.40,
    rho  = 0.45,
    phi_plus = 0.30,
    phi_minus= 0.12,
    sig_p = 0.20,
    sig_n = 0.90
)
# =========================================================
# 1) SIMULATOR (constant-mean; uses your BEGE-GJR recursion)
# =========================================================
def simulate_bege_sharedgjr_constant(vol_params, T=10000, seed=123, max_shape_cap=None):
    """
    Simulate BEGE with asymmetric constants (p0,n0), shared GJR (rho, phi+/phi-),
    and separate scales (sig_p, sig_n). Constant-mean case: Y_t = e_t.

    X_t = σ_p (Γ_p,t − p_t) − σ_n (Γ_n,t − n_t),
    Γ_p,t ~ Gamma(shape=p_t, scale=1), Γ_n,t ~ Gamma(shape=n_t, scale=1),
    p_t, n_t via shared-GJR recursion driven by X_{t-1}.
    """
    p0, n0  = float(vol_params['p0']), float(vol_params['n0'])
    rho     = float(vol_params['rho'])
    phiP    = float(vol_params['phi_plus'])
    phiN    = float(vol_params['phi_minus'])
    sigP    = float(vol_params['sig_p'])
    sigN    = float(vol_params['sig_n'])

    rng = default_rng(seed)

    e = np.zeros(T, dtype=float)
    pseries = np.zeros(T, dtype=float)
    nseries = np.zeros(T, dtype=float)

    # Backcast per your recursion
    denom = max(1e-8, 1.0 - rho - 0.5*(phiP + phiN))
    pseries[0] = max(p0/denom, 1e-4)
    nseries[0] = max(n0/denom, 1e-4)

    gp0 = _gamma.rvs(a=pseries[0], scale=1.0, random_state=rng)
    gn0 = _gamma.rvs(a=nseries[0], scale=1.0, random_state=rng)
    e[0] = sigP*(gp0 - pseries[0]) - sigN*(gn0 - nseries[0])

    inv_den_p = 1.0/(2.0*sigP*sigP)
    inv_den_n = 1.0/(2.0*sigN*sigN)

    for t in range(1, T):
        last = e[t-1]
        phi = (phiP if last > 0.0 else phiN)

        p_t = p0 + rho*pseries[t-1] + (last*last)*phi*inv_den_p
        n_t = n0 + rho*nseries[t-1] + (last*last)*phi*inv_den_n

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
        e[t] = sigP*(gp - p_t) - sigN*(gn - n_t)

    Y = e  # constant mean
    return {'Y': Y, 'pseries': pseries, 'nseries': nseries}

# =========================================================
# 2) EVALUATE TRUE LL / IC WITH JUSTIN'S DENSITY (constant mean)
# =========================================================
def evaluate_ll_ic_true(Y, vol_params, cap_pn=120.0):
    resids = np.asarray(Y, dtype=float)
    p0, n0  = vol_params['p0'], vol_params['n0']
    rho     = vol_params['rho']
    phip    = vol_params['phi_plus']
    phin    = vol_params['phi_minus']
    sigp    = vol_params['sig_p']
    sign    = vol_params['sig_n']

    pseries = gjr_recursion(resids, (p0, rho, phip, phin), sigp)
    nseries = gjr_recursion(resids, (n0, rho, phip, phin), sign)

    if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))) \
       or (np.any(pseries > cap_pn)) or (np.any(nseries > cap_pn)):
        return dict(loglik=-np.inf, AIC=np.inf, BIC=np.inf, N=len(Y), k=7)

    ll_vec = BEGE_log_density(resids, pseries, nseries, sigp, sign)
    ll     = float(np.sum(ll_vec))
    N      = len(Y); k = 7
    AIC    = 2*k - 2*ll
    BIC    = np.log(N)*k - 2*ll
    return dict(loglik=ll, AIC=AIC, BIC=BIC, N=N, k=k)

# =========================================================
# 3) ESTIMATOR WITH WIDE BOUNDS + NARROW STARTS NEAR TRUTH
# =========================================================
def BEGE_AsymSharedGJR_MLE_custom_starts(
    Y,
    mean_type='constant',
    # wide bounds — keep as in your original use
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.005, 10.0),
    rho_bounds=(1e-5, 0.999),
    phi_bounds=(1e-5, 0.999),
    # starting-point control
    init_center=None,          # dict with keys: p0,n0,rho,phi_plus,phi_minus,sig_p,sig_n
    init_radius_pct=0.10,      # ±10% box around center for starts
    n_starts=40,
    frac_near=0.85,            # fraction of starts drawn near truth
    cap_pn=120.0,
    tol=1e-8,
    maxiter=1200,
    random_state=7,
    print_summary=True
):
    """
    Same model and objective as your BEGE_AsymSharedGJR_MLE (constant mean),
    BUT: starting points are mostly drawn in a narrow box around 'init_center',
    while bounds remain wide.
    """
    rng = default_rng(random_state)
    Y = np.asarray(Y, dtype=float)
    N = Y.shape[0]

    # ---- mean model ----
    if mean_type != 'constant':
        raise ValueError("This helper is written for mean_type='constant'.")
    names_mean = []
    num_m = 0

    # ---- bounds (wide) ----
    (sig_lo, sig_hi) = sigma_bounds
    (p0_lo,  p0_hi)  = p0n0_bounds
    (rho_lo, rho_hi) = rho_bounds
    (phi_lo, phi_hi) = phi_bounds

    bounds_full = [
        (p0_lo, p0_hi),     # p0
        (p0_lo, p0_hi),     # n0
        (rho_lo, rho_hi),   # rho
        (phi_lo, phi_hi),   # phi+
        (phi_lo, phi_hi),   # phi-
        (sig_lo, sig_hi),   # sigma_p
        (sig_lo, sig_hi)    # sigma_n
    ]
    names_vol = ['p0','n0','rho','phi+','phi-','σ+','σ-']
    names_full = names_mean + names_vol

    # ---- objective & vector objective (your recursion + Justin density) ----
    BIG = 1e12
    def _negloglik(theta):
        p0, n0, rho, phip, phin, sigp, sign = theta
        res = Y
        pseries = gjr_recursion(res, (float(p0), float(rho), float(phip), float(phin)), float(sigp))
        nseries = gjr_recursion(res, (float(n0), float(rho), float(phip), float(phin)), float(sign))
        if (not np.all(np.isfinite(pseries))) or (not np.all(np.isfinite(nseries))) \
           or (np.any(pseries > cap_pn)) or (np.any(nseries > cap_pn)):
            return BIG
        ll = BEGE_log_density(res, pseries, nseries, float(sigp), float(sign))
        val = -float(np.sum(ll))
        if not np.isfinite(val):
            return BIG
        return val

    # ---- near-truth sampler ----
    def _near_box(val, pct, lo, hi):
        """Uniform in [val*(1-pct), val*(1+pct)] intersected with [lo,hi]."""
        a = max(lo, val*(1.0 - pct))
        b = min(hi, val*(1.0 + pct))
        if b <= a:  # fallback tiny box
            a = max(lo, min(hi, val*0.999))
            b = min(hi, max(lo, val*1.001))
        return rng.uniform(a, b)

    def _draw_start_near(center, pct):
        p0  = _near_box(center['p0'],       pct, p0_lo,  p0_hi)
        n0  = _near_box(center['n0'],       pct, p0_lo,  p0_hi)   # same bounds struct
        rho = _near_box(center['rho'],      pct, rho_lo, rho_hi)
        php = _near_box(center['phi_plus'], pct, phi_lo, phi_hi)
        phn = _near_box(center['phi_minus'],pct, phi_lo, phi_hi)
        sp  = _near_box(center['sig_p'],    pct, sig_lo, sig_hi)
        sn  = _near_box(center['sig_n'],    pct, sig_lo, sig_hi)
        return np.array([p0,n0,rho,php,phn,sp,sn], dtype=float)

    def _draw_start_global():
        return np.array([
            rng.uniform(p0_lo,  p0_hi),
            rng.uniform(p0_lo,  p0_hi),
            rng.uniform(rho_lo, rho_hi),
            rng.uniform(phi_lo, phi_hi),
            rng.uniform(phi_lo, phi_hi),
            rng.uniform(sig_lo, sig_hi),
            rng.uniform(sig_lo, sig_hi),
        ], dtype=float)

    # ---- build starting points ----
    if init_center is None:
        raise ValueError("Please pass init_center=dict(p0, n0, rho, phi_plus, phi_minus, sig_p, sig_n).")
    starts = []
    n_near = int(round(frac_near * n_starts))
    for _ in range(n_near):
        starts.append(_draw_start_near(init_center, init_radius_pct))
    for _ in range(n_starts - n_near):
        starts.append(_draw_start_global())
    starts = np.array(starts)

    # ---- multi-start local optimization (L-BFGS-B)
    best = None
    best_fun = np.inf
    for x0 in starts:
        try:
            out = minimize(_negloglik, x0, method='L-BFGS-B',
                           bounds=bounds_full,
                           options={'maxiter': int(maxiter), 'ftol': float(tol)})
            if np.isfinite(out.fun) and out.fun < best_fun:
                best_fun = float(out.fun)
                best = out
        except Exception:
            continue

    if best is None:
        raise RuntimeError("All starts failed; consider loosening init_radius_pct or cap_pn.")

    params = best.x
    ll = -best.fun
    k  = len(params)
    AIC = 2*k - 2*ll
    BIC = np.log(N)*k - 2*ll

    if print_summary:
        print("\n" + "-"*68)
        print("BEGE (Asym constants & sigmas; shared GJR) — custom starts near truth")
        print("-"*68)
        print(f"{'Parameter':<12}{'Estimate':>14}")
        print("-"*68)
        for nm, v in zip(names_full, params):
            print(f"{nm:<12}{v:>14.6f}")
        print("-"*68)
        print(f"{'LogLik':<12}{ll:>14.6f}")
        print(f"{'AIC':<12}{AIC:>14.6f}")
        print(f"{'BIC':<12}{BIC:>14.6f}")
        print("-"*68)

    return {
        'opt': best,
        'params': params,
        'AIC': AIC,
        'BIC': BIC,
        'loglik': ll,
        'names': names_full
    }


sim = simulate_bege_sharedgjr_constant(vol_true, T=300, seed=1031, max_shape_cap=500.0)
Y_long = sim['Y']

# (b) estimation with wide bounds, but starts narrowly around truth
est = BEGE_AsymSharedGJR_MLE_custom_starts(
    Y_long,
    mean_type='constant',
    sigma_bounds=(1e-5, 2.0),
    p0n0_bounds=(0.005, 10.0),
    rho_bounds=(1e-5, 0.999),
    phi_bounds=(1e-5, 0.999),
    init_center=vol_true,       # <-- narrow starts centered at the *true* params
    init_radius_pct=0.10,       # +/-10% box
    n_starts=50,
    frac_near=0.9,
    cap_pn=500.0,
    tol=1e-8,
    maxiter=500,
    random_state=seed * 100,
    print_summary=True
)

# (c) compare estimate with truth; also LL/IC at true parameters
names = est['names']
hat   = est['params']

true_map = {
    'p0': vol_true['p0'], 'n0': vol_true['n0'], 'rho': vol_true['rho'],
    'phi+': vol_true['phi_plus'], 'phi-': vol_true['phi_minus'],
    'σ+': vol_true['sig_p'], 'σ-': vol_true['sig_n']
}

print("\n=== True vs Estimated (constant-mean; wide bounds but narrow starts) ===")
for nm, vhat in zip(names, hat):
    vtrue = true_map[nm]
    print(f"{nm:5s}  true={vtrue: .6f}   est={vhat: .6f}   abs.err={abs(vhat-vtrue): .6f}")

print(f"\n[Estimated] LogLik: {est['loglik']:.6f}   AIC: {est['AIC']:.6f}   BIC: {est['BIC']:.6f}")
true_eval = evaluate_ll_ic_true(Y_long, vol_true, cap_pn=500.0)
print(f"[True-params] LogLik: {true_eval['loglik']:.6f}   AIC: {true_eval['AIC']:.6f}   BIC: {true_eval['BIC']:.6f}")

