"""Reduced coordinates for the two one-parameter restrictions of Full BEGE."""
from __future__ import annotations

import numpy as np

RESTRICTIONS = {"symmetric_p", "positive_p"}


def parameter_names(restriction):
    if restriction not in RESTRICTIONS:
        raise ValueError(f"Unknown Full BEGE restriction: {restriction!r}")
    return ["p0", "n0", "rho_p", "rho_n",
            "phi_p" if restriction == "symmetric_p" else "phi_p_plus",
            "phi_n_plus", "phi_n_minus", "sigma_p", "sigma_n"]


def expand_volatility(vol, restriction):
    """Insert the constrained negative-p loading into Full's ten coordinates."""
    parameter_names(restriction)  # Validate even if the vector has the wrong size.
    vol = np.asarray(vol, dtype=float)
    if vol.shape != (9,):
        raise ValueError("Restricted Full BEGE requires nine volatility parameters.")
    return np.insert(vol, 5, vol[4] if restriction == "symmetric_p" else 0.0)


def expand_row(row, restriction):
    expanded = row.copy()
    loading = row["param_phi_p" if restriction == "symmetric_p" else "param_phi_p_plus"]
    expanded["param_phi_p_plus"] = loading
    expanded["param_phi_p_minus"] = loading if restriction == "symmetric_p" else 0.0
    return expanded


def sample_volatility(rng, restriction, *, p0_bounds, rho_bounds, phi_bounds,
                      sigma_bounds, floor_eps):
    """Uniform draws with rejection of the documented stability restrictions.

    Conditional on rho, the common p loading has cap 1-rho-delta;
    the positive-only loading has cap 2(1-rho-delta). The n loadings
    follow Full BEGE's half-loading draws and joint stability rejection.
    """
    parameter_names(restriction)
    p_weight = 1.0 if restriction == "symmetric_p" else 0.5
    rp_hi = min(rho_bounds[1], 1.0 - floor_eps - p_weight * phi_bounds[0])
    rn_hi = min(rho_bounds[1], 1.0 - floor_eps - phi_bounds[0])
    if min(rp_hi, rn_hi) <= rho_bounds[0]:
        raise ValueError("Bounds leave no stable restricted Full BEGE starts.")
    for _ in range(10000):
        rp, rn = rng.uniform(rho_bounds[0], rp_hi), rng.uniform(rho_bounds[0], rn_hi)
        cap_p = min(phi_bounds[1], (1.0 - floor_eps - rp) / p_weight)
        cap_n = min(phi_bounds[1], 2.0 * (1.0 - floor_eps - rn))
        if min(cap_p, cap_n) <= phi_bounds[0]:
            continue
        pp = rng.uniform(phi_bounds[0], cap_p)
        np_, nm = rng.uniform(phi_bounds[0], cap_n, size=2)
        if rn + (np_ + nm) / 2.0 >= 1.0 - floor_eps:
            continue
        p0, n0 = rng.uniform(*p0_bounds, size=2)
        sp, sn = rng.uniform(*sigma_bounds, size=2)
        return np.array([p0, n0, rp, rn, pp, np_, nm, sp, sn])
    raise RuntimeError("Unable to draw a stable restricted Full BEGE start.")
