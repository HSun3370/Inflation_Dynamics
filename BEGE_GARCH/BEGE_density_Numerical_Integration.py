# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:34:47 2024

@author: Administrator
"""
import numpy as np

from scipy.stats import gamma




def loglikedgam(resids, p, n, tp, tn, zint=0.01, npoints = 1000):
    resids = np.asarray(resids)
    p = np.asarray(p)
    n = np.asarray(n)
    
    z_lower = resids - zint / 2
    z_upper = resids + zint / 2

    results = np.zeros((2, len(resids)))
    
    for idx, z_val in enumerate([z_lower, z_upper]):
        pmin = -p * tp #+ 1e-4
        pmax = 10 * np.sqrt(p) * tp 
        pgrid = np.linspace(pmin , pmax , npoints)

        ngrid = pgrid - (pmax - pmin)  / npoints  - z_val 
        
        cp = gamma.cdf(pgrid, p, loc=-p  * tp, scale=tp)
        pp = np.diff(np.concatenate([np.zeros(resids.shape).reshape(1,-1), cp]), axis=0)
        cn = 1 - gamma.cdf(ngrid, n , loc=-n  * tn, scale=tn)
        cz = np.sum(cn * pp, axis=0)
        results[idx] = cz

    pz = np.diff(results, axis=0) / zint
    loglik = np.log(np.clip(pz, 1e-20, 1e20))
    
    return loglik

def loglikedgam_constant(resids, p, n, tp, tn, zint=0.01, npoints = 500):
    resids = np.asarray(resids)
    p = np.full_like(resids, p, dtype=np.double)
    n = np.full_like(resids, n, dtype=np.double)
    
    z_lower = resids - zint / 2
    z_upper = resids + zint / 2

    results = np.zeros((2, len(resids)))
    
    for idx, z_val in enumerate([z_lower, z_upper]):
        pmin = -p * tp #+ 1e-4
        pmax = 10 * np.sqrt(p) * tp 
        pgrid = np.linspace(pmin , pmax , npoints)

        ngrid = pgrid - (pmax - pmin)  / npoints  - z_val 
        
        cp = gamma.cdf(pgrid, p, loc=-p  * tp, scale=tp)
        pp = np.diff(np.concatenate([np.zeros(resids.shape).reshape(1,-1), cp]), axis=0)
        cn = 1 - gamma.cdf(ngrid, n , loc=-n  * tn, scale=tn)
        cz = np.sum(cn * pp, axis=0)
        results[idx] = cz

    pz = np.diff(results, axis=0) / zint
    loglik = np.log(np.clip(pz, 1e-20, 1e20))
    
    return loglik