```{raw:typst}
#set page(margin: auto)
```

# Regime-Switching Inflation Models


Let the latent state be
$$
s_t \in \{1,\dots,K\}, \qquad \Pr(s_t = j | s_{t-1} = i) = p_{ij}.
$$

I estimate several specifications that differ in **which blocks of parameters are allowed to switch with $s_t$**. Taking the ARX(2,2) model as an example,
$$
\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1\, \mathrm{SPF}_t + \phi_2\, \mathrm{SPF}_{t-1} + \mu_{t+1},
$$
the parameters are grouped into three blocks, and each block can independently be made regime-dependent:

| Block | State-dependent parameters |
|---|---|
| AR component | $c_{s_t},\ \rho_{1,s_t},\ \rho_{2,s_t}$ |
| SPF component | $\phi_{1,s_t},\ \phi_{2,s_t}$ |
| Shock distribution | $\sigma_{s_t},\ \nu$ |

where $\nu$ denotes the common degrees of freedom when assuming standardized
Student-$t$ distribution for residuals. I keep $\nu$ common across regimes in
the baseline estimation because regime-specific degrees of freedom are weakly
identified and tend to run to the Gaussian limit in sparsely populated regimes.
A given specification is defined by selecting any subset of these three blocks
to be switching; parameters in blocks not selected are held constant across
regimes.

## Estimation And Collection

For each regime-switching specification, I run 50 optimizer starts. Each start
first uses 50 EM iterations to improve the starting values and then applies MLE
to fine tune the likelihood. The collected estimate for a model specification is
the highest log-likelihood attempt that passes all checks:

- optimizer convergence,
- AR mean-process stationarity in every regime,
- positive non-boundary variance and interior, non-Gaussian-limit Student-$t$ degrees of freedom when applicable,
- and valid non-boundary transition probabilities.

The collection step rejects selected attempts with variance estimates at or
below `max(1e-6, 1e-3 * var(endog))`, transition probabilities within `1e-4`
of 0 or 1, or Student-$t$ degrees of freedom at or above `80`.

For the AR stationarity check, only the inflation-lag coefficients enter the AR
polynomial. SPF coefficients are treated as exogenous loadings. If none of the
50 attempts passes all checks, the result table reports the highest-likelihood
attempt as a flagged fallback rather than silently dropping the model.
