```{raw:typst}
#set page(margin: auto)
```

# Restricted Full BEGE specifications outside the report

These specifications are retained as research documentation. They are excluded from the PDF at the user’s request. The shared Full BEGE estimation settings remain documented in README.md.

### Symmetric-p/full-n BEGE

This restriction of Full BEGE sets $\phi_p^+=\phi_p^-=\phi_p$, so the
positive-shape process reacts equally to positive and negative residuals.
The negative-shape process retains separate loadings:

$$
\begin{aligned}
p_t &= p_0+\rho_p p_{t-1}+\frac{\phi_p}{2\sigma_p^2}u_{t-1}^2,\\
n_t &= n_0+\rho_n n_{t-1}
+\frac{\phi_n^+}{2\sigma_n^2}(u_{t-1}^+)^2
+\frac{\phi_n^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The restrictions are $\rho_p+\phi_p<1-\delta$ and
$\rho_n+(\phi_n^++\phi_n^-)/2<1-\delta$.
There are nine free volatility parameters. The symmetry here is across
residual signs in the $p_t$ recursion; the earlier Symmetric BEGE instead
shares persistence and shock loadings across the two shape processes.

### Positive-only-p/full-n BEGE

This restriction of Full BEGE sets $\phi_p^-=0$. Thus only positive residuals
enter the positive-shape update, while the negative-shape update remains full:

$$
\begin{aligned}
p_t &= p_0+\rho_p p_{t-1}
+\frac{\phi_p^+}{2\sigma_p^2}(u_{t-1}^+)^2,\\
n_t &= n_0+\rho_n n_{t-1}
+\frac{\phi_n^+}{2\sigma_n^2}(u_{t-1}^+)^2
+\frac{\phi_n^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The restrictions are $\rho_p+\phi_p^+/2<1-\delta$ and
$\rho_n+(\phi_n^++\phi_n^-)/2<1-\delta$.
There are nine free volatility parameters. Unlike Inflation/Deflation BEGE,
this specification estimates both $\phi_n^+$ and $\phi_n^-$.

Both restrictions use the Full BEGE likelihood, first-state convention,
numerical shape guard, and estimation protocol. The restriction is imposed
through the free parameter vector, so AIC, BIC, and standard errors count the
shared or fixed loading only once or not at all, respectively. Comparisons
with Full BEGE use the same mean process and frequency. A likelihood loss
below 1 favors either restriction by AIC; a loss below $\tfrac12\log N$
favors it by BIC. These thresholds compare fit and parameter count; they
are not tests of statistical significance.

## Additional parameter bounds and starts

| Parameter | Bounds and restrictions | Random-start draw |
|---|---|---|
| **Symmetric-p/full-n and positive-only-p/full-n** | | |
| $\rho_p,\rho_n$ | $[0,1]$ | $U[0,1-\delta]$ |
| $\phi_p$ (symmetric-p) | $[0,2]$; $\rho_p+\phi_p<1-\delta$ | $U[0,1-\delta-\rho_p]$ given $\rho_p$ |
| $\phi_p^+$ (positive-only-p) | $[0,2]$; $\rho_p+\phi_p^+/2<1-\delta$ | $U[0,2(1-\delta-\rho_p)]$ given $\rho_p$; $\phi_p^-=0$ |
| $\phi_n^+,\phi_n^-$ (both) | $[0,2]$; $\rho_n+(\phi_n^++\phi_n^-)/2<1-\delta$ | Same triangular draw as Full BEGE, given $\rho_n$ |
