
```{raw:typst}
#set page(margin: auto)
```
 
# BEGE GARCH Family

All BEGE specifications use the same effective sample, residual definition,
mean-process menu, random-search protocol, density evaluator, and result
selection rules. The model-specific shape recursions and restrictions are
reported.

Every specification is estimated on the quarterly sample (1969Q2--2026Q3, 230
observations) and, as supplementary evidence, on the monthly sample
(1969M2--2026M9, 692 observations), both from the 2026-10-20 data vintage. The
specifications, parameter bounds, random search, and selection rules are
identical across the two frequencies; each specification reports its quarterly
best model followed by its monthly best model.

## Mean Processes

The BEGE exercises estimate all six mean processes:

- **Constant**: $\pi_{t+1} = SPF_t + u_{t+1}$.
- **ARX(1,0)**: $\pi_{t+1} = c + \rho_1 \pi_t + u_{t+1}$.
- **ARX(1,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \phi_1 SPF_t + u_{t+1}$.
- **ARX(2,0)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + u_{t+1}$.
- **ARX(2,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + u_{t+1}$.
- **ARX(2,2)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \phi_2 SPF_{t-1} + u_{t+1}$.

The mean-parameter bounds used in estimation and collection are:

| Model | $c$ | $\rho_1$ | $\rho_2$ | $\phi_1$ | $\phi_2$ |
|---|---|---|---|---|---|
| Constant | --- | --- | --- | --- | --- |
| ARX(1,0) | $(\min \pi_t,\max \pi_t)$ | $[-1,1]$ | --- | --- | --- |
| ARX(1,1) | $(\min \pi_t,\max \pi_t)$ | $[-1,1]$ | --- | $[-10,10]$ | --- |
| ARX(2,0) | $(\min \pi_t,\max \pi_t)$ | $[-2,2]$ | $[-1,1]$ | --- | --- |
| ARX(2,1) | $(\min \pi_t,\max \pi_t)$ | $[-2,2]$ | $[-1,1]$ | $[-10,10]$ | --- |
| ARX(2,2) | $(\min \pi_t,\max \pi_t)$ | $[-2,2]$ | $[-1,1]$ | $[-10,10]$ | $[-10,10]$ |

Best-model collection also imposes mean-process stationarity before a BEGE
estimate is eligible for reporting. For ARX(1,0) and ARX(1,1), this requires
$|\rho_1|<1$. For ARX(2,0), ARX(2,1) and ARX(2,2), the stationarity condition is the
usual AR(2) polynomial restriction:

$$
1-\rho_1 z-\rho_2 z^2 = 0
$$

must have both roots outside the unit circle. Equivalently, the reported
estimate must satisfy

$$
\rho_1+\rho_2 < 1,\qquad
\rho_2-\rho_1 < 1,\qquad
\rho_2 > -1.
$$

Only the inflation-lag coefficients $\rho_1$ and $\rho_2$ enter this
stationarity check; the SPF coefficients $\phi_1$ and $\phi_2$ are treated as
exogenous-regressor loadings and are not included in the AR polynomial.

Starting values for the estimated mean parameters are drawn uniformly from
OLS-centered intervals,  $\hat\theta_j \pm 2\,SE(\hat\theta_j)$, where
$\hat\theta_j$ and $SE(\hat\theta_j)$ are the OLS estimate and HAC(12)
standard error of the mean equation on the estimation sample.

## Residual Dynamics

Each BEGE model writes the inflation residual as

$$
u_t = \sigma_p \omega_{p,t} - \sigma_n \omega_{n,t},
\qquad
\omega_{p,t}\sim\tilde{\Gamma}(p_t,1),\quad
\omega_{n,t}\sim\tilde{\Gamma}(n_t,1),
$$

where $\tilde{\Gamma}(a,1)$ is a centered gamma shock with shape $a$ and unit
scale. Conditional on $p_t$ and $n_t$,

- conditional variance: $\sigma_p^2p_t+\sigma_n^2n_t$,
- conditional skewness numerator: $2(\sigma_p^3p_t-\sigma_n^3n_t)$,
- conditional excess-kurtosis numerator: $6(\sigma_p^4p_t+\sigma_n^4n_t)$.


## BEGE Volatility Specifications

The common parameter bounds used in estimation and result collection are:

| Parameter group | Bound |
|---|---|
| Fixed shapes $\bar{p},\bar{n}$ and intercepts $p_0,n_0$ | $[0,10]$ |
| Persistence parameters $\rho,\rho_p,\rho_n$ | $[0,1]$ |
| Shock-loading parameters $\phi^\pm,\phi_p,\phi_n,\phi_p^\pm,\phi_n^\pm$ | $[0,2]$ |
| Scale parameters $\sigma_p,\sigma_n$ | $[10^{-5},2]$ |

These parameter bounds, together with the stability restrictions of each
specification below, are the only restrictions on the BEGE variance path.
Under them, the implied variance $\sigma_p^2p_t+\sigma_n^2n_t$ is finite and
positive in any finite sample, and I impose no further bound on it. An earlier
version of the estimation also kept the implied variance inside a wide envelope
around an exponentially weighted moving average (EWMA) of squared residuals,
between $10^{-6}$ and $10^{6}$ times that average. I experimented with and
without this bound and removed it, because the likelihood already rules out the
paths it was meant to exclude. When the implied variance becomes much smaller
than the squared residual it must explain, that residual becomes an extreme
tail event; when the variance becomes very large, the density is spread too
thinly over the observed residuals. In both cases the model fits poorly and the
log likelihood falls sharply, so maximizing the likelihood itself avoids
explosive and vanishing variance paths.

To verify this, I estimated each of the five canonical BEGE specifications with
the Constant, ARX(1,1), ARX(2,1) and ARX(2,2) mean processes twice on
the 2026-10-20 data vintage, once with the EWMA bound and once without it, both
times from the same 50,000 random starting points. Without the bound, every
converged candidate estimate stayed inside the former envelope. The best
estimates lie far inside it:
their implied variance stays at least 80,000 times above the former lower bound
and at least 6,700 times below the former upper bound.
In 15 of the 20 combinations of specification and mean process, the best log
likelihoods of the two runs agree within $10^{-4}$. In the other five they
differ by at most $0.20$, in both directions. Each of those best estimates lies
inside the former bound and was therefore attainable in both runs, so the
differences reflect the random search rather than the bound.

### Constant BEGE

The Constant BEGE model keeps both shape parameters fixed:

$$
p_t=\bar{p},\qquad n_t=\bar{n}.
$$

Thus $u_t=\sigma_p\omega_{p,t}-\sigma_n\omega_{n,t}$ with
$\omega_{p,t}\sim\tilde{\Gamma}(\bar{p},1)$ and
$\omega_{n,t}\sim\tilde{\Gamma}(\bar{n},1)$.

### Symmetric BEGE

The Symmetric BEGE model uses the same persistence and shock-loading
parameters in both shape recursions:

$$
\begin{aligned}
p_t &= p_0 + \rho p_{t-1}
      + \frac{\phi^+}{2\sigma_p^2}(u_{t-1}^+)^2
      + \frac{\phi^-}{2\sigma_p^2}(u_{t-1}^-)^2,\\
n_t &= n_0 + \rho n_{t-1}
      + \frac{\phi^+}{2\sigma_n^2}(u_{t-1}^+)^2
      + \frac{\phi^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The stability restriction is

$$
\rho+\frac{\phi^++\phi^-}{2}<1.
$$

### BadGood BEGE

The BadGood BEGE model lets each shape process react symmetrically to squared
residuals, with separate good- and bad-environment parameters:

$$
\begin{aligned}
p_t &= p_0 + \rho_p p_{t-1}
      + \frac{\phi_p}{2\sigma_p^2}u_{t-1}^2,\\
n_t &= n_0 + \rho_n n_{t-1}
      + \frac{\phi_n}{2\sigma_n^2}u_{t-1}^2.
\end{aligned}
$$

The stability restrictions are

$$
\rho_p+\phi_p<1,\qquad \rho_n+\phi_n<1.
$$

### Inflation/Deflation BEGE

The Inflation/Deflation BEGE model lets the good-news shape respond only to
positive residuals and the bad-news shape respond only to negative residuals:

$$
\begin{aligned}
p_t &= p_0 + \rho_p p_{t-1}
      + \frac{\phi_p^+}{2\sigma_p^2}(u_{t-1}^+)^2,\\
n_t &= n_0 + \rho_n n_{t-1}
      + \frac{\phi_n^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The stability restrictions are

$$
\rho_p+\frac{\phi_p^+}{2}<1,\qquad
\rho_n+\frac{\phi_n^-}{2}<1.
$$



### Constant-p Full BEGE

Constant-p Full BEGE fixes the good-news shape and keeps the unrestricted Full
BEGE update for the bad-news shape:

$$
\begin{aligned}
p_t &= p_0,\\
n_t &= n_0 + \rho_n n_{t-1}
      + \frac{\phi_n^+}{2\sigma_n^2}(u_{t-1}^+)^2
      + \frac{\phi_n^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The dynamic shape process satisfies

$$
\rho_n+\frac{\phi_n^++\phi_n^-}{2}<1.
$$

### Constant-n Full BEGE

Constant-n Full BEGE fixes the bad-news shape and keeps the unrestricted Full
BEGE update for the good-news shape:

$$
\begin{aligned}
p_t &= p_0 + \rho_p p_{t-1}
      + \frac{\phi_p^+}{2\sigma_p^2}(u_{t-1}^+)^2
      + \frac{\phi_p^-}{2\sigma_p^2}(u_{t-1}^-)^2,\\
n_t &= n_0.
\end{aligned}
$$

The dynamic shape process satisfies

$$
\rho_p+\frac{\phi_p^++\phi_p^-}{2}<1.
$$

### Full BEGE

The Full BEGE model allows each shape process to respond separately to
positive and negative residuals:

$$
\begin{aligned}
p_t &= p_0 + \rho_p p_{t-1}
      + \frac{\phi_p^+}{2\sigma_p^2}(u_{t-1}^+)^2
      + \frac{\phi_p^-}{2\sigma_p^2}(u_{t-1}^-)^2,\\
n_t &= n_0 + \rho_n n_{t-1}
      + \frac{\phi_n^+}{2\sigma_n^2}(u_{t-1}^+)^2
      + \frac{\phi_n^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The stability restrictions are

$$
\rho_p+\frac{\phi_p^++\phi_p^-}{2}<1,\qquad
\rho_n+\frac{\phi_n^++\phi_n^-}{2}<1.
$$


### EGARCH BEGE (extension)

Beyond the canonical five specifications above, the `EGARCH_BEGE/` folder
contains an explicitly requested extension of Full BEGE in which the shape
processes follow coupled EGARCH-type recursions on the log shapes,

$$
\ln p_t = \omega_p + \beta_p \ln p_{t-1}
  + \alpha_p\left(|z_{t-1}| - \sqrt{2/\pi}\right) + \gamma_p z_{t-1},
$$

with $z_{t-1} = u_{t-1}/\sigma_{t-1}$ (and analogously for $\ln n_t$), and
stability restrictions $\beta_p<1$, $\beta_n<1$. See
`EGARCH_BEGE/SPECIFICATION.md` for the full specification, parameter bounds,
and the mapping to the equivalent EGARCH recursion on the component
volatilities $\sigma_p\sqrt{p_t}$ and $\sigma_n\sqrt{n_t}$. All shared BEGE
settings (mean menu, density evaluator, random-search protocol, selection
rules) are inherited unchanged.


## Random Search

For each BEGE specification and mean process, I randomly sample 10,000
starting points and run the optimizer from each of them.

For volatility parameters, starting values are sampled inside the documented
bounds. Shape intercept starts $p_0,n_0$ are sampled from the positive part of
$[0,10]$, scale starts $\sigma_p,\sigma_n$ from $[10^{-5},2]$, and persistence
and shock-loading starts are drawn so the relevant stability restriction is
satisfied before optimization begins.

When a dynamic BEGE recursion is evaluated, I initialize the
pre-sample $p$ or $n$ state by the parameter-implied unconditional backcast,
$p_0/(1-\rho-\tfrac{1}{2}(\phi^+ + \phi^-))$ or the corresponding restricted
formula, with a small positive numerical floor.[^shape-floor] Constant-p and
Constant-n models set the constant shape path directly.

[^shape-floor]: The numerical floor is $10^{-4}$: if the initial backcast or a later recursion produces a smaller value $p_t,n_t$, the
    code uses $0.0001$ instead.

## Density Evaluation

Likelihood evaluation uses the stabilized BEGE density implementation
documented in the BEGE Density section. The current evaluator keeps the
closed-form hypergeometric expression in stable regions, switches to a guarded
saddlepoint approximation when the closed form becomes numerically fragile.

## Selection And Reporting

For each BEGE specification, the reported best-model page contains only the
likelihood-best admissible estimate across mean processes. The by-mean best
rows remain available in the CSV outputs linked from each report. Selection
requires successful optimizer convergence, mean-process stationarity, and the
documented parameter and stability constraints.


Each report also gives empirical 5%, median, and 95% quantiles for $p_t$,
$n_t$, $\operatorname{Var}_t(u_t)$, the skewness numerator, and the
excess-kurtosis numerator.

The best-model summary table also reports the number of observations $N$
and the sample period of the effective sample.

## Standard Errors

I compute standard errors for the reported best estimate. For each
observation, I approximate the score---the first derivative of that
observation's log likelihood with respect to the parameters---using centered
finite differences. I approximate the Hessian---the matrix of second
derivatives of the total negative log likelihood---using the four-point
centered finite-difference method implemented by
`statsmodels.tools.numdiff.approx_hess`.

The default covariance is the sandwich matrix

$$
\widehat{V}(\hat\theta)
=
\widehat{H}^{-1}
\left(\sum_{t=1}^T \hat g_t \hat g_t'\right)
\widehat{H}^{-1},
$$

where $\widehat{H}$ is the Hessian of the negative log likelihood and
$\hat g_t$ is the numerical score contribution for observation $t$. If the
sandwich calculation is numerically unstable, the collector uses an
observed-information or inverse-OPG fallback. Standard errors that cannot be
reliably identified are reported as `NA`.
