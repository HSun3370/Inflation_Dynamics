
```{raw:typst}
#set page(margin: auto)
```
 
# BEGE GARCH Family

All BEGE specifications use the same effective sample, residual definition,
mean-process menu, random-search protocol, density evaluator, and result
selection rules. The model-specific shape recursions and restrictions are
reported.

## Mean Processes

The BEGE exercises estimate all four mean processes:

- **Constant**: $\pi_{t+1} = SPF_t + u_{t+1}$.
- **ARX(1,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \phi_1 SPF_t + u_{t+1}$.
- **ARX(2,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + u_{t+1}$.
- **ARX(2,2)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \phi_2 SPF_{t-1} + u_{t+1}$.

The mean-parameter bounds used in estimation and collection are:

| Model | $c$ | $\rho_1$ | $\rho_2$ | $\phi_1$ | $\phi_2$ |
|---|---|---|---|---|---|
| Constant | --- | --- | --- | --- | --- |
| ARX(1,1) | $(\min \pi_t,\max \pi_t)$ | $[-1,1]$ | --- | $[-10,10]$ | --- |
| ARX(2,1) | $(\min \pi_t,\max \pi_t)$ | $[-2,2]$ | $[-1,1]$ | $[-10,10]$ | --- |
| ARX(2,2) | $(\min \pi_t,\max \pi_t)$ | $[-2,2]$ | $[-1,1]$ | $[-10,10]$ | $[-10,10]$ |

Best-model collection also imposes mean-process stationarity before a BEGE
estimate is eligible for reporting. For ARX(1,1), this requires
$|\rho_1|<1$. For ARX(2,1) and ARX(2,2), the stationarity condition is the
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
OLS-centered intervals,  $\hat\theta_j \pm 2\,SE(\hat\theta_j)$.

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


To rule out numerically degenerate and explosive variance paths, I use an EWMA path as a scale-adaptive reference for the
BEGE implied variance. This is a loose numerical safeguard, not a
variance-targeting restriction.[^ewma-bound] For residuals $u_t$, define

$$
EWMA_1 =
\frac{\sum_{j=1}^{\tau}\lambda^{j-1}u_j^2}
     {\sum_{j=1}^{\tau}\lambda^{j-1}},
\qquad
EWMA_t = \lambda EWMA_{t-1} + (1-\lambda)u_{t-1}^2,\quad t\geq 2.
$$

The EWMA implied-variance bounds are imposed during optimization and checked
again during collection. With $\lambda=0.94$ and $\tau=\min(75,T)$, let
$s_u^2:=1/T \sum_{j=1}^{T} u_j^2$ and
$M_u:=1+\max_{1\leq s\leq T}u_s^2$. Then

$$
\begin{aligned}
\underline v_t
&:= \max \left\{10^{-6}EWMA_t,\;10^{-8}s_u^2\right\},\\
\overline v_t
&:= \max \left\{\min\left(10^6EWMA_t,\;10^7M_u\right),\;M_u\right\},\\
\underline v_t
&\leq \sigma_p^2p_t+\sigma_n^2n_t \leq \overline v_t,
\qquad t=1,\ldots,T.
\end{aligned}
$$

[^ewma-bound]: The exponential weights follow the variance-forecasting idea in
    J.P. Morgan/Reuters, [*RiskMetrics---Technical Document*, Fourth Edition
    (1996), Chapter 5](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a),
    where recent squared shocks receive more weight than older shocks. Here the
    EWMA path is only a smoothed proxy for the local scale of $u_t^2$, not a
    BEGE variance target. The loose envelope is adapted from the `arch`
    package's
    [`VolatilityProcess.variance_bounds`](https://arch.readthedocs.io/en/latest/_modules/arch/univariate/volatility.html#VolatilityProcess.variance_bounds):
    the lower bound prevents the implied BEGE variance from collapsing toward
    zero, while the upper bound screens out explosive variance paths that would
    make likelihood evaluation numerically unreliable.

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


## Random Search

BEGE estimation uses 50 independent seed jobs for each volatility
specification. Within each seed job, each mean process uses 40 saved random
draws, and each draw runs 25 optimizer starts. Therefore each BEGE
specification and mean-process pair uses

$$
50 \times 40 \times 25 = 50{,}000
$$

optimizer starts.

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
requires successful optimizer convergence, mean-process stationarity,
documented parameter and stability constraints, and EWMA implied-variance
bounds.


Each report also gives empirical 5%, median, and 95% quantiles for $p_t$,
$n_t$, $\operatorname{Var}_t(u_t)$, the skewness numerator, and the
excess-kurtosis numerator.

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
