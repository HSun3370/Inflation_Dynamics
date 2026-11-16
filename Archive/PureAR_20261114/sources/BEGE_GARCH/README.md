
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
identical across the two frequencies. The report presents the quarterly best models first, followed by the monthly best models.

## Mean Processes

The BEGE exercises estimate all six mean processes:

- **Constant**: $\pi_{t+1} = SPF_t + u_{t+1}$.
- **ARX(1,0)**: $\pi_{t+1} = c + \rho_1 \pi_t + u_{t+1}$.
- **ARX(1,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \phi_1 SPF_t + u_{t+1}$.
- **ARX(2,0)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + u_{t+1}$.
- **ARX(2,1)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + u_{t+1}$.
- **ARX(2,2)**: $\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \phi_2 SPF_{t-1} + u_{t+1}$.

The bounds and starting-value draws of the mean parameters are listed with
those of the volatility parameters in the
[parameter table](#estimation-and-random-starts) under Estimation and Random
Starts.

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

## Residual Dynamics

Each BEGE model writes the inflation residual as

$$
u_t = \sigma_p \omega_{p,t} - \sigma_n \omega_{n,t},
\qquad
\omega_{p,t}\sim\tilde{\Gamma}(p_t,1),\quad
\omega_{n,t}\sim\tilde{\Gamma}(n_t,1),
$$

where $\tilde{\Gamma}(a,1)$ is the distribution of $G-a$ for $G\sim\Gamma(a,1)$ (shape $a$, unit scale). The two gamma shocks are conditionally independent given $\mathcal{F}_{t-1}$; their shapes $p_t,n_t$ are known at that time, and $\sigma_p,\sigma_n$ are fixed scale parameters. Define the conditional variance and third and fourth cumulants:

$$
\begin{aligned}
\sigma_t^2&=\sigma_p^2p_t+\sigma_n^2n_t,\\
\kappa_{3,t}&=2(\sigma_p^3p_t-\sigma_n^3n_t),\\
\kappa_{4,t}&=6(\sigma_p^4p_t+\sigma_n^4n_t).
\end{aligned}
$$

Conditional skewness is $\kappa_{3,t}/\sigma_t^3$ and conditional excess kurtosis is $\kappa_{4,t}/\sigma_t^4$. The path-quantile tables report the **unstandardized cumulants** $\kappa_{3,t},\kappa_{4,t}$. Their quantiles describe variation across sample dates, not uncertainty about the estimates.

In the dynamic shape recursions below, $p_0,n_0$ denote intercept parameters, not the initial values of the shape paths. Shape persistence and shock loadings are distinct from the mean coefficients $\rho_1,\rho_2,\phi_1,\phi_2$.


## BEGE Volatility Specifications

The stability restrictions stated below for each specification, and the
parameter bounds listed in the [parameter table](#estimation-and-random-starts)
under Estimation and Random Starts, are imposed during estimation and checked again when selecting results. Shape paths and the implied conditional variance must be finite and positive throughout the sample. The numerical shape guards are listed in the same table; no EWMA variance-path bound is imposed.[^ewma-bound]

[^ewma-bound]: An earlier version of the estimation also kept the implied variance inside a wide envelope around an exponentially weighted moving average (EWMA) of squared residuals, between $10^{-6}$ and $10^{6}$ times that average. The bound was removed after a paired estimation check found that it did not bind at converged estimates. In that check, I estimated each of the five canonical BEGE specifications with the Constant, ARX(1,1), ARX(2,1) and ARX(2,2) mean processes twice on the 2026-10-20 data vintage, once with the EWMA bound and once without it, both times from the same 50,000 random starting points. Without the bound, every converged candidate estimate stayed inside the former envelope; the best estimates keep their implied variance at least 80,000 times above its lower limit and at least 6,700 times below its upper limit. In 15 of the 20 combinations of specification and mean process, the best log likelihoods of the two runs agree within $10^{-4}$; in the other five they differ by at most $0.20$, in both directions, and each of those best estimates lies inside the former bound, so the differences reflect the random search rather than the bound.

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


### BEGE-EGARCH (extension)

Alongside the five canonical specifications and the Constant-p and Constant-n restrictions of Full BEGE, the `BEGE_EGARCH/` folder
contains an explicitly requested extension of Full BEGE in which the shape
processes follow coupled EGARCH-type recursions on the log shapes,

$$
\ln p_t = \omega_p + \beta_p \ln p_{t-1}
  + \alpha_p\left(|z_{t-1}| - \sqrt{2/\pi}\right) + \gamma_p z_{t-1},
$$

with $z_{t-1} = u_{t-1}/\sigma_{t-1}$ (and analogously for $\ln n_t$), and
stability restrictions $\beta_p<1$, $\beta_n<1$. See
`BEGE_EGARCH/SPECIFICATION.md` for the full specification and the mapping to
the equivalent EGARCH recursion on the component volatilities
$\sigma_p\sqrt{p_t}$ and $\sigma_n\sqrt{n_t}$; its parameter bounds,
starting-value draws, and shape initialization are listed in the parameter
table below. All shared BEGE
settings (mean menu, density evaluator, random-search protocol, selection
rules) are inherited unchanged.


## Estimation and Random Starts

For each BEGE specification and mean process, I randomly sample 10,000
starting points and run the optimizer from each of them: 10 independent jobs, each with 40 draws and 25 optimizer starts per draw. Each draw saves its best result. Thus a report covering six mean processes contains 2,400 saved draw winners from 60,000 optimizer starts.

The table below summarizes every BEGE specification. The second column gives
the parameter bounds and restrictions, and the third column gives the
distribution from which each optimizer start draws the parameter. Parameters
are drawn independently, except where a draw is stated given another drawn
parameter. Constant BEGE is maximized by L-BFGS-B subject to the bounds.
Inflation/Deflation BEGE is also maximized by L-BFGS-B, with its stability
restrictions imposed through a likelihood penalty. The other specifications
are maximized by SLSQP, with their stability restrictions as inequality
constraints. Every stability restriction holds with margin $\delta=10^{-6}$,
and every random start satisfies the bounds and stability restrictions before
optimization begins. The mean-process stationarity conditions are checked when
the best estimate is selected; they are not imposed during optimization.

**BEGE parameter bounds, random-start draws, and shape initialization**

| Parameter | Bounds and restrictions | Random-start draw |
|---|---|---|
| **Mean process (all specifications)** | | |
| $c$ | $[\min_t\pi_t,\ \max_t\pi_t]$ | $U[\hat\theta_j\pm2\,\widehat{SE}_j]$ |
| $\rho_1$ in ARX(1,·) | $[-1,1]$; at selection, $\lvert\rho_1\rvert<1$ | $U[\hat\theta_j\pm2\,\widehat{SE}_j]$ |
| $\rho_1,\rho_2$ in ARX(2,·) | $\rho_1\in[-2,2]$, $\rho_2\in[-1,1]$; at selection, $\rho_1+\rho_2<1$, $\rho_2-\rho_1<1$, $\rho_2>-1$ | $U[\hat\theta_j\pm2\,\widehat{SE}_j]$ |
| $\phi_1,\phi_2$ | $[-10,10]$ | $U[\hat\theta_j\pm2\,\widehat{SE}_j]$ |
| **Scales and shape levels** | | |
| $\sigma_p,\sigma_n$ (all specifications) | $[10^{-5},2]$ | $U[10^{-5},2]$ |
| $\bar p,\bar n$ (Constant BEGE) | $[0,10]$ | $U[0,10]$ |
| $p_0,n_0$ (Symmetric, BadGood, Inflation/Deflation, Full BEGE) | $[0,10]$ | $U[0,10]$ |
| $p_0,n_0$ (Constant-p, Constant-n) | $[0,10]$; $p_0,n_0>\delta$ | $U[10^{-5},10]$ |
| $\omega_p,\omega_n$ (BEGE-EGARCH) | $[-5,5]$ | $\omega_j=(1-\beta_j)\ln\bar s_j$ given $\beta_j$, with initial shape $\bar s_j$ log-uniform on $[0.05,5]$ |
| **Symmetric BEGE** | | |
| $\rho$ | $[0,1]$ | $U[0,1-\delta]$ |
| $\phi^+,\phi^-$ | $[0,2]$; $\rho+\tfrac{1}{2}(\phi^++\phi^-)<1-\delta$ | $(\tfrac{1}{2}\phi^+,\tfrac{1}{2}\phi^-)$ uniform on $\{a,b\ge0,\ a+b<1-\delta-\rho\}$ given $\rho$ |
| **BadGood BEGE ($j=p,n$)** | | |
| $\rho_j$ | $[0,1]$ | $U[0,1-\delta]$ |
| $\phi_j$ | $[0,2]$; $\rho_j+\phi_j<1-\delta$ | $U[0,\ 1-\delta-\rho_j]$ given $\rho_j$ |
| **Inflation/Deflation BEGE** | | |
| $\rho_p,\rho_n$ | $[0,1]$ | $U[0,1-\delta]$ |
| $\phi_p^+$ | $[0,2]$; $\rho_p+\tfrac{1}{2}\phi_p^+<1-\delta$ | $U[0,\ 2(1-\delta-\rho_p)]$ given $\rho_p$ |
| $\phi_n^-$ | $[0,2]$; $\rho_n+\tfrac{1}{2}\phi_n^-<1-\delta$ | $U[0,\ 2(1-\delta-\rho_n)]$ given $\rho_n$ |
| **Full BEGE ($j=p,n$), Constant-p ($j=n$), Constant-n ($j=p$)** | | |
| $\rho_j$ | $[0,1]$ | $U[0,1-\delta]$ |
| $\phi_j^+,\phi_j^-$ | $[0,2]$; $\rho_j+\tfrac{1}{2}(\phi_j^++\phi_j^-)<1-\delta$ | $(\tfrac{1}{2}\phi_j^+,\tfrac{1}{2}\phi_j^-)$ uniform on $\{a,b\ge0,\ a+b<1-\delta-\rho_j\}$ given $\rho_j$ |
| **BEGE-EGARCH ($j=p,n$)** | | |
| $\beta_j$ | $[0,1]$; $\beta_j<1-\delta$ | $U[0.2,\ 0.98]$ |
| $\alpha_j$ | $[0,2]$ | $U[0,\ 0.8]$ |
| $\gamma_j$ | $[-2,2]$ | $U[-0.5,\ 0.5]$ |
| **Shape-path initialization and numerical guards** | | |
| First state, GJR-type recursions | $p_1=p_0/(1-\rho_p-\tfrac{1}{2}(\phi_p^++\phi_p^-))$; $n_1$ likewise | --- |
| First state, BEGE-EGARCH | $\ln p_1=\omega_p/(1-\beta_p)$, $\ln n_1=\omega_n/(1-\beta_n)$ | --- |
| Constant shapes | Constant BEGE: $p_t=\bar p$, $n_t=\bar n$; Constant-p: $p_t=p_0$; Constant-n: $n_t=n_0$ | --- |
| Shape floor | $p_t,n_t\ge10^{-4}$, including the first state | --- |
| Log-shape cap | $\ln p_t,\ln n_t\le50$ in BEGE-EGARCH | --- |

Notes:

- $U[a,b]$ is the uniform distribution on $[a,b]$; *log-uniform* draws are uniform in the logarithm. $\hat\theta_j$ and $\widehat{SE}_j$ are the OLS estimate of mean parameter $j$ and its HAC(12) standard error, computed on the estimation sample at run time. On the quarterly and monthly samples, every interval $\hat\theta_j\pm2\,\widehat{SE}_j$ lies inside the parameter's bounds and the AR stationarity region.
- For a pair of loadings drawn *uniform on* $\{a,b\ge0,\ a+b<1-\delta-\rho_j\}$, the two halved loadings are drawn independently from $U[0,1-\delta-\rho_j]$, and the start is redrawn until their sum lies below $1-\delta-\rho_j$. This leaves $\rho_j$ uniform.
- The GJR-type first state is written in Full BEGE notation; the other specifications impose their restrictions on it, for example $p_1=p_0/(1-\rho_p-\phi_p)$ in BadGood BEGE. It is a backcast convention; it need not equal the exact unconditional shape mean of the coupled, asymmetric process. The BEGE-EGARCH first state sets the shock terms to zero. Its centering constant $\sqrt{2/\pi}$ is a Gaussian reference value that need not equal $E\lvert z_t\rvert$ for BEGE innovations, so this first state is not in general the unconditional log-shape mean.
- If the first state or a later recursion step gives a shape below $10^{-4}$, the shape is set to $10^{-4}$. The log-shape cap prevents overflow in $p_t=\exp(\ln p_t)$.

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


Each report gives empirical 5%, median, and 95% quantiles for $p_t$, $n_t$, $\sigma_t^2$, $\kappa_{3,t}$, and $\kappa_{4,t}$, as defined above.

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
observed-information or inverse-OPG fallback, where OPG is the outer product of observation-level score vectors.

Every reported parameter receives a standard error. When an estimate lies on a
bound of its admissible range, or its sandwich standard error is not
numerically identified, the report gives a rough standard error marked with
$\dagger$, because the usual normal approximation does not apply there. The
rough value is the first of the following that is available: (1) for an
estimate on a bound, the sandwich standard error itself, provided the
likelihood is finite just outside the bound, so that the two-sided Hessian
behind it is valid; (2) $1/\sqrt{\partial^2(-\ln L)/\partial\theta_i^2}$,
the curvature in that parameter alone with the other parameters held at their
estimates, computed with steps into the admissible range; (3) the
likelihood-ratio half-width, the smallest move into the admissible range that
lowers the log likelihood by $0.5$; (4) if the likelihood is flat, the distance
to the end of the admissible range. The note under the parameter table names
the method used. Standard errors below the display precision are shown as
$<0.0001$.
