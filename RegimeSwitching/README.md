```{raw:typst}
#set page(margin: auto)
```

# Regime-Switching Inflation Models

The identification and likelihood-maximization procedure follows
[Bekaert, Engstrom and Ermolov (2015)](../Bekaert2015.md), Section 4.1.2 and
Appendix B. I apply their Gaussian regime-switching likelihood, unconditional
initial regime probabilities, variance-based labeling, multiple initial
parameter vectors, and final-solution perturbation checks to the inflation
mean processes below.

## Model Specification

Let the unobserved state follow a time-homogeneous Markov chain,

$$
s_t\in\{1,\ldots,K\},\qquad
p_{ij}=\Pr(s_t=j\mid s_{t-1}=i),\qquad K\in\{2,3\}.
$$

For example, the ARX(2,2) observation equation is

$$
\pi_t=c_{s_t}+\rho_{1,s_t}\pi_{t-1}+\rho_{2,s_t}\pi_{t-2}
       +\phi_{1,s_t}SPF_{t-1}+\phi_{2,s_t}SPF_{t-2}
       +\sigma_{s_t}e_t,
\qquad e_t\stackrel{\mathrm{i.i.d.}}{\sim}N(0,1).
$$

Here $SPF_{t-1}$ is the professional forecast of $\pi_t$ available at $t-1$;
it is stored in the effective-sample dataset's `SPF` column. The `SPF_lag_1`
column supplies the preceding forecast. All lagged regressors are used
as precomputed, with no additional sample trimming.

The mean-process menu is Constant, ARX(1,1), ARX(2,1), and ARX(2,2).
ARX(2,1) omits the second SPF term, and ARX(1,1) also omits the second
inflation lag. Constant anchors expected inflation to the professional
forecast: $\pi_t=SPF_{t-1}+\sigma_{s_t}e_t$.

The parameters are grouped into three blocks:

| Block | Parameters that can switch |
|---|---|
| AR block, including the intercept | $c_k,\rho_{1,k},\rho_{2,k}$ |
| SPF block | $\phi_{1,k},\phi_{2,k}$ |
| Shock distribution | $\sigma_k^2$ |

With two regimes, the estimated configurations switch all three blocks, the
AR and variance blocks, or the variance block alone. With three regimes,
they switch the AR and variance blocks, or the variance block alone.
Constant has no estimated mean coefficients, so only its variance switches.
A block held fixed has the same parameter values in every regime.

Write $\hat\pi_t(k)$ for the mean conditional on regime $k$. The residual
used in its Gaussian density is $u_t(k)=\pi_t-\hat\pi_t(k)$. For the
anchored Constant specification this is $\pi_t-SPF_{t-1}$. Expected inflation
conditional only on observed information averages the regime means using
the predicted regime probabilities.

## Likelihood, Initialization, and Identification

Let $Y_t$ denote observations through $t$, together with the observed
regressors, and let $\theta$ collect the mean coefficients, innovation
variances, and transition probabilities. The sample log likelihood is

$$
\ell(\theta)=\sum_{t=1}^{T}\log f(y_t\mid Y_{t-1};\theta).
$$

For the ARX specifications, $y_t=\pi_t$ and
$x_t=(1,\pi_{t-1},\pi_{t-2},SPF_{t-1},SPF_{t-2})'$ with the appropriate
terms omitted. For Constant, $y_t=\pi_t-SPF_{t-1}$ and $x_t'\beta_k=0$.
Each regime contributes the conditional density

$$
f_k(y_t\mid Y_{t-1};\theta)
=\frac{1}{\sqrt{2\pi\sigma_k^2}}
 \exp\!\left[-\frac{(y_t-x_t'\beta_k)^2}{2\sigma_k^2}\right].
$$

As in Appendix B of Bekaert et al., the density conditional on observed
history sums over the possible regimes:

$$
f(y_t\mid Y_{t-1};\theta)
=\sum_{k=1}^{K}\Pr(s_t=k\mid Y_{t-1};\theta)
                    f_k(y_t\mid Y_{t-1};\theta).
$$

The Hamilton filter computes the probabilities recursively. If
$\xi_{t|t-1}(j)=\Pr(s_t=j\mid Y_{t-1})$, then

$$
\begin{aligned}
\xi_{t|t-1}(j)&=\sum_{i=1}^{K}p_{ij}\xi_{t-1|t-1}(i),\\
\xi_{t|t}(j)&=
 \frac{\xi_{t|t-1}(j)f_j(y_t\mid Y_{t-1};\theta)}
 {\sum_{k=1}^{K}\xi_{t|t-1}(k)f_k(y_t\mid Y_{t-1};\theta)}.
\end{aligned}
$$

The initial distribution $p(s_0)$ is the unconditional distribution of the
Markov chain, satisfying $\bar\xi_j=\sum_i\bar\xi_i p_{ij}$ and
$\sum_j\bar\xi_j=1$. It is recomputed from the transition matrix at each
parameter vector.

Following the identification convention in Section 4.1.2 of Bekaert et al.,
I label regimes in increasing order of their innovation variances,

$$
\sigma_1^2\leq\sigma_2^2\leq\cdots\leq\sigma_K^2.
$$

For distinct variances, regime 1 is the low-volatility regime and regime
$K$ is the high-volatility regime. This labeling removes equivalent
permutations of the state names; it does not change the likelihood.
The same permutation is applied to all switching coefficients, both rows
and columns of the transition matrix, and the regime probability paths.
The CSV files and generated tables use zero-based labels, so Regime 0 is
lowest volatility. Exact variance ties retain their original order and do
not identify a unique ordering within the tied group.

## Likelihood Maximization

I estimate each specification by numerically maximizing its likelihood.
To reduce dependence on local maxima, I run the optimizer from different
initial parameter vectors and retain the highest-likelihood admissible
solution as a provisional estimate. I then assess that solution by randomly
perturbing the estimated parameters, maximizing the likelihood again, and
checking whether the optimizer returns to the provisional maximum. If a
perturbation finds a higher admissible maximum, it becomes the provisional
estimate and is checked with a fresh batch of perturbations. These are the
multiple-start and final-solution checks described in Appendix B of
Bekaert et al.

### Numerical Implementation

Bekaert et al. do not specify the optimizer, number or distribution of
starts, EM initialization, perturbation size, or numerical return tolerance.
The following values document this implementation of their procedure.

Each specification uses 5,000 initial optimizer starts. Start 1 uses the
statsmodels defaults: equal transition probabilities $p_{ij}=1/K$, OLS-based
mean coefficients interpolated across regimes, and variances evenly spaced
from one tenth of the OLS residual variance to that variance. For Constant,
the reference variance is the variance of $\pi_t-SPF_{t-1}$.
The remaining 4,999 starts perturb the default mean coefficients and
variances:

$$
\begin{aligned}
b^{(a)}&=b^{(0)}+w_b,
& w_b&\sim N\!\left(0,[0.25\max\{1,|b^{(0)}|\}]^2\right),\\
(\sigma_k^2)^{(a)}&=\max\{10^{-6},(\sigma_k^2)^{(0)}e^{v_k}\},
& v_k&\sim N(0,0.45^2).
\end{aligned}
$$

The $10^{-6}$ value above applies only to starting vectors. It is not a
lower bound on fitted variances. Initial transition probabilities remain
$1/K$; the optimizer estimates them freely.

Up to 1,000 EM iterations improve each starting vector (raised from 50 on
2026-11-11 at the user's request; in a 40-start test on the leading quarterly
specification, more EM iterations brought more starts to the best optimum
without longer run times), after which BFGS
(even-numbered starts) or L-BFGS (odd-numbered starts) maximizes the exact
Hamilton-filter likelihood for up to 1,500 iterations. EM is used only for
initialization: the final optimization estimates all free parameters jointly.
Transition probabilities use a multinomial-logit parameterization and
variances use $\sigma_k^2=\zeta_k^2$; mean coefficients are unrestricted
during optimization.

After combining all search blocks for a specification, I perturb the
provisional estimate in unrestricted optimization coordinates. If $a_j$ is
one such coordinate, each verification start is

$$
a_j^*=\hat a_j+0.05\max\{1,|\hat a_j|\}z_j,
\qquad z_j\stackrel{\mathrm{i.i.d.}}{\sim}N(0,1).
$$

All free coordinates, including transition logits, are perturbed. Each
batch has 20 refits, alternating L-BFGS and BFGS, with no EM initialization
and at most 1,500 optimizer iterations. L-BFGS uses `pgtol=1e-7` and
`factr=1e4`; BFGS uses `gtol=1e-7`. A return requires optimizer convergence,
all collection checks, and an absolute log-likelihood difference no greater
than $10^{-4}$. Parameter differences are recorded after variance ordering
as an additional diagnostic. Every refit in the final batch must return.
If a higher admissible likelihood is found, verification restarts around
that solution, for at most three batches. An unverified solution is flagged
and excluded from the reported model tables.

These checks establish stability under the specified local perturbations;
they do not prove global maximization. Each initial start and verification
refit has a reproducible seed. Verification occurs once per collected
specification, so changing the Grid block size does not change the procedure.

## Parameter Validity and Collection

The reported estimate must have successful optimizer convergence and a
finite likelihood. Innovation variances must be finite and strictly
positive. Transition probabilities must be finite, belong to $[0,1]$, and
sum to one for every origin regime, allowing numerical roundoff.
Following the user's decision on 2026-11-10 to use the paper's stated
procedure, I impose neither the former $10^{-4}$ transition-probability
cutoff nor the former variance floor
$\max\{10^{-6},10^{-3}\widehat{\operatorname{Var}}(y)\}$.

Without a variance floor, the Gaussian likelihood is unbounded: when a
regime's variance approaches zero, that regime can fit a few observations
exactly. The 2026-11-11 search found such degenerate optima in several
switching-AR specifications, with a regime variance between $10^{-31}$ and
$10^{-4}$. The user decided on 2026-11-12 to keep the floor removed and to
rely on the final-solution verification to reject degenerate optima. They
fail it because the perturbation refits do not return to them. If the saved
parameters of an optimum do not reproduce its saved likelihood to $10^{-6}$,
it is recorded as not reproducible and is not verified. Only the selected
optimum of each specification is stored, so a specification whose selected
optimum fails verification is reported as having no verified estimate, with
its likelihood, smallest regime variance $\min_k\sigma_k^2$, and the reason.
It is not replaced by a lower optimum.

For the inflation ARX specifications, collection also retains the project's
mean-process stationarity condition, checked separately in every regime.
This is an assumption of the inflation application; the paper's benchmark
return equation has no AR coefficients. Only inflation-lag coefficients
enter the check. With $\delta=10^{-6}$, ARX(1,1) requires
$|\rho_{1,k}|<1-\delta$. ARX(2,1) and ARX(2,2) require

$$
\rho_{1,k}+\rho_{2,k}<1-\delta,\qquad
\rho_{2,k}-\rho_{1,k}<1-\delta,\qquad
\rho_{2,k}>-1+\delta.
$$

The SPF loadings are excluded from the AR polynomial. Constant has no AR
coefficients. Final-solution stability must also be verified before a
selected estimate enters the reported table. Within a specification,
selection is by log likelihood; the best-model page selects across
specifications by AIC and also reports BIC.

Only Normal specifications enter the main tables. The optional
`--include-student-t` robustness exercise retains its documented
$2.05<\nu_k<200$ estimation transform and $2.05+10^{-6}<\nu_k<80$
collection restriction; it is not part of the paper's Gaussian procedure.

## Data Frequency and Running the Search

Quarterly estimation uses the newest effective-sample pickle,
`DataSummary/Aggregate_CPI_inflation_Quarterly.pkl`: 1969Q2--2026Q3,
230 observations, vintage 2026-10-20. The same procedure and menu are used
for the monthly effective sample, 1969M2--2026M9, 692 observations.
The shared data construction, including interpolation of October 2025 CPI,
is documented in the Data Summary.

Production runs use the CBS Grid:

```bash
bash RegimeSwitching/submit_regime_switching_grid.sh submit
```

Each frequency has one job per block of 1,000 starts for each specification,
followed by a collector that runs the final-solution checks and writes
`results/` (quarterly) or `results_monthly/` (monthly). The 20 verification
refits per batch are additional to the 5,000 initial starts. All search
blocks must use the current methodology; old blocks selected under the
removed cutoffs cannot be reused as a complete search of the expanded set.

Saved provisional estimates can be relabeled and locally verified with:

```bash
python RegimeSwitching/estimate_regime_switching_models.py --verify-saved
python RegimeSwitching/estimate_regime_switching_models.py --verify-saved \
    --data-path DataSummary/Aggregate_CPI_inflation_Monthly.pkl \
    --output-dir RegimeSwitching/results_monthly
```

This checks the saved sample window, observation count, and recomputed
likelihood before verification. It does not repeat the initial search.
Reports identify saved searches that used the former cutoffs: those searches
must be rerun before claiming likelihood-best selection over the expanded
admissible set. `regime_switching_stability_attempts.csv` records every
verification refit, including failed or non-returning attempts.
The initial-search attempt log retains the optimizer's original state indices
in failure diagnostics; selected parameters and report probability paths use
the common variance ordering.
`--report-only` rebuilds reports and variance labels without performing
optimization or inventing a missing verification result.

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

For sequential monthly estimation:

```bash
python RegimeSwitching/estimate_regime_switching_models.py \
    --data-path DataSummary/Aggregate_CPI_inflation_Monthly.pkl \
    --output-dir RegimeSwitching/results_monthly
```
