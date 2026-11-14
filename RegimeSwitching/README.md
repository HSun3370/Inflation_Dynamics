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

I use the hidden-state notation $(X_t,Z_t,Q_t,\mathbb P)$ of the VAR-regime
derivation. Here $X_t$ contains observed inflation lags and professional
forecasts, $Z_t$ is a hidden coordinate vector, and $Q_t$ is a column vector
of probabilities. There are $K\in\{2,3\}$ regimes. Write $e_i$ for coordinate
vector $i$, and let

$$
p_{ij}=\Pr(Z_{t+1}=e_j\mid Z_t=e_i),\qquad
\mathbb P=(p_{ij}),\qquad \mathbb P\mathbf 1_K=\mathbf 1_K.
$$

**Timing:** regime $Z_t$ generates inflation $\pi_{t+1}$. Conditional on
$Z_t=e_i$ and the observed regressors, the observation equation is

$$
\pi_{t+1}=\mathbb D_iX_t+\mathbb F_iW_{t+1},\qquad
\mathbb F_i=\sigma_i>0,\qquad W_{t+1}\sim N(0,1).
$$

In the alternative observation-date convention, this same regime is called
$s_{t+1}$: $Z_t=e_{s_{t+1}}$. The change of index leaves the likelihood
unchanged. It matters when matching probabilities to dates below.

The Gaussian shocks are independent across dates and independent of the
regime transitions. The next regime and the next inflation observation
are conditionally independent given the current regime and regressors.
For ARX(2,2),

$$
X_t=(1,\pi_t,\pi_{t-1},SPF_t,SPF_{t-1})',\qquad
\mathbb D_i=(c_i,\rho_{1,i},\rho_{2,i},\phi_{1,i},\phi_{2,i}).
$$

Thus $\mathbb D_iX_t$ is the regime-conditional mean and
$\mathbb F_i\mathbb F_i'=\sigma_i^2$ is its innovation variance.
$SPF_t$ denotes the professional forecast of $\pi_{t+1}$ available at $t$.
At the row for $\pi_{t+1}$, it is stored in `SPF`; `SPF_lag_1` supplies
$SPF_{t-1}$. The precomputed inflation-lag columns supply $\pi_t$ and
$\pi_{t-1}$. No observations are removed to construct these lags.

The mean menu is Constant, ARX(1,1), ARX(2,1), and ARX(2,2).
ARX(2,1) omits $SPF_{t-1}$; ARX(1,1) also omits $\pi_{t-1}$.
Constant fixes the mean at $SPF_t$, so
$\pi_{t+1}=SPF_t+\sigma_iW_{t+1}$ in regime $i$; it has no estimated
mean coefficients. This is a conditional inflation regression: the
observed SPF series is an input, and no separate law of motion or
likelihood for SPF is estimated. Unlike the multivariate VAR example,
there is no additional estimated state equation for $X_t$.

### Switching Selected Parameter Blocks

All switching blocks use the **same** hidden regime $Z_t$. A block that
is fixed across regimes still has estimated coefficients unless its values
are explicitly anchored, as in Constant. Sharing is an equality restriction
on parameters, not a separate estimation of each block.

| Case | Restrictions | Regime density |
| --- | --- | --- |
| Only variance switches | $\mathbb D_i=\mathbb D$; $\sigma_i$ can differ | $N(\mathbb D X_t,\sigma_i^2)$ |
| Only the mean switches | $\mathbb D_i$ can differ; $\sigma_i=\sigma$ | $N(\mathbb D_iX_t,\sigma^2)$ |
| Mean and variance switch | Both $\mathbb D_i$ and $\sigma_i$ can differ | $N(\mathbb D_iX_t,\sigma_i^2)$ |
| Selected mean coefficients and variance switch | Selected columns of $\mathbb D_i$ are shared; $\sigma_i$ can differ | $N(\mathbb D_iX_t,\sigma_i^2)$ |

For example, switching the AR block while sharing the SPF block in
ARX(1,1) gives

$$
\pi_{t+1}=c_i+\rho_{1,i}\pi_t+\phi_1SPF_t+\sigma_iW_{t+1},
\qquad Z_t=e_i.
$$

The coefficient $\phi_1$ is a single unknown, used in every regime. The
likelihood estimates it jointly with all $c_i,\rho_{1,i},\sigma_i$ and
transition probabilities. Sharing the entire mean instead gives
$c+\rho_1\pi_t+\phi_1SPF_t$ in every regime; the regimes then differ only
in their innovation variances. Independent mean and variance chains would
be a different model with a joint state space; they are not used here.

The **estimated menu always switches variance**. With two regimes, it
switches all mean coefficients, the AR block (including the intercept)
with shared SPF coefficients, or no mean coefficients. With three regimes,
it uses the latter two configurations. Constant contributes one
variance-only specification for each regime count. This gives
$3\times5+2=17$ specifications at each frequency. The mean-only row above
explains the general restriction; it is not an additional estimated model.

## Filtering and the Likelihood

Let $\mathcal F_t$ denote the observation history through $t$ and the
regressors available for forecasting $\pi_{t+1}$. All probabilities below
are evaluated at a fixed parameter vector $\theta$, which is suppressed in
the notation. Following the derivation, define

$$
Q_t=E(Z_t\mid\mathcal F_t),\qquad
\psi_{t+1}=\operatorname{vec}\{\psi_i(\pi_{t+1},X_t)\}.
$$

The regime-conditional Gaussian density is

$$
\psi_i(y^*,X_t)=\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\!\left[-\frac{(y^*-\mathbb D_iX_t)^2}{2\sigma_i^2}\right].
$$

Here $y^*$ is a possible realization of inflation itself, not a further
difference of inflation. The density is evaluated at $y^*=\pi_{t+1}$.
$Q_t$ is the probability of the regime generating this observation,
**before** observing it. Four steps give the recursion.

**1. Condition on the hidden regime.** Given $Z_t=e_i$, the joint density
of $Z_{t+1}=e_j$ and $\pi_{t+1}=y^*$ is

$$
p_{ij}\psi_i(y^*,X_t).
$$

**2. Average over the hidden regime.** Multiply by $Q_{i,t}$ and sum over
$i$. The column vector of joint densities, indexed by next regime $j$, is

$$
\mathbb P'\operatorname{diag}(Q_t)
\operatorname{vec}\{\psi_i(y^*,X_t)\}.
$$

**3. Sum over the next regime.** Since rows of $\mathbb P$ sum to one,
the predictive density of the observed inflation value is

$$
f_{t+1}=\mathbf 1_K'\mathbb P'\operatorname{diag}(Q_t)\psi_{t+1}
       =Q_t'\psi_{t+1}.
$$

This is a mixture of Gaussian densities. Its log is the contribution of
observation $t+1$ to the sample likelihood.

**4. Apply Bayes' rule and propagate.** First update beliefs about the
regime that generated the observation, then move to the next regime:

$$
R_t:=E(Z_t\mid\mathcal F_{t+1})
=\frac{\operatorname{diag}(Q_t)\psi_{t+1}}{f_{t+1}},\qquad
Q_{t+1}=\mathbb P'R_t
=\frac{\mathbb P'\operatorname{diag}(Q_t)\psi_{t+1}}
       {Q_t'\psi_{t+1}}.
$$

$R_t$ and $Q_{t+1}$ use the same observations but concern different
hidden dates. Keeping this intermediate vector avoids confusing the
updated regime probability with the prediction for the next regime.
Initialize with the unconditional distribution and accumulate the likelihood:

$$
Q_0=\bar Q(\mathbb P),\qquad
\bar Q=\mathbb P'\bar Q,\qquad\mathbf 1_K'\bar Q=1,
\qquad
\ell(\theta)=\sum_{t=0}^{T-1}\log(Q_t'\psi_{t+1}).
$$

The initial distribution is recomputed whenever $\mathbb P$ changes.
In implementation the likelihood is evaluated in log space to avoid
underflow in small Gaussian densities.

### How Partial Switching Changes the Update

For two regimes, the posterior odds after observing inflation are

$$
\frac{R_{1,t}}{R_{2,t}}
=\frac{Q_{1,t}}{Q_{2,t}}
 \frac{\psi_1(\pi_{t+1},X_t)}{\psi_2(\pi_{t+1},X_t)}.
$$

Only the density ratio changes when a parameter block is shared.
With a common mean $m_t=\mathbb D X_t$, write $r_{t+1}=\pi_{t+1}-m_t$.
Variance-only switching gives

$$
\log\frac{R_{1,t}}{R_{2,t}}
=\log\frac{Q_{1,t}}{Q_{2,t}}+\log\frac{\sigma_2}{\sigma_1}
-\frac{r_{t+1}^2}{2}
 \left(\frac{1}{\sigma_1^2}-\frac{1}{\sigma_2^2}\right).
$$

Large surprises favor the high-variance regime; small surprises favor
the low-variance regime. In the illustrative mean-only case, with
$m_{i,t}=\mathbb D_iX_t$ and a common variance,

$$
\log\frac{R_{1,t}}{R_{2,t}}
=\log\frac{Q_{1,t}}{Q_{2,t}}
-\frac{(\pi_{t+1}-m_{1,t})^2-(\pi_{t+1}-m_{2,t})^2}{2\sigma^2}.
$$

Observations favor the closer conditional mean. When both blocks switch,
both the residual fit and variance normalization matter. The transition
step $Q_{t+1}=\mathbb P'R_t$ is identical in all cases. If means and
variances are both identical, the densities cancel: $R_t=Q_t$, so inflation
provides no information about the regime.

### Predictive Mean and Inflation Innovation

Let $m_{i,t}=\mathbb D_iX_t$, including the anchored mean when appropriate.
The project's expected inflation and residual are

$$
\hat\pi_{t+1}=\sum_i Q_{i,t}m_{i,t},\qquad
u_{t+1}=\pi_{t+1}-\hat\pi_{t+1}.
$$

The residual inside regime $i$'s density is instead
$\pi_{t+1}-m_{i,t}=\sigma_iW_{t+1}$ conditional on $Z_t=e_i$.
These residuals coincide when the mean is shared. The predictive variance
reported in the empirical path quantiles is

$$
\operatorname{Var}(u_{t+1}\mid\mathcal F_t)
=\sum_i Q_{i,t}\sigma_i^2
 +\sum_i Q_{i,t}(m_{i,t}-\hat\pi_{t+1})^2.
$$

The first term is the average within-regime variance; the second reflects
uncertainty about the regime mean. The second vanishes for variance-only
switching. Even with a common $\sigma$, switching means can therefore
produce a time-varying predictive variance.

## Smoothing: Learning from the Full Sample

Filtering processes observations in sequence. Smoothing uses all $T$
observations to reassess earlier regimes. Define

$$
S_t:=E(Z_t\mid\mathcal F_T),\qquad 0\leq t\leq T-1.
$$

At the last observation $S_{T-1}=R_{T-1}$. To move backward, Bayes' rule gives

$$
\Pr(Z_t=e_i\mid Z_{t+1}=e_j,\mathcal F_{t+1})
=\frac{R_{i,t}p_{ij}}{Q_{j,t+1}}.
$$

Once $Z_{t+1}$ and the observed history are given, later observations add
no further information about $Z_t$ in this conditional regression model.
Average this backward probability over $S_{t+1}$ to obtain

$$
S_{i,t}=R_{i,t}\sum_j p_{ij}\frac{S_{j,t+1}}{Q_{j,t+1}},\qquad
S_t=\operatorname{diag}(R_t)\mathbb P
       \operatorname{diag}(Q_{t+1})^{-1}S_{t+1}.
$$

Run this recursion for $t=T-2,\ldots,0$. Terms for states with zero
predicted probability contribute zero and are omitted. The joint smoothed
probability of a transition is

$$
J_{ij,t}:=\Pr(Z_t=e_i,Z_{t+1}=e_j\mid\mathcal F_T)
=\frac{R_{i,t}p_{ij}}{Q_{j,t+1}}S_{j,t+1}.
$$

Its row sums are $S_{i,t}$ and column sums are $S_{j,t+1}$. The smoother
uses the same formula whether means, variances, or selected coefficients
switch. Sharing restrictions already enter $R_t$ and $Q_{t+1}$ through
the Gaussian densities. There is one smoother for the joint regime, not
one smoother for each switching block. Regime shading in the results
uses $\arg\max_i S_{i,t}$, displayed at the date of $\pi_{t+1}$; these
are classifications using the full sample, not real-time predictions.

## Estimation with Shared Coefficients

Filtering computes $\ell(\theta)$ for a given parameter vector. Smoothing
supplies the weights used to improve an initial parameter vector before
numerically maximizing that same likelihood. Write $\theta^{\mathrm{old}}$
for the preceding iterate. The expected complete-data log likelihood,
up to constants, is

$$
\begin{aligned}
\mathcal Q(\theta\mid\theta^{\mathrm{old}})
={}&\sum_i S_{i,0}\log\bar Q_i(\mathbb P)
 +\sum_{t=0}^{T-2}\sum_{i,j}J_{ij,t}\log p_{ij}\\
&-\frac{1}{2}\sum_{t=0}^{T-1}\sum_i S_{i,t}
 \left[\log\sigma_i^2+
 \frac{(\pi_{t+1}-\mathbb{D}_iX_t)^2}{\sigma_i^2}\right].
\end{aligned}
$$

The weights $S$ and $J$ are computed at $\theta^{\mathrm{old}}$ and held
fixed during an update. The script uses the following Gaussian mean and
variance updates.

**Mean coefficients.** At the preceding innovation variances, solve

$$
\min_{\{\mathbb D_i\}}
\sum_{t=0}^{T-1}\sum_i
\frac{S_{i,t}}{(\sigma_i^2)^{\mathrm{old}}}
(\pi_{t+1}-\mathbb D_iX_t)^2
\quad\text{subject to all coefficient-sharing restrictions.}
$$

If every coefficient switches, this separates into a weighted regression
for each regime; the variance factor is constant within that regression
and cancels. If all coefficients are shared, it is one weighted regression
with observation weight $\sum_i S_{i,t}/(\sigma_i^2)^{\mathrm{old}}$.
With only some shared coefficients, all coefficients must be solved
jointly. For example, the common SPF coefficient obeys a single score
equation summed over regimes, while each switching AR coefficient has
its own regime-specific score equation. The implementation stacks these
weighted regressions with a single column for each shared coefficient;
it uses no ridge penalty. Constant has no mean update.

**Innovation variances.** Using the updated regime means,

$$
(\sigma_i^2)^{\mathrm{new}}
=\frac{\sum_{t=0}^{T-1}S_{i,t}
 (\pi_{t+1}-\mathbb D_i^{\mathrm{new}}X_t)^2}
 {\sum_{t=0}^{T-1}S_{i,t}}.
$$

For the illustrative common-variance restriction, the corresponding
update is $T^{-1}\sum_{t,i}S_{i,t}
(\pi_{t+1}-\mathbb D_i^{\mathrm{new}}X_t)^2$.
For Constant, substitute $SPF_t$ for the regime mean.

**Transition probabilities and final optimization.** If the initial
probabilities were fixed independently of $\mathbb P$, the exact
expected-count update would be

$$
p_{ij}^{\mathrm{new}}=
\frac{\sum_{t=0}^{T-2}J_{ij,t}}
     {\sum_{t=0}^{T-2}S_{i,t}}.
$$

Here the initial probabilities are instead $\bar Q(\mathbb P)$, so the
first term of $\mathcal Q$ also depends on the transition matrix. The
simple count ratio is therefore not an exact maximizer of this objective.
The existing implementation uses an expected-count transition update
only as an initializer, followed by full joint BFGS/L-BFGS optimization
of $\ell(\theta)$ with $\bar Q(\mathbb P)$ recomputed at every evaluation.
Thus the final estimates do not rely on the count ratio being an exact
stationary-initialized EM update. The procedure is maximum likelihood
estimation; it does not perform the Gibbs sampling used in the VAR example.

An independent implementation of these equations agrees with the production
likelihood, predicted probabilities $Q_t$, filtered probabilities $R_t$,
smoothed probabilities $S_t$, and transition probabilities $J_{ij,t}$ for
all 34 saved quarterly and monthly specifications. The largest absolute
likelihood difference is below $3\times10^{-14}$; the largest probability
difference is below $8\times10^{-15}$. Separate short-sample checks sum over
every possible regime path and verify the same recursions, including the
illustrative mean-only case. These checks verify the calculations; the
final-solution checks below determine which estimates enter the results.

## Identification

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

On 2026-11-12, I corrected the Gaussian EM mean update to estimate shared
and switching coefficients jointly. At each iteration it minimizes
$\sum_{t,i}S_{i,t}(\pi_{t+1}-\mathbb D_iX_t)^2/\sigma_i^2$
at the preceding variance estimates, with equality restrictions on shared
coefficients. A stacked weighted least-squares solve implements this update;
the weighted residual sums of squares then update the variances. The
statsmodels 0.14.2 initializer instead estimates shared coefficients by pooled
OLS before estimating switching coefficients. That shortcut need not maximize
the mean-block EM objective. It does not invalidate the final joint likelihood
optimization, but it can change which local optimum the search reaches.
The transition update remains an initialization step: its expected-count
formula does not account for the dependence of the stationary initial
distribution on the transition parameters. The subsequent BFGS/L-BFGS stage
maximizes the full stationary-initialized likelihood jointly.

This correction was motivated by comparison with the weighted M-step in
[`TenuousBeliefs`, `QuantMFR`](https://github.com/lphansen/TenuousBeliefs/tree/QuantMFR),
commit `f207800b7927f175839ebb74bc6c9187741afc09`. That repository estimates a
multivariate VAR with all coefficients switching; its equations cannot be
substituted directly for the inflation models with shared SPF coefficients.
The inflation implementation retains its existing model menu, precomputed
regressors, 5,000 starts, variance ordering, and final perturbation checks.
No ridge penalty or variance floor is added. The optional Student-t initializer
is unchanged. Search blocks from before this correction are not combined with
the corrected search. See [the estimation audit](audit_20261112/report.md)
for the code comparison and empirical results.

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

The [derivation-to-code audit](notation_audit/report.md) gives the exact
array/date correspondence, coefficient-sharing checks, numerical comparisons,
and the finite-sample qualification on the transition initializer.
Run `python RegimeSwitching/audit_derivation.py` to repeat the comparison
against all saved parameters; it does not re-estimate the models.

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
The best-model report also lists the 5%, median, and 95% empirical quantiles
over the effective sample of the conditional variance of the residual
$u_t=\pi_t-\mathrm{E}(\pi_t\mid\mathcal{F}_{t-1})$,
$\operatorname{Var}(u_{t+1}\mid\mathcal F_t)$ derived above, using $Q_t$
and the regime conditional means, as in the BEGE reports.

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
