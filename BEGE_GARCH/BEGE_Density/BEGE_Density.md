```{raw:typst}
#set page(margin: auto)
```

# BEGE Density Evaluation

The BEGE (Bad Environment–Good Environment) shock is defined as the difference of two centered gamma variables,
$$
u_t = \sigma_p\bigl(\Gamma(p_t,1) - p_t\bigr) - \sigma_n\bigl(\Gamma(n_t,1) - n_t\bigr),
$$
where $p_t$ and $n_t$ are the time-varying shape parameters driven by the GARCH recursion and $\sigma_p$, $\sigma_n$ are fixed scale parameters. The density $f(u_t \mid p_t, n_t, \sigma_p, \sigma_n)$ enters the log-likelihood at every observation, so both accuracy and speed of evaluation matter for estimation.

This section compares three implementations of the BEGE log density:

- the **closed-form** implementation, which evaluates the exact analytic expression involving the confluent hypergeometric function $U$;
- the **stabilized** implementation (current), which guards the closed-form expression with a saddlepoint approximation in fragile large-shape regions;
- **convolution-based numerical integration**, which is slow but serves as an independent benchmark.

The main finding is that the closed-form formula is accurate for small and moderate shape values, but becomes numerically fragile when the recursive shapes enter a transitional range where the BEGE distribution is already close to Gaussian. In that range, large log terms in the hypergeometric expression nearly cancel, and finite-precision arithmetic can produce artificial pointwise log densities that are orders of magnitude too large, which can dominate the likelihood search.

## Executive Summary

- For moderate shape values, the stabilized and closed-form implementations agree to numerical precision.
- On the fixed-shape timing tests below, the stabilized implementation is about 1.5 times faster than the closed-form implementation and about 957 times faster than 5,000-grid numerical integration at the median across the five test cases.
- Stored estimates from earlier production runs exhibited implausibly large likelihoods in a subset of dynamic BEGE specifications. For example, one ARX(2,1) BadGood row had a stored log-likelihood of 4920.7089, whereas the stabilized implementation gives −722.4342 and the saddlepoint approximation gives −722.7255.
- At the pointwise level, one anomalous row reported $\ell_t = 90.3648$ for an observation with conditional variance 42.9373. The stabilized, saddlepoint, and Fourier-inversion values all give approximately −2.8008. This confirms a numerical failure of the hypergeometric evaluation, not an economic feature of the BEGE dynamics.

## Implementation Design

The table below summarizes the design choices in the stabilized implementation relative to the closed-form implementation.

| Issue | Closed-form implementation | Stabilized implementation | Rationale |
|---|---|---|---|
| Hypergeometric evaluation | Direct `scipy.special.hyperu` with scalar `mpmath.hyperu` fallback. | Vectorized SciPy where numerically stable; selective high-precision `mpmath` fallback; log-domain asymptotic fallback when `hyperu` is nonfinite. | Avoids the per-observation cost of high-precision arithmetic while preserving stable exact values. |
| Large-shape transition region | Closed-form branch until a hard maximum-shape cutoff. | Guards exact values once $p_t+n_t \geq 40$; smooth log-sum-exp blend between total shape 50 and 80; replaces exact values when exact and saddlepoint disagree by more than 2 log units. | Numerical failures are driven by cancellation in total shape and can occur before a single-shape cutoff is reached. |
| Near-normal limit | No explicit Gaussian-limit rule. | Substitutes a Gaussian density when total shape exceeds 500 and standardized skewness and excess kurtosis are both below 0.03. | Gamma shocks converge analytically to a normal law under variance-preserving shape rescaling. |
| Density continuity | Hard branch switches can create likelihood discontinuities. | Log-sum-exp blending in the transition band. | Keeps the likelihood smoother for numerical optimization. |
| Failure diagnostics | High likelihoods can enter stored results silently. | The collector recomputes stored estimates with the stabilized density and writes per-path and per-layer diagnostics. | Prevents stale or unstable density values from determining best models. |

The guarding thresholds are deliberately conservative: exact hypergeometric values are retained when they agree with the saddlepoint approximation, and large-shape exact values are replaced by the saddlepoint approximation only when the two disagree beyond the stated tolerance.

The accuracy and speed comparisons below are performed on the ARX(1,1) residuals from the canonical effective sample (215 observations). As a reference, the Gaussian OLS log-likelihood is −207.381. The residuals are mean-zero with a standard deviation of 0.636326, skewness of −1.047441, and excess kurtosis of 8.210174.

## Shape Parameter Range

The ranges below use eligible ARX(1,1) rows from the BadGood BEGE results. For each stored parameter vector the recursive shape paths are recomputed on the common ARX(1,1) residuals; the density comparison holds these shape values constant across time.

| Model | Rows | $p$ q05 | $p$ median | $p$ q95 | $n$ q05 | $n$ median | $n$ q95 | $\sigma_p$ median | $\sigma_n$ median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BadGood | 994 | 1.4285 | 2.8130 | 6.4330 | 0.0714 | 0.3320 | 2.8342 | 0.2355 | 0.5555 |

## Fixed-Shape Parameter Sets

The log-likelihood and timing comparison uses five representative parameter sets drawn from the BadGood quantiles. In each case $\sigma_p$ and $\sigma_n$ are held at their median values.

| Set | $p$ | $n$ | $\sigma_p$ | $\sigma_n$ |
| --- | ---: | ---: | ---: | ---: |
| $p$ at q05, $n$ at median | 1.428538 | 0.331970 | 0.235491 | 0.555532 |
| $p$ at median, $n$ at median | 2.813004 | 0.331970 | 0.235491 | 0.555532 |
| $p$ at q95, $n$ at median | 6.433015 | 0.331970 | 0.235491 | 0.555532 |
| $p$ at median, $n$ at q05 | 2.813004 | 0.071355 | 0.235491 | 0.555532 |
| $p$ at median, $n$ at q95 | 2.813004 | 2.834194 | 0.235491 | 0.555532 |

## Log-Likelihood Comparison

The stabilized and closed-form implementations are compared against convolution-based numerical integration at several grid sizes.

| Parameter set | Stabilized | Closed-form | Numerical 100 | Numerical 500 | Numerical 1000 | Numerical 5000 | Numerical 10000 | Numerical 50000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $p$ q05, $n$ median | -214.162839 | -214.162839 | -223.900355 | -213.330306 | -214.116656 | -214.148282 | -214.146613 | -214.160080 |
| $p$ median, $n$ median | -188.957117 | -188.957117 | -196.801762 | -188.438308 | -189.359597 | -189.033173 | -188.947127 | -188.954400 |
| $p$ q95, $n$ median | -194.340481 | -194.340481 | -221.076698 | -196.996227 | -192.679189 | -194.464527 | -194.368904 | -194.347495 |
| $p$ median, $n$ q05 | -212.877473 | -212.877473 | -380.654962 | -214.556763 | -214.448495 | -212.976377 | -213.323667 | -212.904257 |
| $p$ median, $n$ q95 | -235.887717 | -235.887717 | -234.563988 | -235.609633 | -235.748009 | -235.859932 | -235.873999 | -235.885265 |

- Numerical integration converges toward the analytic log-likelihood as the grid grows, but convergence is slow and requires very fine grids.
- At coarse and moderate grid sizes, numerical integration overestimates the log-likelihood.

## Evaluation Speed

All times are in seconds per call on the 215-observation residual vector.

| Parameter set | Stabilized | Closed-form | Numerical 100 | Numerical 500 | Numerical 1000 | Numerical 5000 | Numerical 10000 | Numerical 50000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $p$ q05, $n$ median | 0.0003 | 0.0005 | 0.0086 | 0.0427 | 0.0883 | 0.4326 | 0.9102 | 4.5478 |
| $p$ median, $n$ median | 0.0004 | 0.0006 | 0.0075 | 0.0358 | 0.0730 | 0.3674 | 0.7613 | 3.8117 |
| $p$ q95, $n$ median | 0.0003 | 0.0005 | 0.0064 | 0.0308 | 0.0663 | 0.3258 | 0.6520 | 3.2415 |
| $p$ median, $n$ q05 | 0.0003 | 0.0005 | 0.0076 | 0.0358 | 0.0737 | 0.3567 | 0.7479 | 3.6527 |
| $p$ median, $n$ q95 | 0.0004 | 0.0006 | 0.0059 | 0.0284 | 0.0611 | 0.2900 | 0.5937 | 2.9724 |

- The stabilized implementation is modestly faster than the closed-form implementation for moderate-shape inputs (median speedup approximately 1.5 times), reflecting the avoidance of scalar `mpmath` fallback calls.
- Both analytic implementations are orders of magnitude faster than numerical integration. Relative to 5,000-grid integration, the median speedup of the stabilized implementation is approximately 957 times.

![Numerical integration convergence](results/BEGE_Density_numerical_convergence.png)

## Large-Shape Robustness

The closed-form BEGE density contains a confluent hypergeometric term that is accurate for small and moderate shape states but becomes numerically fragile as the recursive shapes grow, because large log terms in the expression nearly cancel. In that regime, limited-precision evaluation can produce artificial pointwise log densities that are orders of magnitude from their true values.

The stabilized implementation applies the following guarded evaluation rule:

- retain the exact hypergeometric expression for small-shape observations;
- substitute the saddlepoint density when total shape enters the fragile range, using a smooth log-sum-exp transition rather than a hard cutoff;
- override exact values with the saddlepoint value whenever the two disagree by more than 2 log units in the guarded region;
- use the Gaussian density when total shape exceeds 500 and both standardized skewness and excess kurtosis are below 0.03.

### Saddlepoint Approximation

The saddlepoint approximation provides an analytical, grid-free approximation to a density derived from the cumulant-generating function (CGF). For a target observation $u_t$, the method finds the unique real number $\hat{s}$ — the saddlepoint — satisfying $K'(\hat{s}) = u_t$, where $K(s) = \log \mathbb{E}[e^{su_t}]$ is the CGF of $u_t$. This is a single nonlinear equation solved per observation by Newton's method. The log density is then given by the Lugannani–Rice formula:
$$
\log f(u_t) \approx K(\hat{s}) - \hat{s}\,u_t - \tfrac{1}{2}\log\bigl\{2\pi K''(\hat{s})\bigr\}.
$$
The approximation is exact for Gaussian and gamma distributions and is highly accurate for sums of gamma shocks. For the BEGE shock, the CGF is available in closed form and its first two derivatives reduce to simple rational functions of $s$, so each saddlepoint evaluation costs a fixed small number of floating-point operations regardless of the shape values. This stands in contrast to convolution-based numerical integration, where accuracy grows slowly with grid size (compare the 100- and 50,000-point columns in the log-likelihood table above).

The saddlepoint approximation also respects the analytic normal limit of the BEGE distribution: as the gamma shapes grow while $\sigma_p^2 p_t + \sigma_n^2 n_t$ remains fixed, the standardized skewness and excess kurtosis shrink toward zero, and the saddlepoint density converges to the Gaussian density with the same conditional variance.

### Shape Tail Consistency

Holding the other parameters at the BadGood medians ($p = 2.8130$, $n = 0.3320$, $\sigma_p = 0.2355$, $\sigma_n = 0.5555$), the figure below varies either $p$ or $n$ up to 5000 and compares all three implementations. Numerical integration uses 5,000 grid points.

![Shape consistency](results/BEGE_Density_shape_consistency.png)

### Saddlepoint Threshold Experiment

To verify the guarding rule, consider the variance-preserving rescaling
$$
p_t(c) = c\,p_t, \quad n_t(c) = c\,n_t, \quad
\sigma_p(c) = \sigma_p/\sqrt{c}, \quad \sigma_n(c) = \sigma_n/\sqrt{c},
$$
which holds the conditional variance path constant while scaling the gamma shapes. The BEGE distribution should converge monotonically to a Gaussian law as $c$ increases. The experiment uses the seed 45, draw 25, ARX(2,1) path and varies $c$ from 0.1 to 2.0. The five log-likelihood columns correspond to: (i) exact high-precision mpmath branches, (ii) the closed-form implementation with the original single-variable cutoff ($\max(p_t, n_t) \geq 180$), (iii) the current stabilized implementation, (iv) the saddlepoint approximation, and (v) the Gaussian density with the same conditional variance.

| $c$ | Med. total shape | Max shape | Original cutoff obs. | Exact mpmath | Original cutoff | Stabilized | Saddlepoint | Gaussian |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1000 | 19.4893 | 15.7964 | 0 | -728.2549 | -728.2549 | -728.2549 | -727.1415 | -720.5045 |
| 0.4000 | 77.9574 | 63.1856 | 0 | -724.0184 | -724.0184 | -724.0388 | -723.9614 | -720.5045 |
| 0.8000 | 155.9147 | 126.3712 | 0 | -722.6555 | -722.6555 | -722.6530 | -722.9801 | -720.5045 |
| 1.0000 | 194.8934 | 157.9640 | 0 | -722.4380 | -722.4380 | -722.4342 | -722.7255 | -720.5045 |
| 1.1000 | 214.3827 | 173.7604 | 0 | -1187.7835 | -1187.7835 | -722.3558 | -722.6246 | -720.5045 |
| 1.1395 | 222.0811 | 180.0000 | 1 | -94.4486 | -174.4248 | -722.3280 | -722.5884 | -720.5045 |
| 1.2000 | 233.8721 | 189.5568 | 162 | 3934.4808 | -1329.2867 | -722.2885 | -722.5364 | -720.5045 |
| 1.8000 | 350.8081 | 284.3352 | 197 | 8749.6077 | -722.0183 | -722.0077 | -722.1699 | -720.5045 |
| 2.0000 | 389.7868 | 315.9280 | 200 | 10616.6236 | -721.9500 | -721.9408 | -722.0858 | -720.5045 |

![Saddlepoint threshold experiment](results/BEGE_Density_saddlepoint_threshold_experiment.png)

The table shows that the original single-variable cutoff is insufficient: exact-branch failures appear around $c = 1.10$ (total shape ≈ 214), before the $\max(p_t, n_t) = 180$ threshold is reached at $c = 1.1395$. The stored likelihood at $c = 1$ was 4920.7089, consistent with latent hypergeometric failures at the base parameter values. The implemented rule therefore conditions on total shape:

- guard exact hypergeometric values once $p_t + n_t \geq 40$;
- smooth-blend exact and saddlepoint values between total shape 50 and 80;
- replace exact values when exact and saddlepoint disagree by more than 2 log units in the guarded region;
- use the Gaussian limit only after total shape 500 and only when standardized skewness and excess kurtosis are both below 0.03.

## Diagnostic Evidence

The table below examines BadGood results from earlier production runs. The stored log-likelihood is the value written by the closed-form implementation at estimation time. The same parameter vectors are then recomputed on the same effective-sample residuals using four alternative evaluators: the stabilized density, the all-saddlepoint density, the Gaussian density with the same conditional variance path, and exact high-precision mpmath branches.

| Mean | Seed | Draw | Stored | Stabilized | Saddlepoint | Gaussian | Exact mpmath | Max shape | Med. total shape | Med. $\sigma_t^2$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ARX(2,1) | 45 | 25 | 4920.7089 | -722.4342 | -722.7255 | -720.5045 | -722.4380 | 157.9640 | 194.8934 | 135.4737 |
| Constant | 29 | 31 | 4846.4315 | -770.7119 | -770.6706 | -770.2032 | -1889.5667 | 171.9426 | 234.7388 | 225.1110 |
| ARX(2,1) | 15 | 3 | 3098.3394 | -599.0093 | -598.9692 | -598.1718 | -1186.8621 | 199.7122 | 220.0504 | 42.6849 |
| ARX(2,2) | 35 | 31 | 2892.0304 | -734.4489 | -734.2511 | -730.1723 | -807.8124 | 173.4795 | 205.7171 | 146.5852 |
| ARX(1,1) | 20 | 14 | 1330.8412 | -590.1301 | -590.0846 | -592.7498 | -584.8732 | 170.2031 | 206.8959 | 40.1486 |
| ARX(2,1) | 41 | 2 | 1113.7517 | -684.6683 | -684.6022 | -680.7040 | -684.8619 | 146.8413 | 164.9777 | 90.3592 |
| ARX(1,1) | 13 | 32 | 846.3034 | -694.3325 | -694.2827 | -692.9410 | -791.0434 | 156.0623 | 190.5299 | 104.1080 |
| Constant | 1 | 9 | 651.3535 | -561.1096 | -561.0324 | -561.6200 | -1101.1363 | 209.9112 | 191.0321 | 30.1035 |

![Stored likelihood row diagnostics](results/BEGE_Density_gigantic_loglik_row_diagnostics.png)

Across 8,000 BadGood rows in this diagnostic pass, the stored log-likelihoods contained 150 rows above −150, including 13 rows above zero. The stabilized recomputation produced zero rows above −150, with a maximum of −161.8929.

The table below reports the observations with the largest exact-versus-stabilized gaps. The Fourier-inversion column provides an additional pointwise check: it numerically inverts the BEGE characteristic function via quadrature over the frequency domain, and is independent of both the hypergeometric and convolution-based implementations.

| Mean | Seed | Draw | Obs. | $u_t$ | $p_t$ | $n_t$ | $\sigma_t^2$ | Exact mpmath $\ell_t$ | Stabilized $\ell_t$ | Saddlepoint $\ell_t$ | Gaussian $\ell_t$ | Fourier $\ell_t$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ARX(2,1) | 15 | 3 | 177 | 0.0913 | 42.9229 | 187.7988 | 42.9373 | 90.3648 | -2.8008 | -2.8008 | -2.7989 | -2.8007 |
| ARX(2,1) | 15 | 3 | 171 | 0.2476 | 42.9393 | 189.9662 | 42.9916 | 87.2647 | -2.8052 | -2.8052 | -2.8002 | -2.8051 |
| ARX(2,1) | 15 | 3 | 160 | 0.1429 | 44.3305 | 199.7122 | 44.4494 | 77.4285 | -2.8191 | -2.8191 | -2.8163 | -2.8191 |
| Constant | 1 | 9 | 185 | -0.5080 | 188.8323 | 17.0463 | 30.2154 | 32.6685 | -2.6480 | -2.6480 | -2.6274 | -2.6503 |
| Constant | 1 | 9 | 148 | 0.5209 | 180.3713 | 17.1093 | 30.2612 | 30.6176 | -2.6069 | -2.6069 | -2.6284 | -2.6090 |
| Constant | 1 | 9 | 85 | 1.0855 | 174.4991 | 17.0414 | 30.1040 | 28.9611 | -2.5957 | -2.5957 | -2.6408 | -2.5979 |

![Pointwise density diagnostics](results/BEGE_Density_gigantic_loglik_point_diagnostics.png)

A conditional variance in the range 30–44 cannot support pointwise log densities between 29 and 90; such values are numerically impossible. The stabilized, saddlepoint, and Fourier-inversion evaluations all agree at approximately −2.80, confirming that the anomalies originate from a numerical failure of the hypergeometric evaluation rather than any feature of the BEGE dynamics.

## Summary

- Convolution-based numerical integration converges to the analytic log-likelihood but is too slow for production estimation and remains sensitive to grid size; it serves as an independent convergence benchmark.
- The shape-tail consistency plot confirms agreement across a wide shape range for all implementations; the 5,000-grid numerical integration line is included for comparison.
- The saddlepoint threshold should be conditioned on total shape $p_t + n_t$ and on exact-versus-saddlepoint disagreement; a single-variable maximum-shape cap is an unreliable guard condition.
- Large-shape log-likelihood evaluation should not rely solely on the closed-form hypergeometric expression. The guarded saddlepoint and Gaussian-limit rules in the stabilized implementation prevent spurious large-likelihood values from influencing BEGE estimates.
