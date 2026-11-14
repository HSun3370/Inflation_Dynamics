
```{raw:typst}
#set page(margin: auto)
```
# GARCH Family

This section presents three GARCH-type volatility models — GARCH, GJR-GARCH, and EGARCH — each combined with one of three conditional distributions for residuals: normal, Student's $t$, and a two-component Gaussian mixture.

Let $\{u_t\}_{t=1}^{T}$ denote the residual process, and let $\mathcal{F}_{t-1}$ denote the information set available at time $t-1$. I assume
$$
u_t = \sigma_t \, z_t, \qquad z_t | \mathcal{F}_{t-1} \stackrel{\text{i.i.d.}}{\sim} D(0,1),
$$
where $\sigma_t^2 = \operatorname{Var}(u_t | \mathcal{F}_{t-1})$ is the conditional variance and $D(0,1)$ is a standardized distribution (normal, Student's $t$, or Gaussian mixture) with zero mean and unit variance. Throughout, $z_t = u_t / \sigma_t$ denotes the standardized residual and $\theta$ denotes the full parameter vector (mean process, volatility process parameters together with any distributional parameters).

## Volatility Recursion Specifications

**GARCH$(p,q)$:**
$$
\sigma_{t}^{2} \;=\; \omega \;+\; \sum_{i=1}^{p}\alpha_{i}\, u_{t-i}^{2} \;+\; \sum_{k=1}^{q}\beta_{k}\,\sigma_{t-k}^{2},
$$
with $\omega > 0$, $\alpha_i \geq 0$, $\beta_k \geq 0$, and $\sum_i \alpha_i + \sum_k \beta_k < 1$ for stationarity.



**GJR-GARCH$(p,q)$:**
$$
\sigma_{t}^{2} \;=\; \omega \;+\; \sum_{i=1}^{p}\alpha_{i}\,(u_{t-i}^{+})^{2} \;+\; \sum_{j=1}^{p}\gamma_{j}\,(u_{t-j}^{-})^{2} \;+\; \sum_{k=1}^{q}\beta_{k}\,\sigma_{t-k}^{2},
$$
with $\omega > 0$, $\alpha_i \geq 0$, $\gamma_j \geq 0$, and $\beta_k \geq 0$. The asymmetry coefficient $\gamma_j$ captures the leverage effect: when $\alpha_j > \gamma_j$, positive shocks raise future volatility more than negative shocks of equal magnitude.

Under the assumption that $z_t$ is symmetric   $$\mathbb{E}[(u_{t-i}^{+})^{2} | \mathcal{F}_{t-i-1}] = \mathbb{E}[(u_{t-i}^{-})^{2} | \mathcal{F}_{t-i-1}] = \tfrac{1}{2}\sigma_{t-i}^{2}$$ , the stationarity condition is
$$
\frac{1}{2}\sum_{i=1}^{p}\alpha_{i} \;+\; \frac{1}{2}\sum_{j=1}^{p}\gamma_{j} \;+\; \sum_{k=1}^{q}\beta_{k} \;<\; 1.
$$


**EGARCH$(p,q)$:**
$$
\ln\sigma_{t}^{2} \;=\; \omega \;+\; \sum_{i=1}^{p}\alpha_{i} \left(|z_{t-i}| - \sqrt{\tfrac{2}{\pi}}\right) \;+\; \sum_{j=1}^{p}\gamma_{j}\, z_{t-j} \;+\; \sum_{k=1}^{q}\beta_{k}\,\ln\sigma_{t-k}^{2},
$$
where $z_t = u_t / \sigma_t$ is the standardized residual. The term $\alpha_i$ captures the magnitude effect (response to the size of shocks) while $\gamma_j$ captures the sign / leverage effect. Modeling $\ln\sigma_t^2$ guarantees positivity of $\sigma_t^2$ without sign restrictions on $(\alpha_i, \gamma_j, \beta_k)$, and the recursion is covariance-stationary in $\ln\sigma_t^2$ whenever
$$
\sum_{k=1}^{q}\beta_{k} \;<\; 1.
$$

<!-- **Remark on the centering constant.** The constant $\sqrt{2/\pi}$ equals $\mathbb{E}|z_t|$ when $z_t \sim \mathcal{N}(0,1)$, so under a Gaussian innovation the term $|z_{t-i}| - \sqrt{2/\pi}$ has zero conditional mean and $\omega$ admits the clean interpretation
$$
\mathbb{E}[\ln\sigma_t^2] \;=\; \frac{\omega}{1 - \sum_{k=1}^{q}\beta_k}.
$$
Following the convention of the `arch` Python package, we retain $\sqrt{2/\pi}$ as a fixed constant in the recursion even when the innovation distribution is Student's $t$ or a Gaussian mixture. Under those distributions, $\mathbb{E}|z_t| \neq \sqrt{2/\pi}$, so the centered term $|z_{t-i}| - \sqrt{2/\pi}$ no longer has zero mean and $\omega$ loses the unconditional-mean interpretation above. This choice is observationally innocuous: replacing $\sqrt{2/\pi}$ with the distribution-specific constant $c_D = \mathbb{E}_D|z_t|$ yields an equivalent model whose fitted $\hat\alpha_i, \hat\gamma_j, \hat\beta_k$ are unchanged, with only the intercept shifted by
$$
\omega \;=\; \omega^{*} \;+\; \left(\sqrt{\tfrac{2}{\pi}} - c_D\right)\sum_{i=1}^{p}\alpha_{i}.
$$
The likelihood, the fitted $\sigma_t^2$ path, and all forecasts are identical under the two parameterizations. -->

## Estimation Initialization And Collection

The GARCH-family estimation is executed by the `arch` package (version 7.0.0), which maximizes the likelihood by SLSQP subject to its own parameter bounds and linear constraints. Each model combination (distribution, mean process, volatility family, and order) uses **5,000 optimizer starts**. Start 1 is the `arch` default start; for the Gaussian mixture it is the Student's $t$ estimates with $(p_1,\mu_1,\sigma_1^2)=(0.3,0.1,1)$ when that vector is feasible, and the `arch` default otherwise. The other 4,999 starts are random draws.

Every random draw lies inside `arch`'s feasible set and inside the stationary region, where volatility persistence and the inflation-lag AR roots are below $1-10^{-6}$ in modulus. The feasible-set requirement matters because `arch` silently replaces any infeasible starting vector with its default start. Within those sets the draws are as wide as possible. Parameters that `arch` bounds are drawn over their full `arch` range. Parameters that `arch` leaves unbounded (mean-process coefficients, EGARCH $\alpha_i$ and $\gamma_j$) are drawn over the finite ranges listed in the *Parameter Ranges and Random Starts* table at the beginning of each results report.

The reported estimate for a combination is the highest log likelihood among starts that pass five checks:

- the optimizer converged (SLSQP success and `arch` convergence flag 0);
- the first-order (KKT) conditions hold: the numerical gradient of the log likelihood, net of non-negative multipliers on the active bounds and constraints, is at most $0.005\,T$ in absolute value;
- the volatility stationarity condition holds with a margin of $10^{-6}$;
- the inflation-lag AR roots lie inside the unit circle;
- for the Gaussian mixture, the two component variances satisfy $\min(\sigma_1^2,\sigma_2^2)/\max(\sigma_1^2,\sigma_2^2)\ge 0.01$.

The second check is needed because, with 5,000 wide starts, SLSQP occasionally reports success at points on extremely steep ridges of the likelihood, where gradients are $10^1$--$10^6$; these points are not maxima. SLSQP also sometimes stops just short of a genuine maximum. So when a start that would become the best fails the check, SLSQP is restarted from its own solution (up to five times) before the check is applied again; ridge points do not move and stay rejected. The last check is a Hathaway-type restriction that keeps the mixture likelihood bounded. Without it, the best mixture fits put a component with variance $10^{-6}$ (the `MixNormal` lower bound) and weight of a few percent on one or two standardized residuals. If no start passes all checks, the script reports the best converged start that satisfies the first-order conditions and flags the row in the CSV `selection_status` column. The summary CSV also records, for each combination, how many starts were admissible, how many reached the reported log likelihood (within 0.01), and which parameters of the reported estimate sit on an `arch` bound. Every start's outcome is saved in `garch_family_restart_attempts.csv.gz`, which is kept out of git because of its size. Conditional-variance recursion starts use the default initialization implemented by the `arch` package.

Standard errors are robust (sandwich) and are computed only for each combination's reported estimate. Each best model in the reports also lists the 5%, median, and 95% empirical quantiles of its conditional variance path $\sigma_t^2$ over the effective sample, as in the BEGE reports. Because $z_t$ has unit variance, $\sigma_t^2$ is the conditional variance of $u_t$ for all three distributions. For GJR-GARCH, `arch` estimates the indicator form $\sigma_t^2=\omega+\alpha_i u_{t-i}^2+(\gamma_i-\alpha_i)u_{t-i}^2\mathbf 1(u_{t-i}<0)+\dots$. The reports therefore add the negative-shock coefficient $\gamma_i=\alpha_i+(\gamma_i-\alpha_i)$ of the specification above. Its standard error comes from the delta method on the same covariance matrix, including the covariance term:
$$
\operatorname{Var}(\hat\gamma_i)=\operatorname{Var}(\hat\alpha_i)+\operatorname{Var}(\widehat{\gamma_i-\alpha_i})+2\operatorname{Cov}(\hat\alpha_i,\widehat{\gamma_i-\alpha_i}).
$$

### Running the search on the CBS Grid

```
bash GARCH/submit_garch_grid.sh submit    # one job per combination + a held collector per frequency
bash GARCH/submit_garch_grid.sh status    # finished combinations and queue state
bash GARCH/submit_garch_grid.sh collect   # collect again by hand if needed
```

Each combination runs as a single-core job with 1 GB of memory. At 5,000 starts a job takes about 10 minutes with normal innovations, 15--20 minutes with Student's $t$, and about an hour with the Gaussian mixture. Per-combination results, SGE logs, and job scripts go to `/scratch/hs3370/Inflation_Dynamics/GARCH/<run>/`. The collector then writes `results_garch_distributions/` (quarterly) and `results_garch_distributions_monthly/` (monthly). Without the Grid, `python estimate_garch_results.py` runs all combinations in sequence.

`python estimate_garch_results.py --report-only [--data-path ...] [--output-dir ...]` rebuilds the conditional-variance quantile columns (`sigma2_q05`, `sigma2_median`, `sigma2_q95`) of the summary CSV and the Markdown report from the saved CSVs without re-estimating. The variance path is recomputed at the saved estimates, and the script stops unless they reproduce the reported log likelihood.


## Log-Likelihood Specifications

For each candidate distribution we give (i) the standardized density of $z_t$, (ii) the conditional density of $u_t | \mathcal{F}_{t-1}$ obtained via the change of variables $u_t = \sigma_t z_t$ (which contributes a Jacobian factor $1/\sigma_t$), (iii) the per-observation log-likelihood $\ell_t(\theta)$, and (iv) the sample log-likelihood
$$
\ln L(\theta) \;=\; \sum_{t=1}^{T} \ell_t(\theta).
$$

### Normal distribution

The standardized innovation satisfies $z_t | \mathcal{F}_{t-1} \sim \mathcal{N}(0,1)$ with density
$$
\phi(z) \;=\; \frac{1}{\sqrt{2\pi}}\,\exp \left\{-\frac{z^{2}}{2}\right\}.
$$
By the change of variables $u_t = \sigma_t z_t$, the conditional density of $u_t$ is
$$
f(u_t | \mathcal{F}_{t-1};\,\theta) \;=\; \frac{1}{\sigma_t}\,\phi \left(\frac{u_t}{\sigma_t}\right) \;=\; \frac{1}{\sqrt{2\pi\,\sigma_t^{2}}}\,\exp \left\{-\frac{u_t^{2}}{2\,\sigma_t^{2}}\right\}.
$$
The per-observation log-likelihood is
$$
\ell_t(\theta) \;=\; -\frac{1}{2} \left(\ln 2\pi \;+\; \ln\sigma_t^{2} \;+\; \frac{u_t^{2}}{\sigma_t^{2}}\right),
$$
and the sample log-likelihood is
$$
\ln L(\theta) \;=\; -\frac{1}{2}\sum_{t=1}^{T} \left(\ln 2\pi \;+\; \ln\sigma_t^{2} \;+\; \frac{u_t^{2}}{\sigma_t^{2}}\right).
$$

### Student's $t$ distribution

Let $\nu > 2$ denote the degrees-of-freedom parameter. The standardized Student's $t$ density (rescaled to unit variance) is
$$
f_\nu(z) \;=\; \frac{\Gamma \left(\tfrac{\nu+1}{2}\right)}{\Gamma \left(\tfrac{\nu}{2}\right)\sqrt{\pi(\nu-2)}}\left(1 + \frac{z^{2}}{\nu-2}\right)^{-\frac{\nu+1}{2}},
$$
where $\Gamma(\cdot)$ is the gamma function. The restriction $\nu > 2$ ensures finite variance. The conditional density of $u_t$ is
$$
f(u_t | \mathcal{F}_{t-1};\,\theta) \;=\; \frac{1}{\sigma_t}\, f_\nu \left(\frac{u_t}{\sigma_t}\right) \;=\; \frac{\Gamma \left(\tfrac{\nu+1}{2}\right)}{\Gamma \left(\tfrac{\nu}{2}\right)\sqrt{\pi(\nu-2)\,\sigma_t^{2}}}\left(1 + \frac{u_t^{2}}{(\nu-2)\,\sigma_t^{2}}\right)^{-\frac{\nu+1}{2}}.
$$
The per-observation log-likelihood is
$$
\ell_t(\theta) \;=\; \ln\Gamma \left(\tfrac{\nu+1}{2}\right) - \ln\Gamma \left(\tfrac{\nu}{2}\right) - \tfrac{1}{2}\ln \big(\pi(\nu-2)\sigma_t^{2}\big) - \tfrac{\nu+1}{2}\ln \left(1 + \frac{u_t^{2}}{(\nu-2)\,\sigma_t^{2}}\right),
$$
and the sample log-likelihood is
$$
\ln L(\theta) \;=\; T \left[\ln\Gamma \left(\tfrac{\nu+1}{2}\right) - \ln\Gamma \left(\tfrac{\nu}{2}\right) - \tfrac{1}{2}\ln \big(\pi(\nu-2)\big)\right] - \tfrac{1}{2}\sum_{t=1}^{T}\ln\sigma_t^{2} - \tfrac{\nu+1}{2}\sum_{t=1}^{T}\ln \left(1 + \frac{u_t^{2}}{(\nu-2)\,\sigma_t^{2}}\right).
$$

### Gaussian mixture distribution

The standardized innovation follows a two-component Gaussian mixture. The free parameters are $(p_1, \mu_1, \sigma_1^{2})$:
$$
f_{\mathrm{mix}}(z) \;=\; p_1\,\varphi(z;\,\mu_1,\sigma_1^{2}) \;+\; p_2\,\varphi(z;\,\mu_2,\sigma_2^{2}), \qquad \varphi(z;\mu,s^{2}) \;=\; \frac{1}{\sqrt{2\pi s^{2}}}\exp \left\{-\frac{(z-\mu)^{2}}{2 s^{2}}\right\}.
$$
To enforce $\mathbb{E}[z_t]=0$ and $\operatorname{Var}(z_t)=1$, the remaining parameters are determined by
$$
p_2 = 1 - p_1, \qquad \mu_2 = -\frac{p_1\mu_1}{p_2}, \qquad \sigma_2^{2} = \frac{1 - p_1(\sigma_1^{2} + \mu_1^{2}) - p_2\mu_2^{2}}{p_2}.
$$
This leaves three free parameters: $(p_1, \mu_1, \sigma_1^{2})$.

**Hierarchical representation.** The mixture admits the latent-variable form $S \sim \mathrm{Bernoulli}(p_1)$, $X_1 \sim \mathcal{N}(\mu_1,\sigma_1^{2})$, $X_2 \sim \mathcal{N}(\mu_2,\sigma_2^{2})$, with $z = X_1$ if $S=1$ and $z = X_2$ otherwise. As a consequence, moments and the CDF decompose as
$$
\mathbb{E}[z^{n}] \;=\; p_1\,\mathbb{E}[X_1^{n}] \;+\; p_2\,\mathbb{E}[X_2^{n}], \qquad F_{\mathrm{mix}}(a) \;=\; p_1\,\Phi \left(\tfrac{a-\mu_1}{\sigma_1}\right) \;+\; p_2\,\Phi \left(\tfrac{a-\mu_2}{\sigma_2}\right),
$$
where $\Phi$ is the standard normal CDF.

The conditional density of $u_t$ is
$$
f(u_t | \mathcal{F}_{t-1};\,\theta) \;=\; \frac{1}{\sigma_t}\, f_{\mathrm{mix}} \left(\frac{u_t}{\sigma_t}\right) \;=\; \frac{1}{\sigma_t}\left[\,p_1\,\varphi(z_t;\,\mu_1,\sigma_1^{2}) \;+\; p_2\,\varphi(z_t;\,\mu_2,\sigma_2^{2})\right], \qquad z_t = \frac{u_t}{\sigma_t}.
$$
The per-observation log-likelihood is
$$
\ell_t(\theta) \;=\; -\ln\sigma_t \;+\; \ln \left[\,p_1\,\varphi(z_t;\,\mu_1,\sigma_1^{2}) \;+\; p_2\,\varphi(z_t;\,\mu_2,\sigma_2^{2})\right],
$$
and the sample log-likelihood is
$$
\ln L(\theta) \;=\; -\sum_{t=1}^{T}\ln\sigma_t \;+\; \sum_{t=1}^{T}\ln \left[\,p_1\,\varphi(z_t;\,\mu_1,\sigma_1^{2}) \;+\; p_2\,\varphi(z_t;\,\mu_2,\sigma_2^{2})\right].
$$

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

## Monthly Estimation

The same estimation menu (three volatility families, six mean processes, three residual distributions, 5,000 starts per combination) is run on the monthly effective sample of the newest data vintage (see the Data Summary). `submit_garch_grid.sh` runs both frequencies by default; the sequential equivalent is:

```
python estimate_garch_results.py \
    --data-path ../DataSummary/Aggregate_CPI_inflation_Monthly.pkl \
    --output-dir results_garch_distributions_monthly
```

Monthly outputs are stored in `results_garch_distributions_monthly/`, parallel to the quarterly `results_garch_distributions/`. All model settings are identical across the two frequencies.
