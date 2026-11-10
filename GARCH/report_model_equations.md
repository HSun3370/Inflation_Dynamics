```{raw:typst}
#set page(margin: auto)
```

## Equations and Parameter Restrictions

The following specifications apply to both quarterly and monthly estimation. They are presented once here with the quarterly results. Each mean equation is paired with one volatility equation and one standardized innovation distribution.

Write inflation as

$$
\pi_t=\hat{\pi}_t+u_t,\qquad
u_t=\sigma_t z_t,\qquad
E_{t-1}(z_t)=0,\quad E_{t-1}(z_t^2)=1.
$$

Here $\hat{\pi}_t$ is expected inflation and $\sigma_t^2$ is the conditional variance. Let $\varepsilon=10^{-6}$ denote the numerical stationarity margin. The scale $\bar v$ is the mean squared residual at the initial mean estimate: OLS residuals for ARX models and $\pi_t-SPF_{t-1}$ in the time indexing used below for the Constant model.

### Mean Equations

**Constant (professional-forecast benchmark).**

$$
\hat{\pi}_{t+1}=SPF_t.
$$

**Constraints.** The forecast coefficient is fixed at one; there is no estimated intercept or autoregressive coefficient.

**ARX(1,1).**

$$
\hat{\pi}_{t+1}=c+\rho_1\pi_t+\phi_1 SPF_t.
$$

**Constraints.**

$$
c,\phi_1\in\mathbb R,\qquad |\rho_1|<1-\varepsilon.
$$

The intercept and SPF loading have no finite parameter bounds. Only the inflation-lag coefficient enters the stationarity restriction.

**ARX(2,1).**

$$
\hat{\pi}_{t+1}=c+\rho_1\pi_t+\rho_2\pi_{t-1}+\phi_1 SPF_t.
$$

**Constraints.** The intercept and SPF loading are unrestricted. The AR(2) stationarity conditions are

$$
\rho_1+\rho_2<1,\qquad
\rho_2-\rho_1<1,\qquad
\rho_2>-1.
$$

For reported estimates, both roots $r$ of $r^2-\rho_1r-\rho_2=0$ must additionally satisfy $|r|<1-\varepsilon$.

**ARX(2,2).**

$$
\hat{\pi}_{t+1}=c+\rho_1\pi_t+\rho_2\pi_{t-1}
+\phi_1 SPF_t+\phi_2 SPF_{t-1}.
$$

**Constraints.** The same AR(2) restrictions apply. The coefficients $c,\phi_1,\phi_2\in\mathbb R$ have no finite bounds; the SPF coefficients do not enter the AR stationarity polynomial.

### Volatility Equations

The estimated lag orders are $(p,q)\in\{(1,1),(1,2),(2,1),(2,2)\}$.

Define $u_t^+=u_t\mathbf 1(u_t>0)$ and $u_t^-=u_t\mathbf 1(u_t\leq0)$.

**GARCH$(p,q)$.**

$$
\sigma_t^2=\omega+\sum_{i=1}^{p}\alpha_i u_{t-i}^2
+\sum_{k=1}^{q}\beta_k\sigma_{t-k}^2.
$$

**Constraints.**

$$
10^{-8}\bar v\leq\omega\leq10\bar v,\qquad
0\leq\alpha_i\leq1,\qquad 0\leq\beta_k\leq1,
$$

$$
\sum_{i=1}^{p}\alpha_i+\sum_{k=1}^{q}\beta_k<1-\varepsilon.
$$

The positive intercept and nonnegative coefficients ensure positive conditional variances; the persistence restriction rules out integrated or explosive volatility.

**GJR-GARCH$(p,q)$.**

$$
\sigma_t^2=\omega
+\sum_{i=1}^{p}\alpha_i(u_{t-i}^+)^2
+\sum_{i=1}^{p}\gamma_i(u_{t-i}^-)^2
+\sum_{k=1}^{q}\beta_k\sigma_{t-k}^2.
$$

**Constraints.**

$$
10^{-8}\bar v\leq\omega\leq10\bar v,\qquad
0\leq\alpha_i\leq1,\qquad \gamma_i\geq0,\qquad
0\leq\beta_k\leq1,
$$

$$
-1\leq\gamma_i-\alpha_i\leq2,\qquad
\frac12\sum_{i=1}^{p}(\alpha_i+\gamma_i)
+\sum_{k=1}^{q}\beta_k<1-\varepsilon.
$$

Positive and negative inflation surprises have loadings $\alpha_i$ and $\gamma_i$, respectively. The bound on $\gamma_i-\alpha_i$ is the estimation bound on the equivalent indicator coefficient. The one-half weighting in the reported persistence check corresponds to the symmetric-innovation stationarity condition and is the restriction used across the estimated innovation distributions.

**EGARCH$(p,q)$.**

$$
\begin{aligned}
\ln\sigma_t^2={}&\omega
+\sum_{i=1}^{p}\alpha_i\left(|z_{t-i}|-\sqrt{2/\pi}\right)\\
&+\sum_{i=1}^{p}\gamma_i z_{t-i}
+\sum_{k=1}^{q}\beta_k\ln\sigma_{t-k}^2.
\end{aligned}
$$

**Constraints.**

$$
\ln\bar v-\ln(10^4)\leq\omega\leq\ln\bar v+\ln(10^4),
$$

$$
\alpha_i,\gamma_i\in\mathbb R,\qquad
0\leq\beta_k\leq q,\qquad
\sum_{k=1}^{q}\beta_k<1-\varepsilon.
$$

The logarithmic specification ensures positive variance. Magnitude and sign coefficients have no finite estimation bounds. The centering constant remains fixed at $\sqrt{2/\pi}$ for every innovation distribution.

### Innovation Distributions

**Normal.**

$$
z_t\sim\mathcal N(0,1).
$$

**Constraints.** Mean zero and variance one are fixed; there are no additional distributional parameters.

**Standardized Student's $t$.**

$$
z_t=\sqrt{\frac{\nu-2}{\nu}}\,\xi_t,\qquad \xi_t\sim t_\nu.
$$

**Constraints.**

$$
2.05\leq\nu\leq500.
$$

The scaling gives unit variance, while the lower bound ensures finite variance.

**Two-component Gaussian mixture.**

$$
z_t\sim p_1\mathcal N(\mu_1,\sigma_1^2)
+p_2\mathcal N(\mu_2,\sigma_2^2),
$$

where standardization implies

$$
p_2=1-p_1,\qquad \mu_2=-\frac{p_1\mu_1}{p_2},
$$

$$
\sigma_2^2=\frac{1-p_1(\sigma_1^2+\mu_1^2)-p_2\mu_2^2}{p_2}.
$$

**Constraints.**

$$
10^{-4}\leq p_1\leq1-10^{-4},\qquad -50\leq\mu_1\leq50,
$$

$$
10^{-6}\leq\sigma_1^2\leq100,\qquad \sigma_2^2>0,\qquad
\frac{\min(\sigma_1^2,\sigma_2^2)}{\max(\sigma_1^2,\sigma_2^2)}\geq0.01.
$$

The component moments give zero mixture mean and unit mixture variance. The variance-ratio condition screens out near-degenerate components. Component variances $\sigma_1^2,\sigma_2^2$ refer to standardized innovations and are distinct from the time-varying conditional variance $\sigma_t^2$.
