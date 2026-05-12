
```{raw:typst}
#set page(margin: auto)
```
# GARCH Family

This section presents three GARCH-type volatility models — GARCH, GJR-GARCH, and EGARCH — each combined with one of three conditional distributions for residuals: normal, Student's $t$, and a finite mixture of normals.

Let $\{u_t\}_{t=1}^{T}$ denote the residual process, and let $\mathcal{F}_{t-1}$ denote the information set available at time $t-1$. We assume
$$xw
u_t = \sigma_t \, z_t, \qquad z_t | \mathcal{F}_{t-1} \stackrel{\text{i.i.d.}}{\sim} D(0,1),
$$
where $\sigma_t^2 = \operatorname{Var}(u_t | \mathcal{F}_{t-1})$ is the conditional variance and $D(0,1)$ is a standardized distribution (normal, Student's $t$, or mixture of normals) with zero mean and unit variance. Throughout, $z_t = u_t / \sigma_t$ denotes the standardized residual and $\theta$ denotes the full parameter vector (mean process, volatility process parameters together with any distributional parameters).

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
\ln\sigma_{t}^{2} \;=\; \omega \;+\; \sum_{i=1}^{p}\alpha_{i} \left(\,|z_{t-i}| - \mathbb{E}|z_{t-i}|\,\right) \;+\; \sum_{j=1}^{p}\gamma_{j}\, z_{t-j} \;+\; \sum_{k=1}^{q}\beta_{k}\,\ln\sigma_{t-k}^{2},
$$
where $z_t = u_t / \sigma_t$ is the standardized residual. The term $\alpha_i$ captures the magnitude effect (response to the size of shocks), while $\gamma_j$ captures the sign / leverage effect. Modeling $\ln\sigma_t^2$ guarantees positivity of $\sigma_t^2$ without sign restrictions on $(\alpha_i, \gamma_j, \beta_k)$, and the recursion is covariance-stationary in $\ln\sigma_t^2$ whenever
$$
\sum_{k=1}^{q}\beta_{k} \;<\; 1.
$$

The constant $\mathbb{E}|z_t|$ depends on the assumed distribution of the standardized innovation:

- **Standard normal**, $z_t \sim \mathcal{N}(0,1)$:
$$
\mathbb{E}|z_t| \;=\; \sqrt{\tfrac{2}{\pi}}.
$$

- **Standardized Student's $t$** with $\nu > 2$ degrees of freedom (rescaled to unit variance):
$$
\mathbb{E}|z_t| \;=\; \frac{2\,\sqrt{\nu-2}\;\Gamma \left(\tfrac{\nu+1}{2}\right)}{(\nu-1)\,\sqrt{\pi}\;\Gamma \left(\tfrac{\nu}{2}\right)}.
$$
As $\nu \to \infty$ this collapses to the normal value $\sqrt{2/\pi}$; for finite $\nu$ it is strictly larger, reflecting the heavier tails.

- **Mixture of two normals** with parameters $(p_1, \mu_1, \sigma_1^{2})$ and induced $(p_2, \mu_2, \sigma_2^{2})$ (as defined in the likelihood section): using $\mathbb{E}|X| = \mu\,[2\Phi(\mu/s) - 1] + 2s\,\varphi(\mu/s)$ for $X \sim \mathcal{N}(\mu, s^{2})$ (a folded-normal mean), where $\Phi$ and $\varphi$ are the standard normal CDF and PDF,
$$
\mathbb{E}|z_t| \;=\; \sum_{m=1}^{2} p_m \left[\,\mu_m \left(2\Phi(\mu_m/\sigma_m) - 1\right) \;+\; 2\sigma_m\,\varphi(\mu_m/\sigma_m)\right].
$$

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

### Mixture of two normals

The standardized innovation follows a two-component Gaussian mixture with parameters $(p_1, \mu_1, \sigma_1^{2})$:
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



### Gaussian Mixture distribution
The mixture of 2 normal distribution given parameters $p_1$,$\mu_1$,$\sigma^2_1$ is 
$$
 f\left(z_t\right) =   
 p_1  \frac{1}{\sqrt{ 2\pi\sigma_1^2} } exp\{ -\frac{(z_t-\mu_1)^2}{2\sigma_1^2} \}
+p_2\frac{1}{\sqrt{ 2\pi\sigma_2^2} } exp\{ -\frac{(z_t-\mu_2)^2}{2\sigma_2^2} \}
$$
where $p_2 = 1- p_1 $,$\mu_2 = \frac{-p_1\mu_1}{p_2}$, and
$\sigma_2^2 =\frac{1-p_2\mu_2^2 -p_1(\sigma_1^2 + \mu_1^2 )}{p_2} $

N th order moment can be calculated as:
$$
E[z^n] = \int z^n   \left[p_1 \frac{1}{\sqrt{ 2\pi\sigma_1^2} } exp\{ -\frac{(z-\mu_1)^2}{2\sigma_1^2} \}
+(1-p_1)\frac{1}{\sqrt{ 2\pi\sigma_2^2} } exp\{ -\frac{(z-\mu_2)^2}{2\sigma_2^2} \}  \right] dz
$$

$$
E[z^n] =  p_1 E[x_1^n|x_1 \sim N(\mu_1,\sigma_1^2) ] + p_2 E[x_2^n|x_2 \sim N(\mu_2,\sigma_2^2) ]
$$


Similary, CDF can be calculated analytically by
$$
\Phi(a) = \int_{-\inf}^{a} z   \left[p_1 \frac{1}{\sqrt{ 2\pi\sigma_1^2} } exp\{ -\frac{(z-\mu_1)^2}{2\sigma_1^2} \}
+p_2\frac{1}{\sqrt{ 2\pi\sigma_2^2} } exp\{ -\frac{(z-\mu_2)^2}{2\sigma_2^2} \}  \right] dz
$$

$$
\Phi(a) = p_1\Phi_{x_1}(a) + p_2\Phi_{x_2}(a)
$$

We can think mixture of two normal distributions generated by three random variable $Z$,$X_1$,$X_2$. $Z$ follows Bernoulli($P_1$), $X_1 \sim  N(\mu_1,\sigma_1^2)$ and $X_2 \sim  N(\mu_2,\sigma_2^2)$. In order to generate a random sample point, we can first use $Z$ to decide which normal distributions to use, and then randomly pick a point in that normal distribution. 


Log likelihood function is
$$
 L\left(u_t\right) =  \sum_{t=1}^{T} \ln \frac{1}{\sigma_t} \left[
 p_1  \frac{1}{\sqrt{ 2\pi\sigma_1^2} } exp\{ -\frac{(z_t-\mu_1)^2}{2\sigma_1^2} \}
+p_2\frac{1}{\sqrt{ 2\pi\sigma_2^2} } exp\{ -\frac{(z_t-\mu_2)^2}{2\sigma_2^2} \} \right]
$$,
where $z_t = \frac{u_t}{\sigma_t}$


## Estimation Summary




## Best Estimation Illustrations

 
**Table 2: Model selection by AIC: GJR vs EGARCH**

| Distribution   | GJR Mean | GJR Vol | GJR AIC                          | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|----------------|----------|---------|----------------------------------|-------------|------------|------------|
| Normal         | (1,1)    | (2,1)   | 359.4710                         | (1,1)       | (1,1)      | 368.2429   |
| $t$            | (1,1)    | (2,1)   | <span style="color:red">343.1431</span> | (2,1)       | (1,1)      | 344.6254   |
| Skew $t$       | (1,1)    | (2,1)   | 345.1312                         | (2,1)       | (1,1)      | 346.5791   |
| GED            | (1,1)    | (2,1)   | 345.1925                         | (1,1)       | (1,1)      | 347.6934   |
| Mix of Normal  | (1,1)    | (2,1)   | 345.8222                         | (2,1)       | (2,1)      | 343.5023   |

**Table 3: Model selection by BIC: GJR vs EGARCH**

| Distribution   | GJR Mean | GJR Vol | GJR BIC  | EGARCH Mean | EGARCH Vol | EGARCH BIC                       |
|----------------|----------|---------|----------|-------------|------------|----------------------------------|
| Normal         | FC       | (2,1)   | 381.4280 | FC          | (1,1)      | 383.4754                         |
| $t$            | FC       | (1,1)   | 367.8350 | FC          | (1,1)      | <span style="color:red">367.1732</span> |
| Skew $t$       | FC       | (1,1)   | 370.4242 | FC          | (1,1)      | 369.6649                         |
| GED            | FC       | (1,1)   | 371.8489 | FC          | (1,1)      | 370.9644                         |
| Mix of Normal  | FC       | (1,1)   | 373.7106 | FC          | (1,1)      | 371.0689                         |

