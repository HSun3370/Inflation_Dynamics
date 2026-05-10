
```{raw:typst}
#set page(margin: auto)
```

# GARCH 

## Model Specification

GARCH(p,q) model specification:
$$
 \sigma_{t}^{2}=\omega +\sum_{i=1}^{p}\alpha_{i} u_{t-i}^2
         +\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{2} 
$$





# GARCH Estimation in Python's `arch` Package: Sample Use and Backcasting

## How `arch` Uses the Sample

For a GARCH(p, q) model with $N$ observations and residuals $\epsilon_t$:

$$
\sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \, \epsilon_{t-i}^2 + \sum_{j=1}^{q} \beta_j \, \sigma_{t-j}^2
$$

`arch` uses **all $N$ observations** in the log-likelihood — it does not drop the first $\max(p, q)$. The unobserved pre-sample values are filled in via **backcasting**: a single value $\hat{\sigma}_{\text{bc}}^2$ initializes all required pre-sample quantities,

$$
\hat{\epsilon}_{0}^2 = \hat{\epsilon}_{-1}^2 = \cdots = \hat{\sigma}_{0}^2 = \hat{\sigma}_{-1}^2 = \cdots = \hat{\sigma}_{\text{bc}}^2
$$

## The Four Standard Backcasting Methods

**1. Sample (unconditional) variance.** Equal weighting of squared residuals:

$$
\hat{\sigma}_{\text{bc}}^2 = \frac{1}{N} \sum_{t=1}^{N} \epsilon_t^2
$$

Default in EViews and MATLAB.

**2. Model-implied long-run variance (variance targeting).** Unconditional variance from current parameters:

$$
\hat{\sigma}_{\text{bc}}^2 = \frac{\omega}{1 - \sum_{i} \alpha_i - \sum_{j} \beta_j}
$$

Undefined when $\sum \alpha_i + \sum \beta_j \geq 1$.


## Estimation Summary




## Best Estimation Illustrations



