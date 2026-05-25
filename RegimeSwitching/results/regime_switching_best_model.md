```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among converged models.

## Model Summary
- Mean process: `ARX(2,1)`
- Error distribution: `Normal`
- #Regime: `3`
- Switching AR: `Y`
- Switching SPF: `N`
- Switching distribution variance: `Y`
- Sample: `1969Q2 to 2022Q4`
- Nobs: `215`
- LogLik: `-156.722`
- AIC: `351.445`
- BIC: `415.487`

## Mean Model Specification
$$\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1 SPF_t + \mu_{t+1}$$

Regime-specific mean coefficients:

|   regime |   $c$ |   $\rho_1$ |   $\rho_2$ |     $\phi_1$ |  $\sigma$ | 
|---------:|------------:|------------------:|------------------:|---------:|---------:|
|        0 |    0.588059 |         -0.468519 |         -0.153493 | 0.851381 |  0.0215162 |
|        1 |   -0.156002 |          0.263892 |          0.276048 | 0.851381 |  0.186912  |
|        2 |    0.337041 |          0.229981 |         -0.218419 | 0.851381 |  1.21593   |


## Transition Probability Matrix
| From \ To   |   Regime 0 |    Regime 1 |    Regime 2 | 
|:------------|-----------:|------------:|------------:| 
| Regime 0    |   0.765608 | 0.234391    | 3.06731e-08 | 
| Regime 1    |   0.039901 | 0.907905    | 0.0521942   | 
| Regime 2    |   0.176172 | 2.83845e-12 | 0.823828    | 



## Regime Classification Plot
![Inflation with Smoothed Predicted Regimes](best_model_regime_classification.png)