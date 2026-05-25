```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among converged models.

## Model Summary
- Model label: `ARX(2,1)_student_t_r3_arY_spfN`
- Mean process: `ARX(2,1)`
- Mean equation target: `Inflation`
- Intercept included: `Y`
- Intercept switches by regime: `Y`
- Error distribution: `Student t`
- #Regime: `3`
- Switching AR: `Y`
- Switching SPF: `N`
- Switching distribution variance: `Y`
- Switching nu: `Y`
- Sample: `1969Q2 to 2022Q4`
- Nobs: `215`
- LogLik: `-150.126`
- AIC: `344.253`
- BIC: `418.407`

## Mean Model Specification
`Inflation = c_(s_t) + beta_{Inflation_lag_1}(s_t) * Inflation_lag_1 + beta_{Inflation_lag_2}(s_t) * Inflation_lag_2 + beta_{SPF} * SPF + u_t`

Regime-specific mean coefficients:
|   regime |   Intercept |   Inflation_lag_1 |   Inflation_lag_2 |      SPF |
|---------:|------------:|------------------:|------------------:|---------:|
|        0 |   0.872769  |        -0.54292   |        -0.238057  | 0.574951 |
|        1 |   0.381796  |         0.0492982 |        -0.0989449 | 0.574951 |
|        2 |  -0.0670625 |         0.26872   |         0.448281  | 0.574951 |

## Mean Process Parameters
| parameter          |   estimate |
|:-------------------|-----------:|
| const[0]           |  0.872769  |
| const[1]           |  0.381796  |
| const[2]           | -0.0670625 |
| Inflation_lag_1[0] | -0.54292   |
| Inflation_lag_1[1] |  0.0492982 |
| Inflation_lag_1[2] |  0.26872   |
| Inflation_lag_2[0] | -0.238057  |
| Inflation_lag_2[1] | -0.0989449 |
| Inflation_lag_2[2] |  0.448281  |
| SPF                |  0.574951  |

## Distribution Parameters
| parameter   |    estimate |
|:------------|------------:|
| sigma2[0]   |   0.0118058 |
| sigma2[1]   |   0.612973  |
| sigma2[2]   |   0.208789  |
| nu[0]       | 183.071     |
| nu[1]       |   2.39744   |
| nu[2]       | 104.326     |

## Transition Probabilities (Vector Form)
| parameter   |    estimate |
|:------------|------------:|
| p[0->0]     | 0.832798    |
| p[1->0]     | 0.0480447   |
| p[2->0]     | 0.000150046 |
| p[0->1]     | 0.165512    |
| p[1->1]     | 0.646779    |
| p[2->1]     | 0.24607     |

## Transition Probability Matrix
| From \ To   |    Regime 0 |   Regime 1 |   Regime 2 |   Row Sum |
|:------------|------------:|-----------:|-----------:|----------:|
| Regime 0    | 0.832798    |   0.165512 | 0.00168973 |         1 |
| Regime 1    | 0.0480447   |   0.646779 | 0.305177   |         1 |
| Regime 2    | 0.000150046 |   0.24607  | 0.75378    |         1 |

## Regime Classification Plot
![Inflation with Smoothed Predicted Regimes](best_model_regime_classification.png)