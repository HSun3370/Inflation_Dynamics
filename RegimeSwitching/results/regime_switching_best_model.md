```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among converged models.

## Model Summary
- Model label: `ARX(1,1)_normal_r3_arY_spfN`
- Mean process: `ARX(1,1)`
- Error distribution: `Normal`
- #Regime: `3`
- Switching AR: `Y`
- Switching SPF: `N`
- Switching distribution variance: `Y`
- Switching nu: `N`
- Sample: `1969Q2 to 2022Q4`
- Nobs: `215`
- LogLik: `-166.916`
- AIC: `359.833`
- BIC: `403.651`

## Mean Process Parameters
| parameter          |   estimate |
|:-------------------|-----------:|
| Inflation_lag_1[0] | 0.00416504 |
| Inflation_lag_1[1] | 0.426532   |
| Inflation_lag_1[2] | 0.087587   |
| SPF[2]             | 0.975286   |

## Distribution Parameters
| parameter   |   estimate |
|:------------|-----------:|
| sigma2[0]   |  0.0790304 |
| sigma2[1]   |  0.165777  |
| sigma2[2]   |  1.08714   |

## Transition Probabilities
| parameter   |    estimate |
|:------------|------------:|
| p[0->0]     | 0.872496    |
| p[1->0]     | 2.934e-08   |
| p[2->0]     | 0.246608    |
| p[0->1]     | 0.127503    |
| p[1->1]     | 0.796975    |
| p[2->1]     | 3.67488e-12 |