```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among estimates that passed optimizer convergence, stationarity, parameter-bound, distribution, and transition checks.

## Model Summary
| Mean process | Regime-switching parts | Non-switching parts | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- |
| ARX(2,1) | AR block, including $c$, $\sigma^2$ | SPF block | -157.6865 | 353.3729 | 417.4150 |


## Mean Process
$$
\pi_{t+1} = c_{s_t} + \rho_{1,s_t}\,\pi_t + \rho_{2,s_t}\,\pi_{t-1} + \phi_1\,SPF_t + u_{t+1}
$$

## Mean Parameter Values
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $c$ | Y | 0.2893 | -0.1234 | 0.1561 |
| $\rho_1$ | Y | 0.0142 | 0.1989 | 0.2226 |
| $\rho_2$ | Y | -0.1378 | 0.4114 | -0.0895 |
| $\phi_1$ | N | 0.7487 | 0.7487 | 0.7487 |

## Distribution Parameters
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $\sigma^2$ | Y | 0.0517 | 0.1973 | 1.4556 |

## Transition Probability Matrix
| From \ To | Regime 0 | Regime 1 | Regime 2 | Row Sum |
| --- | --- | --- | --- | --- |
| Regime 0 | 0.7705 | 0.2291 | 0.0004 | 1.0000 |
| Regime 1 | 0.1010 | 0.8367 | 0.0623 | 1.0000 |
| Regime 2 | 0.2227 | 0.0005 | 0.7768 | 1.0000 |

## Regime Classification Plot
![Inflation with Smoothed Predicted Regimes](best_model_regime_classification.png)