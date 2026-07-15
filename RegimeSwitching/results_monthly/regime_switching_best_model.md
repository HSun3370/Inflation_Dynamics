```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among estimates that passed optimizer convergence, stationarity, parameter-bound, distribution, and transition checks.

## Model Summary
| Mean process | Regime-switching parts | Non-switching parts | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- |
| ARX(2,1) | AR block, including $c$, $\sigma^2$ | SPF block | 119.1499 | -200.2998 | -114.1578 |

## Switching Blocks
| Block | Switching |
| --- | --- |
| AR block, including $c$ | Y |
| SPF block | N |
| Shock distribution ($\sigma^2$) | Y |

## Mean Process
$$
\pi_{t+1} = c_{s_t} + \rho_{1,s_t}\,\pi_t + \rho_{2,s_t}\,\pi_{t-1} + \phi_1\,SPF_t + u_{t+1}
$$

## Mean Parameter Values
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $c$ | Y | 0.2179 | -0.0112 | 0.0480 |
| $\rho_1$ | Y | 0.0504 | 0.5104 | 0.5410 |
| $\rho_2$ | Y | -0.3566 | 0.1595 | -0.0440 |
| $\phi_1$ | N | 0.4716 | 0.4716 | 0.4716 |

## Distribution Parameters
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $\sigma^2$ | Y | 0.0174 | 0.0197 | 0.1595 |

## Transition Probability Matrix
| From \ To | Regime 0 | Regime 1 | Regime 2 | Row Sum |
| --- | --- | --- | --- | --- |
| Regime 0 | 0.9142 | 0.0439 | 0.0419 | 1.0000 |
| Regime 1 | 0.0399 | 0.9073 | 0.0528 | 1.0000 |
| Regime 2 | 0.0598 | 0.0973 | 0.8429 | 1.0000 |

## Regime Classification Plot
![Inflation with Smoothed Predicted Regimes](best_model_regime_classification.png)