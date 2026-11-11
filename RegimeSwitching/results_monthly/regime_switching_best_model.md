```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```



# Best Regime-Switching Model

The initial search used the former transition and variance cutoffs. The expanded admissible set requires a fresh 5,000-start search.

Selected by AIC among the eligible estimates. Regimes are labeled by increasing innovation variance.

Final-solution stability: not_checked.

## Model Summary
| Mean process | Regime-switching parts | Non-switching parts | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- |
| ARX(2,1) | AR block, including $c$, $\sigma^2$ | SPF block | 116.7792 | -195.5584 | -109.3062 |

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
| $c$ | Y | 0.2172 | -0.0137 | 0.0499 |
| $\rho_1$ | Y | 0.0512 | 0.5057 | 0.5434 |
| $\rho_2$ | Y | -0.3544 | 0.1678 | -0.0587 |
| $\phi_1$ | N | 0.4744 | 0.4744 | 0.4744 |

## Distribution Parameters
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $\sigma$ | Y | 0.1317 | 0.1386 | 0.3943 |

## Transition Probability Matrix
| From \ To | Regime 0 | Regime 1 | Regime 2 | Row Sum |
| --- | --- | --- | --- | --- |
| Regime 0 | 0.9114 | 0.0493 | 0.0393 | 1.0000 |
| Regime 1 | 0.0376 | 0.9091 | 0.0533 | 1.0000 |
| Regime 2 | 0.0632 | 0.0777 | 0.8592 | 1.0000 |

## Regime Classification Plot
![Inflation with Smoothed Predicted Regimes](best_model_regime_classification.png)