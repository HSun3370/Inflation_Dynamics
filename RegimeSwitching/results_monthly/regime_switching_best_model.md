```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among the eligible estimates. Regimes are labeled by increasing innovation variance.

Final-solution stability: verified.

## Model Summary
| Mean process | Regime-switching parts | Non-switching parts | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- |
| ARX(1,1) | $\sigma^2$ | AR block, including $c$, SPF block | 100.3983 | -176.7965 | -122.3215 |

## Empirical Path Quantiles
Quantiles over the effective sample of $v_{t+1}=\operatorname{Var}(u_{t+1}\mid\mathcal F_t)$, where $u_{t+1}=\pi_{t+1}-\hat\pi_{t+1}$. Using the regime probabilities $Q_t$ before observing inflation and means $m_{i,t}=\mathbb D_iX_t$:

$$
v_{t+1}=\sum_i Q_{i,t}\sigma_i^2+\sum_i Q_{i,t}(m_{i,t}-\hat\pi_{t+1})^2,\qquad\hat\pi_{t+1}=\sum_i Q_{i,t}m_{i,t}.
$$

| Series | 5% | Median | 95% |
| --- | --- | --- | --- |
| $v_{t+1}$ | 0.0246 | 0.0420 | 0.1287 |

## Switching Blocks
| Block | Switching |
| --- | --- |
| AR block, including $c$ | N |
| SPF block | N |
| Shock distribution ($\sigma^2$) | Y |

## Mean Process
$$
\pi_{t+1} = c + \rho_1\,\pi_t + \phi_1\,SPF_t + \sigma_i W_{t+1},\qquad Z_t=e_i.
$$

Here $Z_t=e_i$ denotes regime $i$, which generates $\pi_{t+1}$, and $W_{t+1}$ is the standardized innovation with the distribution reported below.

## Mean Parameter Values
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $c$ | N | 0.0182 | 0.0182 | 0.0182 |
| $\rho_1$ | N | 0.4453 | 0.4453 | 0.4453 |
| $\phi_1$ | N | 0.5638 | 0.5638 | 0.5638 |

## Distribution Parameters
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $\sigma$ | Y | 0.1320 | 0.2914 | 0.6741 |

## Transition Probability Matrix
| From \ To | Regime 0 | Regime 1 | Regime 2 | Row Sum |
| --- | --- | --- | --- | --- |
| Regime 0 | 0.9482 | 0.0518 | 0.0000 | 1.0000 |
| Regime 1 | 0.0749 | 0.8978 | 0.0273 | 1.0000 |
| Regime 2 | 0.0000 | 0.2919 | 0.7081 | 1.0000 |

## Regime Classification Plot
Shading selects the largest component of $S_t=E(Z_t\mid\mathcal F_T)$ and displays it at the date of $\pi_{t+1}$. It uses the full sample.

![Inflation with Smoothed Regime Classifications](best_model_regime_classification.png)