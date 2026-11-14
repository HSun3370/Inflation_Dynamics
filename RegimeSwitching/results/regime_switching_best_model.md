```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among the eligible estimates. Regimes are labeled by increasing innovation variance.

Final-solution stability: verified.

## Model Summary
| Mean process | Regime-switching parts | Non-switching parts | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- |
| ARX(1,1) | AR block, including $c$, $\sigma^2$ | SPF block | -179.8177 | 377.6354 | 408.5781 |

## Empirical Path Quantiles
Quantiles over the effective sample of $v_{t+1}=\operatorname{Var}(u_{t+1}\mid\mathcal F_t)$, where $u_{t+1}=\pi_{t+1}-\hat\pi_{t+1}$. Using the regime probabilities $Q_t$ before observing inflation and means $m_{i,t}=\mathbb D_iX_t$:

$$
v_{t+1}=\sum_i Q_{i,t}\sigma_i^2+\sum_i Q_{i,t}(m_{i,t}-\hat\pi_{t+1})^2,\qquad\hat\pi_{t+1}=\sum_i Q_{i,t}m_{i,t}.
$$

| Series | 5% | Median | 95% |
| --- | --- | --- | --- |
| $v_{t+1}$ | 0.1983 | 0.2721 | 0.8430 |

## Switching Blocks
| Block | Switching |
| --- | --- |
| AR block, including $c$ | Y |
| SPF block | N |
| Shock distribution ($\sigma^2$) | Y |

## Mean Process
$$
\pi_{t+1} = c_i + \rho_{1,i}\,\pi_t + \phi_1\,SPF_t + \sigma_i W_{t+1},\qquad Z_t=e_i.
$$

Here $Z_t=e_i$ denotes regime $i$, which generates $\pi_{t+1}$, and $W_{t+1}$ is the standardized innovation with the distribution reported below.

## Mean Parameter Values
| Parameter | Switching | Regime 0 | Regime 1 |
| --- | --- | --- | --- |
| $c$ | Y | 0.2258 | 0.0987 |
| $\rho_1$ | Y | 0.0293 | 0.3597 |
| $\phi_1$ | N | 0.7415 | 0.7415 |

## Distribution Parameters
| Parameter | Switching | Regime 0 | Regime 1 |
| --- | --- | --- | --- |
| $\sigma$ | Y | 0.3287 | 0.9486 |

## Transition Probability Matrix
| From \ To | Regime 0 | Regime 1 | Row Sum |
| --- | --- | --- | --- |
| Regime 0 | 0.9216 | 0.0784 | 1.0000 |
| Regime 1 | 0.1633 | 0.8367 | 1.0000 |

## Regime Classification Plot
Shading selects the largest component of $S_t=E(Z_t\mid\mathcal F_T)$ and displays it at the date of $\pi_{t+1}$. It uses the full sample.

![Inflation with Smoothed Regime Classifications](best_model_regime_classification.png)