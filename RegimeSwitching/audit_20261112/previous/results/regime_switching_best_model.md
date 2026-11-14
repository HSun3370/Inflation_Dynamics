```{raw:typst}
#set page(margin: auto)
```

# Best Regime-Switching Model

Selected by AIC among the eligible estimates. Regimes are labeled by increasing innovation variance.

Final-solution stability: verified.

## Model Summary
| Mean process | Regime-switching parts | Non-switching parts | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- |
| ARX(1,1) | AR block, including $c$, $\sigma^2$ | SPF block | -167.3355 | 366.6710 | 421.6803 |

## Empirical Path Quantiles
Quantiles over the effective sample of the conditional variance of the residual $u_t=\pi_t-\mathrm{E}(\pi_t\mid\mathcal{F}_{t-1})$, $\sigma_t^2=\sum_k \xi_{k,t|t-1}\,(\sigma_k^2+m_{k,t}^2)-\big(\sum_k \xi_{k,t|t-1}\,m_{k,t}\big)^2$, where $\xi_{k,t|t-1}$ is the predicted probability of regime $k$ and $m_{k,t}$ its conditional mean.

| Series | 5% | Median | 95% |
| --- | --- | --- | --- |
| $\sigma_t^2$ | 0.1080 | 0.3078 | 0.9827 |

## Switching Blocks
| Block | Switching |
| --- | --- |
| AR block, including $c$ | Y |
| SPF block | N |
| Shock distribution ($\sigma^2$) | Y |

## Mean Process
$$
\pi_{t+1} = c_{s_t} + \rho_{1,s_t}\,\pi_t + \phi_1\,SPF_t + u_{t+1}
$$

## Mean Parameter Values
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $c$ | Y | 0.4496 | -0.0795 | 0.1578 |
| $\rho_1$ | Y | -0.3794 | 0.5227 | 0.1875 |
| $\phi_1$ | N | 0.7821 | 0.7821 | 0.7821 |

## Distribution Parameters
| Parameter | Switching | Regime 0 | Regime 1 | Regime 2 |
| --- | --- | --- | --- | --- |
| $\sigma$ | Y | 0.2067 | 0.3996 | 1.0549 |

## Transition Probability Matrix
| From \ To | Regime 0 | Regime 1 | Regime 2 | Row Sum |
| --- | --- | --- | --- | --- |
| Regime 0 | 0.8154 | 0.1846 | -0.0000 | 1.0000 |
| Regime 1 | 0.0415 | 0.8617 | 0.0968 | 1.0000 |
| Regime 2 | 0.1716 | 0.0495 | 0.7789 | 1.0000 |

## Regime Classification Plot
![Inflation with Smoothed Predicted Regimes](best_model_regime_classification.png)