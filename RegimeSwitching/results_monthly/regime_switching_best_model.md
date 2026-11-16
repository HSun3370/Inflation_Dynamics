```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```



# Best Regime-Switching Model

Selected by AIC among the eligible estimates. The model has $K=3$ regimes, labeled by increasing innovation variance.

Final-solution stability: verified.

Effective sample: 1969-02–2026-09 (692 observations).

## Model Summary
| Mean process | Regime-switching parts | Non-switching parts | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- |
| ARX(2,1) | AR block, including $c$, $\sigma^2$ | SPF block | 116.7792 | -195.5584 | -109.3062 |

## Unconditional and Sample-Averaged Moments

The first two rows average the fitted predictive distributions uniformly over the observed sample. They are not stationary long-run moments of inflation: the model does not specify the joint stationary law of inflation lags and SPF. The third row is the stationary distribution of the regime shock $\varepsilon_{t+1}=\sigma_iz_{t+1}$, weighted by $\bar Q$. When means switch, this shock differs from the forecast residual $u_{t+1}=\pi_{t+1}-\hat{\pi}_{t+1}$.

| Distribution | Mean | Variance | Skewness | Kurtosis |
| --- | --- | --- | --- | --- |
| Inflation: sample-averaged predictive mixture | 0.3242 | 0.1057 | 0.4764 | 5.0715 |
| Residual: sample-averaged predictive mixture | 0.0000 | 0.0569 | 0.0094 | 6.3266 |
| Regime shock: stationary mixture | 0.0000 | 0.0525 | 0.0000 | 6.8249 |

Mean is measured in percentage points per period and variance in squared percentage points. Skewness and kurtosis are dimensionless. Kurtosis is Pearson kurtosis (Gaussian = 3); excess kurtosis is this value minus 3.

## Empirical Distributions of Conditional Moments

Each observation contributes one value of the predictive mean, variance, skewness, and kurtosis, calculated using $Q_t$ before observing inflation. The table and empirical CDFs describe variation over dates, not sampling uncertainty about parameters. The formulas are given in the regime-switching methodology section.

| Conditional moment | Average | 5% | 25% | Median | 75% | 95% |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 0.3242 | 0.0639 | 0.2095 | 0.2710 | 0.3861 | 0.8122 |
| variance | 0.0569 | 0.0269 | 0.0304 | 0.0390 | 0.0707 | 0.1393 |
| skewness | 0.0308 | -0.7009 | -0.1087 | 0.0748 | 0.2319 | 0.5703 |
| kurtosis | 6.3105 | 3.1735 | 5.2608 | 6.8674 | 7.5770 | 7.9806 |

The regime means differ, so the predictive distribution can be skewed even though each regime is Gaussian. Its variance includes uncertainty about the regime mean.

The average conditional skewness or kurtosis generally differs from the skewness or kurtosis of the sample-averaged distribution above.

![Empirical CDFs of the four conditional moments](best_model_conditional_moment_ecdfs.png)

## Switching Blocks
| Block | Switching |
| --- | --- |
| AR block, including $c$ | Y |
| SPF block | N |
| Shock distribution ($\sigma^2$) | Y |

## Mean Process
$$
\pi_{t+1} = c_i + \rho_{1,i}\,\pi_t + \rho_{2,i}\,\pi_{t-1} + \phi_1\,SPF_t + \sigma_i z_{t+1},\qquad Z_t=e_i.
$$

Here $Z_t=e_i$ denotes regime $i$, which generates $\pi_{t+1}$, and $z_{t+1}$ is the standardized innovation with the distribution reported below.

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

## Sample Mean Inflation by Regime
| Regime | Assigned dates | Sample mean of $\pi$ |
| --- | --- | --- |
| 0 | 255 | 0.2492 |
| 1 | 282 | 0.3633 |
| 2 | 155 | 0.3808 |

**Note.** Each observation $\pi_{t+1}$ is assigned to the regime $i=\arg\max_j\widetilde{Q}_{j,t}$, using the same smoothed classification as the plot. The table reports the simple, unweighted average of observed inflation over the dates assigned to each regime, in percent per quarter or month, without annualization. These sample means illustrate average inflation within each classified regime; they are not model-implied unconditional means.

## Transition Probability Matrix
| From \ To | Regime 0 | Regime 1 | Regime 2 | Row Sum |
| --- | --- | --- | --- | --- |
| Regime 0 | 0.9114 | 0.0493 | 0.0393 | 1.0000 |
| Regime 1 | 0.0376 | 0.9091 | 0.0533 | 1.0000 |
| Regime 2 | 0.0632 | 0.0777 | 0.8592 | 1.0000 |

## Regime Classification Plot
Shading selects the largest component of $\widetilde{Q}_t=E[Z_t\mid\mathcal{F}_T]$ and displays it at the date of $\pi_{t+1}$. It uses the full sample.

![Inflation with Smoothed Regime Classifications](best_model_regime_classification.png)
