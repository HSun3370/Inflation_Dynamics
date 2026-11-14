```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```



Each model is selected from 5,000 starts; only estimates that passed all checks (optimizer convergence, AR stationarity, positive finite variances, valid transition probabilities, and final-solution stability) are reported. Regimes are labeled by increasing innovation variance. `Sw.Dist` reflects whether $\sigma^2$ is regime-specific.

| Mean | K | Sw.AR | Sw.SPF | Sw.Dist | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 2 | N | N | Y | 19.5811 | -31.1623 | -13.0039 |
| ARX(1,1) | 2 | N | N | Y | 89.9563 | -165.9125 | -134.1354 |
| ARX(1,1) | 2 | Y | N | Y | 91.8501 | -165.7002 | -124.8439 |
| ARX(1,1) | 2 | Y | Y | Y | 91.8590 | -163.7179 | -118.3221 |
| ARX(1,1) | 3 | N | N | Y | 100.3983 | -176.7965 | -122.3215 |
| ARX(2,1) | 2 | N | N | Y | 90.0731 | -164.1462 | -127.8295 |
| ARX(2,1) | 2 | Y | N | Y | 92.3528 | -162.7057 | -112.7702 |
| ARX(2,1) | 2 | Y | Y | Y | 92.3793 | -160.7587 | -106.2836 |
| ARX(2,2) | 2 | N | N | Y | 90.1563 | -162.3127 | -121.4564 |
| ARX(2,2) | 2 | Y | N | Y | 92.4836 | -160.9672 | -106.4922 |
| ARX(2,2) | 2 | Y | Y | Y | 93.1905 | -158.3810 | -94.8268 |

The following specifications have no verified estimate and are excluded from the table above and from model selection. LogLik and the smallest regime variance $\min_k\sigma_k^2$ refer to the rejected estimate. No variance floor is imposed, so the Gaussian likelihood is unbounded as a regime variance approaches zero; an estimate with $\min_k\sigma_k^2$ near zero is such a degenerate optimum.

| Mean | K | Sw.AR | Sw.SPF | LogLik | Min $\sigma^2$ | Refits returned | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 3 | N | N | 38.6851 | 1.32e-02 | 15/20 | Not all perturbation refits returned |
| ARX(1,1) | 3 | Y | N | 118.1786 | 8.25e-06 | 9/20 | Not all perturbation refits returned |
| ARX(2,1) | 3 | N | N | 100.6976 | 1.75e-02 | 19/20 | Not all perturbation refits returned |
| ARX(2,1) | 3 | Y | N | 134.3093 | 3.47e-07 | 0/20 | Not all perturbation refits returned |
| ARX(2,2) | 3 | N | N | 100.8275 | 1.75e-02 | 18/20 | Not all perturbation refits returned |
| ARX(2,2) | 3 | Y | N | 133.3244 | 9.15e-07 | 0/20 | Not all perturbation refits returned |