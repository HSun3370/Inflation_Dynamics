```{raw:typst}
#set page(margin: auto)
```

Each model is selected from 5,000 starts; only estimates that passed all checks (optimizer convergence, AR stationarity, positive finite variances, valid transition probabilities, and final-solution stability) are reported. Regimes are labeled by increasing innovation variance. `Sw.Dist` reflects whether $\sigma^2$ is regime-specific.

| Mean | K | Sw.AR | Sw.SPF | Sw.Dist | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 2 | N | N | Y | -191.8370 | 391.6741 | 405.4264 |
| Constant | 3 | N | N | Y | -186.8808 | 391.7616 | 422.7044 |
| ARX(1,1) | 2 | N | N | Y | -182.9057 | 379.8114 | 403.8779 |
| ARX(1,1) | 2 | Y | N | Y | -179.8177 | 377.6354 | 408.5781 |
| ARX(1,1) | 2 | Y | Y | Y | -179.6711 | 379.3423 | 413.7231 |
| ARX(1,1) | 3 | N | N | Y | -179.7370 | 383.4741 | 424.7310 |
| ARX(1,1) | 3 | Y | N | Y | -167.3355 | 366.6710 | 421.6803 |
| ARX(2,1) | 2 | N | N | Y | -181.6105 | 379.2209 | 406.7256 |
| ARX(2,1) | 3 | N | N | Y | -176.7746 | 379.5492 | 424.2443 |
| ARX(2,2) | 2 | N | N | Y | -181.5572 | 381.1145 | 412.0572 |
| ARX(2,2) | 2 | Y | Y | Y | -178.7086 | 385.4172 | 433.5503 |
| ARX(2,2) | 3 | N | N | Y | -176.7722 | 381.5444 | 429.6775 |

The following specifications have no verified estimate and are excluded from the table above and from model selection. LogLik and the smallest regime variance $\min_k\sigma_k^2$ refer to the rejected estimate. No variance floor is imposed, so the Gaussian likelihood is unbounded as a regime variance approaches zero; an estimate with $\min_k\sigma_k^2$ near zero is such a degenerate optimum.

| Mean | K | Sw.AR | Sw.SPF | LogLik | Min $\sigma^2$ | Refits returned | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARX(2,1) | 2 | Y | N | -91.2733 | 4.52e-31 | -- | Likelihood not reproduced by its parameters |
| ARX(2,1) | 2 | Y | Y | -175.4087 | 8.72e-05 | 9/20 | Not all perturbation refits returned |
| ARX(2,1) | 3 | Y | N | -83.8557 | 1.12e-25 | 0/20 | Not all perturbation refits returned |
| ARX(2,2) | 2 | Y | N | -102.8982 | 7.09e-29 | -- | Likelihood not reproduced by its parameters |
| ARX(2,2) | 3 | Y | N | -19.4123 | 1.49e-31 | 0/20 | Not all perturbation refits returned |