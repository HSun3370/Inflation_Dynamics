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
| ARX(2,1) | 2 | N | N | Y | -181.6105 | 379.2209 | 406.7256 |
| ARX(2,1) | 2 | Y | N | Y | -179.5673 | 381.1345 | 418.9534 |
| ARX(2,1) | 3 | N | N | Y | -176.7746 | 379.5492 | 424.2443 |
| ARX(2,2) | 2 | N | N | Y | -181.5572 | 381.1145 | 412.0572 |
| ARX(2,2) | 2 | Y | N | Y | -179.5338 | 383.0675 | 424.3245 |
| ARX(2,2) | 2 | Y | Y | Y | -178.7086 | 385.4172 | 433.5503 |
| ARX(2,2) | 3 | N | N | Y | -176.7722 | 381.5444 | 429.6775 |

The following specifications have no verified estimate and are excluded from the table above and from model selection. LogLik and the smallest regime variance $\min_k\sigma_k^2$ refer to the rejected estimate. No variance floor is imposed, so the Gaussian likelihood is unbounded as a regime variance approaches zero; an estimate with $\min_k\sigma_k^2$ near zero is such a degenerate optimum.

| Mean | K | Sw.AR | Sw.SPF | LogLik | Min $\sigma^2$ | Refits returned | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARX(1,1) | 3 | Y | N | -158.3755 | 1.47e-07 | 2/20 | Not all perturbation refits returned |
| ARX(2,1) | 2 | Y | Y | -175.4087 | 8.72e-05 | 9/20 | Not all perturbation refits returned |
| ARX(2,1) | 3 | Y | N | -151.5653 | 2.73e-08 | 1/20 | Not all perturbation refits returned |
| ARX(2,2) | 3 | Y | N | -19.3506 | 1.70e-31 | 0/20 | Not all perturbation refits returned |