```{raw:typst}
#set page(margin: auto)
```

Each model is selected from 50 starts; only estimates that passed all checks (optimizer convergence, AR stationarity, parameter bounds, and valid transition probabilities) are reported. Specifications where no start passed all checks are excluded. `Sw.Dist` reflects whether $\sigma^2$ is regime-specific.

| Mean | K | Sw.AR | Sw.SPF | Sw.Dist | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 2 | N | N | Y | 21.0036 | -34.0071 | -15.8720 |
| ARX(1,1) | 2 | N | N | Y | 91.9010 | -169.8020 | -138.0655 |
| ARX(1,1) | 2 | Y | N | Y | 93.7691 | -169.5382 | -128.7341 |
| ARX(1,1) | 2 | Y | Y | Y | 93.7702 | -167.5405 | -122.2026 |
| ARX(1,1) | 3 | N | N | Y | 91.9024 | -159.8047 | -105.3992 |
| ARX(1,1) | 3 | Y | N | Y | 112.5283 | -193.0566 | -120.5160 |
| ARX(2,1) | 2 | N | N | Y | 91.9497 | -167.8995 | -131.6292 |
| ARX(2,1) | 2 | Y | N | Y | 94.1403 | -166.2805 | -116.4089 |
| ARX(2,1) | 2 | Y | Y | Y | 94.1451 | -164.2902 | -109.8847 |
| ARX(2,1) | 3 | N | N | Y | 91.9657 | -157.9314 | -98.9921 |
| ARX(2,1) | 3 | Y | N | Y | 119.1499 | -200.2998 | -114.1578 |
| ARX(2,2) | 2 | N | N | Y | 92.0116 | -166.0232 | -125.2191 |
| ARX(2,2) | 2 | Y | N | Y | 94.2497 | -164.4994 | -110.0939 |
| ARX(2,2) | 2 | Y | Y | Y | 95.0807 | -162.1613 | -98.6883 |
| ARX(2,2) | 3 | N | N | Y | 102.9279 | -177.8558 | -114.3828 |
| ARX(2,2) | 3 | Y | N | Y | 119.9819 | -199.9639 | -109.2881 |