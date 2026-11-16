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
| ARX(1,1) | 3 | Y | N | Y | -167.3355 | 366.6710 | 421.6803 |
| ARX(2,1) | 2 | N | N | Y | -181.6105 | 379.2209 | 406.7256 |
| ARX(2,1) | 2 | Y | N | Y | -179.5673 | 381.1345 | 418.9534 |
| ARX(2,1) | 2 | Y | Y | Y | -178.8889 | 381.7779 | 423.0348 |
| ARX(2,2) | 2 | N | N | Y | -181.5572 | 381.1145 | 412.0572 |
| ARX(2,2) | 2 | Y | N | Y | -179.5338 | 383.0675 | 424.3245 |
| ARX(2,2) | 2 | Y | Y | Y | -178.7086 | 385.4172 | 433.5503 |