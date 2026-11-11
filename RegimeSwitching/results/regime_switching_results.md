```{raw:typst}
#set page(margin: auto)
```

These saved estimates were selected from 5,000 starts under the previous collection rules. Regimes are relabeled by increasing innovation variance. Final-solution stability has not yet been verified. `Sw.Dist` reflects whether $\sigma^2$ is regime-specific.

The initial search used the former transition and variance cutoffs. A fresh 5,000-start search is required to rank all candidates under the updated collection rules; verification of saved estimates checks their local stability only.

| Mean | K | Sw.AR | Sw.SPF | Sw.Dist | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 2 | N | N | Y | -191.8370 | 391.6741 | 405.4264 |
| Constant | 3 | N | N | Y | -186.8817 | 391.7634 | 422.7061 |
| ARX(1,1) | 2 | N | N | Y | -182.9057 | 379.8114 | 403.8779 |
| ARX(1,1) | 2 | Y | N | Y | -179.8177 | 377.6354 | 408.5781 |
| ARX(1,1) | 2 | Y | Y | Y | -179.6711 | 379.3423 | 413.7231 |
| ARX(1,1) | 3 | N | N | Y | -179.7484 | 383.4968 | 424.7537 |
| ARX(1,1) | 3 | Y | N | Y | -167.3366 | 366.6733 | 421.6826 |
| ARX(2,1) | 2 | N | N | Y | -181.6105 | 379.2209 | 406.7256 |
| ARX(2,1) | 2 | Y | N | Y | -179.5673 | 381.1345 | 418.9534 |
| ARX(2,1) | 2 | Y | Y | Y | -178.8889 | 381.7779 | 423.0348 |
| ARX(2,1) | 3 | N | N | Y | -179.5464 | 385.0927 | 429.7877 |
| ARX(2,1) | 3 | Y | N | Y | -161.4051 | 360.8101 | 426.1336 |
| ARX(2,2) | 2 | N | N | Y | -181.5572 | 381.1145 | 412.0572 |
| ARX(2,2) | 2 | Y | N | Y | -179.5338 | 383.0675 | 424.3245 |
| ARX(2,2) | 2 | Y | Y | Y | -178.7086 | 385.4172 | 433.5503 |
| ARX(2,2) | 3 | N | N | Y | -176.7801 | 381.5602 | 429.6933 |
| ARX(2,2) | 3 | Y | N | Y | -160.0023 | 360.0046 | 428.7662 |