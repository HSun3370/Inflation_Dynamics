```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```



These saved estimates were selected from 5,000 starts under the previous collection rules. Regimes are relabeled by increasing innovation variance. Final-solution stability has not yet been verified. `Sw.Dist` reflects whether $\sigma^2$ is regime-specific.

The initial search used the former transition and variance cutoffs. A fresh 5,000-start search is required to rank all candidates under the updated collection rules; verification of saved estimates checks their local stability only.

| Mean | K | Sw.AR | Sw.SPF | Sw.Dist | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 2 | N | N | Y | 19.5811 | -31.1623 | -13.0039 |
| Constant | 3 | N | N | Y | 38.6812 | -59.3623 | -18.5061 |
| ARX(1,1) | 2 | N | N | Y | 89.9563 | -165.9125 | -134.1354 |
| ARX(1,1) | 2 | Y | N | Y | 91.8501 | -165.7002 | -124.8439 |
| ARX(1,1) | 2 | Y | Y | Y | 91.8590 | -163.7179 | -118.3221 |
| ARX(1,1) | 3 | N | N | Y | 100.3806 | -176.7611 | -122.2861 |
| ARX(1,1) | 3 | Y | N | Y | 110.7946 | -189.5891 | -116.9558 |
| ARX(2,1) | 2 | N | N | Y | 90.0731 | -164.1462 | -127.8295 |
| ARX(2,1) | 2 | Y | N | Y | 92.3528 | -162.7057 | -112.7702 |
| ARX(2,1) | 2 | Y | Y | Y | 92.3793 | -160.7587 | -106.2836 |
| ARX(2,1) | 3 | N | N | Y | 100.6740 | -175.3481 | -116.3334 |
| ARX(2,1) | 3 | Y | N | Y | 116.7792 | -195.5584 | -109.3062 |
| ARX(2,2) | 2 | N | N | Y | 90.1563 | -162.3127 | -121.4564 |
| ARX(2,2) | 2 | Y | N | Y | 92.4836 | -160.9672 | -106.4922 |
| ARX(2,2) | 2 | Y | Y | Y | 93.1905 | -158.3810 | -94.8268 |
| ARX(2,2) | 3 | N | N | Y | 100.8116 | -173.6233 | -110.0691 |
| ARX(2,2) | 3 | Y | N | Y | 117.6247 | -195.2495 | -104.4578 |