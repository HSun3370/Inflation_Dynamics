```{raw:typst}
#set page(margin: auto)
```

Each model is selected from 50 starts; only estimates that passed all checks (optimizer convergence, AR stationarity, parameter bounds, and valid transition probabilities) are reported. Specifications where no start passed all checks are excluded. `Sw.Dist` reflects whether $\sigma^2$ is regime-specific.

| Mean | K | Sw.AR | Sw.SPF | Sw.Dist | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | 2 | N | N | Y | -186.6300 | 381.2600 | 394.7425 |
| Constant | 3 | N | N | Y | -180.6569 | 379.3139 | 409.6496 |
| ARX(1,1) | 2 | N | N | Y | -177.1889 | 368.3779 | 391.9723 |
| ARX(1,1) | 2 | Y | N | Y | -175.1175 | 368.2350 | 398.5708 |
| ARX(1,1) | 2 | Y | Y | Y | -175.0924 | 370.1848 | 403.8912 |
| ARX(1,1) | 3 | N | N | Y | -174.3093 | 372.6186 | 413.0662 |
| ARX(1,1) | 3 | Y | N | Y | -162.3751 | 356.7503 | 410.6805 |
| ARX(2,1) | 2 | N | N | Y | -176.2156 | 368.4311 | 395.3962 |
| ARX(2,1) | 2 | Y | N | Y | -174.9422 | 371.8845 | 408.9615 |
| ARX(2,1) | 2 | Y | Y | Y | -174.3768 | 372.7535 | 413.2012 |
| ARX(2,1) | 3 | Y | N | Y | -157.6865 | 353.3729 | 417.4150 |
| ARX(2,2) | 2 | N | N | Y | -175.9610 | 369.9221 | 400.2578 |
| ARX(2,2) | 2 | Y | N | Y | -174.5337 | 373.0674 | 413.5150 |
| ARX(2,2) | 2 | Y | Y | Y | -174.1552 | 376.3105 | 423.4994 |