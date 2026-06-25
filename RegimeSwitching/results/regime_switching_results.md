```{raw:typst}
#set page(margin: auto)
```

Each model is selected from 50 starts; only estimates that passed all checks (optimizer convergence, AR stationarity, parameter bounds, and valid transition probabilities) are reported. Specifications where no start passed all checks are excluded. Student-$t$ degrees of freedom ($\nu$) are held common across regimes; `Sw.Dist` reflects whether $\sigma^2$ is regime-specific.

| Mean | Dist | K | Sw.AR | Sw.SPF | Sw.Dist | LogLik | AIC | BIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constant | Normal | 2 | N | N | Y | -186.6300 | 381.2600 | 394.7425 |
| Constant | Normal | 3 | N | N | Y | -180.6569 | 379.3139 | 409.6496 |
| Constant | Student t | 2 | N | N | Y | -185.7189 | 381.4378 | 398.2910 |
| Constant | Student t | 3 | N | N | Y | -180.6407 | 381.2814 | 414.9877 |
| ARX(1,1) | Normal | 2 | N | N | Y | -177.1889 | 368.3779 | 391.9723 |
| ARX(1,1) | Normal | 2 | Y | N | Y | -175.1175 | 368.2350 | 398.5708 |
| ARX(1,1) | Normal | 2 | Y | Y | Y | -175.0924 | 370.1848 | 403.8912 |
| ARX(1,1) | Normal | 3 | N | N | Y | -174.3093 | 372.6186 | 413.0662 |
| ARX(1,1) | Normal | 3 | Y | N | Y | -162.3751 | 356.7503 | 410.6805 |
| ARX(1,1) | Student t | 2 | N | N | Y | -175.5345 | 367.0689 | 394.0340 |
| ARX(1,1) | Student t | 2 | Y | N | Y | -166.6098 | 353.2196 | 386.9259 |
| ARX(1,1) | Student t | 2 | Y | Y | Y | -166.4669 | 354.9338 | 392.0108 |
| ARX(1,1) | Student t | 3 | Y | N | Y | -159.0882 | 352.1764 | 409.4772 |
| ARX(2,1) | Normal | 2 | N | N | Y | -176.2156 | 368.4311 | 395.3962 |
| ARX(2,1) | Normal | 2 | Y | N | Y | -174.9422 | 371.8845 | 408.9615 |
| ARX(2,1) | Normal | 2 | Y | Y | Y | -174.3768 | 372.7535 | 413.2012 |
| ARX(2,1) | Normal | 3 | Y | N | Y | -157.6865 | 353.3729 | 417.4150 |
| ARX(2,1) | Student t | 2 | N | N | Y | -173.4411 | 364.8823 | 395.2180 |
| ARX(2,1) | Student t | 2 | Y | N | Y | -162.6866 | 349.3733 | 389.8209 |
| ARX(2,1) | Student t | 2 | Y | Y | Y | -161.4838 | 348.9676 | 392.7859 |
| ARX(2,1) | Student t | 3 | Y | N | Y | -147.8112 | 335.6223 | 403.0351 |
| ARX(2,2) | Normal | 2 | N | N | Y | -175.9610 | 369.9221 | 400.2578 |
| ARX(2,2) | Normal | 2 | Y | N | Y | -174.5337 | 373.0674 | 413.5150 |
| ARX(2,2) | Normal | 2 | Y | Y | Y | -174.1552 | 376.3105 | 423.4994 |
| ARX(2,2) | Student t | 2 | N | N | Y | -173.2563 | 366.5127 | 400.2190 |
| ARX(2,2) | Student t | 2 | Y | N | Y | -162.6810 | 351.3619 | 395.1802 |
| ARX(2,2) | Student t | 2 | Y | Y | Y | -159.4737 | 348.9474 | 399.5070 |
| ARX(2,2) | Student t | 3 | N | N | Y | -173.2563 | 376.5126 | 427.0722 |