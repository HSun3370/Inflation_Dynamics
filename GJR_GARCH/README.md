```{raw:typst}
#set page(margin: auto)
```



# GJR-GARCH Specification

GJR GARCH model specification:
$$
\sigma_{t}^{2}=\omega
        + \sum_{i=1}^{p}\alpha_{i}\left|u_{t-i}\right|^{2}
        +\sum_{j=1}^{o}\gamma_{j}\left|u_{t-j}\right|^{2}
        I\left[u_{t-j}<0\right]+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{2}
$$




# Inflation Dynamics

 

To start the random search of initial mean parameters, I draw uniform samples from $(\mu - 2\sigma,\ \mu + 2\sigma)$ where $\mu$ and $\sigma$ are mean and standard deviation from OLS regression. I also set the AR coefficient bound to avoid the AR process to explode. We have four types of mean processes.

**Table 1: Parameter Bounds for Mean Process Specifications**

| Model     | $c$                         | $\rho_1$            | $\rho_2$            | $\phi_1$       | $\phi_2$       |
|-----------|-----------------------------|---------------------|---------------------|----------------|----------------|
| Constant  | ---                         | ---                 | ---                 | ---            | ---            |
| ARX(1,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-0.999,\ 0.999)$  | ---                 | $(-10,\ 10)$   | ---            |
| ARX(2,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | ---            |
| ARX(2,2)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | $(-10,\ 10)$   |

## II. GARCH Model Summarization

**Table 2: Model selection by AIC: GJR vs EGARCH**

| Distribution   | GJR Mean | GJR Vol | GJR AIC                          | EGARCH Mean | EGARCH Vol | EGARCH AIC |
|----------------|----------|---------|----------------------------------|-------------|------------|------------|
| Normal         | (1,1)    | (2,1)   | 359.4710                         | (1,1)       | (1,1)      | 368.2429   |
| $t$            | (1,1)    | (2,1)   | <span style="color:red">343.1431</span> | (2,1)       | (1,1)      | 344.6254   |
| Skew $t$       | (1,1)    | (2,1)   | 345.1312                         | (2,1)       | (1,1)      | 346.5791   |
| GED            | (1,1)    | (2,1)   | 345.1925                         | (1,1)       | (1,1)      | 347.6934   |
| Mix of Normal  | (1,1)    | (2,1)   | 345.8222                         | (2,1)       | (2,1)      | 343.5023   |

**Table 3: Model selection by BIC: GJR vs EGARCH**

| Distribution   | GJR Mean | GJR Vol | GJR BIC  | EGARCH Mean | EGARCH Vol | EGARCH BIC                       |
|----------------|----------|---------|----------|-------------|------------|----------------------------------|
| Normal         | FC       | (2,1)   | 381.4280 | FC          | (1,1)      | 383.4754                         |
| $t$            | FC       | (1,1)   | 367.8350 | FC          | (1,1)      | <span style="color:red">367.1732</span> |
| Skew $t$       | FC       | (1,1)   | 370.4242 | FC          | (1,1)      | 369.6649                         |
| GED            | FC       | (1,1)   | 371.8489 | FC          | (1,1)      | 370.9644                         |
| Mix of Normal  | FC       | (1,1)   | 373.7106 | FC          | (1,1)      | 371.0689                         |

