
```{raw:typst}
#set page(margin: auto)
```



#  BEGE Density

BEGE density function $f_{BEGE}(\mu | p, n, \sigma_p, \sigma_n)$ is the function that calculates the density of the observation $\mu$ given the parameters $\{p, n, \sigma_p, \sigma_n\}$.

To compare Justin's analytic BEGE density code with the old numerical one, I conducted an experiment using synthetic data. I generated $200$ independent observations from a standard normal distribution and fixed the BEGE parameters to

$$
p = 5, \qquad n = 7, \qquad \sigma_p = 0.8, \qquad \sigma_n = 1.2.
$$

I computed the sum of log-likelihood using two approaches:

1. The *old BEGE function*, which approximates the density via numerical integration on a uniform grid, and whose accuracy depends strongly on the number of grid points; and
2. *Justin's analytic BEGE function*.

The old numerical method was evaluated over a wide range of grid resolutions (up to $50{,}000$ points), and the resulting log-likelihood sums were plotted alongside the benchmark value computed from Justin's code.

![BEGE LLF Comparison](BEGELLF_comparison.png)

**Findings:**

- The old numerical likelihood converges toward Justin's analytic likelihood as the number of grid points increases, but the convergence is slow and requires very fine grids.
- For small or moderate grid sizes, the numerical integration *overestimates* the log-likelihood. This explains why, for a given set of parameters, Justin's implementation produces a lower log-likelihood than the old code.
- Justin's analytic BEGE formula provides a more accurate and more computationally efficient evaluation.


# BEGE GARCH Family

To start the random search of initial mean parameters, I draw uniform samples from $(\mu - 2\sigma,\ \mu + 2\sigma)$ where $\mu$ and $\sigma$ are mean and standard deviation from OLS regression. I also set the AR coefficient bound to avoid the AR process to explode. We have four types of mean processes.

**Table 1: Parameter Bounds for Mean Process Specifications**

| Model     | $c$                         | $\rho_1$            | $\rho_2$            | $\phi_1$       | $\phi_2$       |
|-----------|-----------------------------|---------------------|---------------------|----------------|----------------|
| Constant  | ---                         | ---                 | ---                 | ---            | ---            |
| ARX(1,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-0.999,\ 0.999)$  | ---                 | $(-10,\ 10)$   | ---            |
| ARX(2,1)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | ---            |
| ARX(2,2)  | $(\min \pi_t,\ \max \pi_t)$ | $(-1.999,\ 1.999)$  | $(-0.999,\ 0.999)$  | $(-10,\ 10)$   | $(-10,\ 10)$   |
 