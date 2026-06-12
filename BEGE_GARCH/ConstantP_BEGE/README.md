```{raw:typst}
#set page(margin: auto)
```

## Constant-p Full BEGE

This specification is an explicitly requested extension of the baseline BEGE menu. It keeps the good-news shape process constant while allowing the bad-news shape process to follow the unrestricted Full BEGE GJR update:

$$
\begin{aligned}
u_t &= \sigma_p \omega_{p,t} - \sigma_n \omega_{n,t},\\
\omega_{p,t} &\sim \tilde{\Gamma}(p_t,1),\qquad
\omega_{n,t}\sim \tilde{\Gamma}(n_t,1),\\
p_t &= p_0,\\
n_t &= n_0 + \rho_n n_{t-1}
      + \frac{\phi_n^+}{2\sigma_n^2}(u_{t-1}^+)^2
      + \frac{\phi_n^-}{2\sigma_n^2}(u_{t-1}^-)^2.
\end{aligned}
$$

The parameter bounds follow the Full BEGE bounds:

- $0 \leq p_0, n_0 < 10$,
- $0 \leq \rho_n \leq 1$,
- $0 \leq \phi_n^+, \phi_n^- \leq 2$,
- $10^{-5} < \sigma_p, \sigma_n < 2$.

The dynamic shape process satisfies

$$
\rho_n + \frac{\phi_n^+ + \phi_n^-}{2} < 1.
$$

The implied variance path $\sigma_p^2 p_t + \sigma_n^2 n_t$ must satisfy the project EWMA lower and upper bounds at every effective-sample observation.

## Batch Estimation

`BEGE_constant_p1.py` estimates the Constant-p Full BEGE specification on the canonical effective sample from `DataSummary/README.md`, **1969Q2--2022Q4** with **215 observations**. The runner uses precomputed lag columns when available, so lagged ARX regressors do not trim the sample again.

The script estimates all four mean-process specifications:

- Constant
- ARX(1,1)
- ARX(2,1)
- ARX(2,2)

Results are checkpointed to `output/raw/draw_###.csv` after each draw.

`collect_constant_p_results.py` merges raw seed files and writes:

- `results/all_estimations.csv`, without standard-error columns or optimizer messages.
- `results/by_mean/constant.csv`, `results/by_mean/ARX_1_1.csv`, `results/by_mean/ARX_2_1.csv`, and `results/by_mean/ARX_2_2.csv`, which split the eligible cleaned estimations by mean process and retain empirical path quantiles.
- `results/selection_diagnostics.csv`, with stored and corrected likelihood criteria plus selection diagnostics.
- `results/path_quantile_diagnostics.csv`, with each estimate's parameters and empirical 5%, median, and 95% quantiles for $p_t$, $n_t$, $\sigma_t^2$, $s_t^2$, and $k_t^2$ from the fixed effective-sample recursion.
- `results/best_loglik_with_se.csv`, with standard errors for the likelihood-best admissible fit.
- `results/best_model.md`, with only the likelihood-best admissible fit.

`SimulationConstantP.sh` defaults to 50 seed jobs, 40 draws per mean process, 25 optimizer starts per draw, and 800 SLSQP iterations per start. This gives 50,000 optimizer starts for each mean process and Constant-p BEGE specification. With all four mean processes, that is 200,000 optimizer starts total.
