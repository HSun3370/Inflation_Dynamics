+++ {"part": "abstract"}
This computational notebook documents the data construction, model specifications, estimation workflows, and best-model reporting used to study U.S. quarterly inflation dynamics.
+++
 
This project studies U.S. quarterly inflation dynamics by combining survey-based inflation expectations with several conditional-volatility models for inflation innovations. The notebook is organized around five linked pieces.

1. **Data summary.** The data section constructs the effective inflation sample  **1969Q2--2022Q4**, with **215 observations** and give statistical summary.  

2. **Mean processes.** The mean-process menu includes Constant, ARX(1,1), ARX(2,1), and ARX(2,2) specifications. These define $\hat{\pi}_t$ and therefore the residual sequence passed to volatility models.

3. **GARCH-family volatility models.** The GARCH section estimates GARCH, GJR-GARCH, and EGARCH specifications with normal, standardized Student's $t$, and  Gaussian-mixture innovations.

4. **Regime-switching GARCH.** The regime-switching section studies state-dependent mean  dynamics with normal and standardized Student's $t$ innovations.

5. **BEGE-GARCH models.** The BEGE section estimates the baseline volatility specifications: Constant, Symmetric, InflationDeflation, BadGood and Full BEGE models. It also includes the requested Constant-p Full BEGE and Constant-n Full BEGE extensions.

Notations:
- $\pi_t$: realized inflation.
- $\mathcal{F}_{t-1}$: information set available before inflation at time $t$ is realized.
- $SPF_{t-1}$: professional forecast of inflation based on $\mathcal{F}_{t-1}$.
- $\hat{\pi}_t$: expected inflation, $E[\pi_t | \mathcal{F}_{t-1}]$.
- $u_t$: inflation innovation, defined as $u_t := \pi_t - \hat{\pi}_t$.
- $u_t^+$: positive residual component, $u_t^+ := u_t \cdot \mathbf{1}(u_t > 0)$.
- $u_t^-$: negative residual component, $u_t^- := u_t \cdot \mathbf{1}(u_t \leq 0)$.

The rest of the notebook uses this notation consistently across data summaries, likelihood definitions, estimation scripts, and reported best-model equations.
