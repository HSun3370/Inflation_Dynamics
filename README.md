+++ {"part": "abstract"}
This computational notebook documents the data construction, model specifications, estimation workflows, and best-model reporting used to study U.S. quarterly inflation dynamics.
+++
 
This notebook is organized around five pieces.

1. **Data summary.** The data section constructs the effective inflation sample  **1969Q2--2026Q1** (quarterly, **228 observations**) and **1969M2--2026M7** (monthly, **690 observations**) and give statistical summary.  

2. **Mean processes.** The mean-process menu includes Constant, ARX(1,1), ARX(2,1), and ARX(2,2) specifications. These define $\hat{\pi}_t$ and therefore the residual sequence passed to volatility models.

3. **GARCH-family volatility models.** The GARCH section estimates GARCH, GJR-GARCH, and EGARCH specifications with normal, standardized Student's $t$, and  Gaussian-mixture residuals.

4. **Regime-switching models.** The regime-switching section studies state-dependent mean dynamics with normal residuals.

5. **BEGE-GARCH models.** The BEGE section estimates   Constant, Symmetric, InflationDeflation, BadGood, Constant-p Full BEGE, Constant-n Full BEGE, and Full BEGE models.

The following notation is used throughout:
- $\pi_t$: realized inflation.
- $\mathcal{F}_{t-1}$: information set available  at time $t-1$.
- $SPF_{t-1}$: professional forecast of inflation $\pi_t$ based on $\mathcal{F}_{t-1}$.
- $\hat{\pi}_t$: expected inflation of selected mean process, $E[\pi_t | \mathcal{F}_{t-1}]$.
- $u_t$: inflation innovation, defined as $u_t := \pi_t - \hat{\pi}_t$.
- $u_t^+$: positive residual component, $u_t^+ := u_t \cdot \mathbf{1}(u_t > 0)$.
- $u_t^-$: negative residual component, $u_t^- := u_t \cdot \mathbf{1}(u_t \leq 0)$.

