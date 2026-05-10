



+++ {"part": "abstract"}
This book presents computation details of estimating inflation dynamics.
+++

This notebook presents computation details of estimating inflation dynamics.


So far we have covered estimation of 
1. GARCH
2. EGARCH
3. GJR-GARCH
4. SPARCH
5. Regime-switching GARCH
6. BEGE GARCH


State variable definitions:
1. $\pi_t$: inflation.
2. $\mathcal{F}_{t-1}$ : filtration at time t.
3. $SPF_t$: professional forecast observed in $\mathcal{F}_{t-1}$.
4. $\hat{\pi}_t$: expected inflation  $E [\pi_t |\mathcal{F}_{t-1}]$.
5. $u_t$: inflation innovation (residual), defined as $u_t := \pi_t - \hat{\pi}_t$.
6. $u_t^+$ : postive residual, defined as $u_t^+ := u_t \cdot \mathbf{1}(u_t >0)$
7. $u_t^-$ : negative residual, defined as $u_t^- := u_t \cdot \mathbf{1}(u_t \leq 0)$