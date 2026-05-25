```{raw:typst}
#set page(margin: auto)
```

# Regime-Switching Inflation Models


Let the latent state be
$$
s_t \in \{1,\dots,K\}, \qquad \Pr(s_t = j | s_{t-1} = i) = p_{ij}.
$$

I estimate several specifications that differ in **which blocks of parameters are allowed to switch with $s_t$**. Taking the ARX(2,2) model as an example,
$$
\pi_{t+1} = c + \rho_1 \pi_t + \rho_2 \pi_{t-1} + \phi_1\, \mathrm{SPF}_t + \phi_2\, \mathrm{SPF}_{t-1} + \mu_{t+1},
$$
the parameters are grouped into three blocks, and each block can independently be made regime-dependent:

| Block | State-dependent parameters |
|---|---|
| AR component | $c_{s_t},\ \rho_{1,s_t},\ \rho_{2,s_t}$ |
| SPF component | $\phi_{1,s_t},\ \phi_{2,s_t}$ |
| Shock distribution | $\sigma_{s_t},\ \nu_{s_t}$ |

where $\nu_{s_t}$ denotes the degrees of freedom when assuming standardized Student-$t$ distribution for residuals. A given specification is defined by selecting any subset of these three blocks to be switching; parameters in blocks not selected are held constant across regimes.