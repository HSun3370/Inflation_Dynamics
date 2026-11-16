```{raw:typst}
#set page(margin: auto)
```

# Restricted Full BEGE: comparison

Both restrictions retain the full $n_t$ recursion and estimate nine volatility parameters, one fewer than Full BEGE. Every comparison below holds the mean process, sample, density evaluator, and shape initialization convention fixed. Each restricted search uses 10,000 random starts per mean and frequency.

$\Delta\ell$, $\Delta\mathrm{AIC}$ and $\Delta\mathrm{BIC}$ denote the restricted-model value minus the currently reported Full BEGE value for the same mean. Negative criterion differences favor the restriction. Specifically, $\Delta\mathrm{AIC}=-2\Delta\ell-2$ and $\Delta\mathrm{BIC}=-2\Delta\ell-\log N$. “Symmetric-p” and “Positive-only-p” in the table both retain full n dynamics.

## Quarterly

Sample: 1969Q2--2026Q3; $N=230$.

| Mean | Restriction | LogLik | $\Delta\ell$ | $\Delta$AIC | $\Delta$BIC |
|---|---|---:|---:|---:|---:|
| Constant | Symmetric-p | -178.6250 | -0.0723 | -1.8555 | -5.2936 |
| ARX(1,1) | Symmetric-p | -172.3897 | -0.8028 | -0.3945 | -3.8326 |
| ARX(2,1) | Symmetric-p | -171.1773 | -0.4921 | -1.0159 | -4.4540 |
| ARX(2,2) | Symmetric-p | -170.3622 | -1.3923 | 0.7846 | -2.6535 |
| Constant | Positive-only-p | -178.1481 | 0.4046 | -2.8092 | -6.2473 |
| ARX(1,1) | Positive-only-p | -173.0200 | -1.4331 | 0.8662 | -2.5718 |
| ARX(2,1) | Positive-only-p | -171.4193 | -0.7340 | -0.5319 | -3.9700 |
| ARX(2,2) | Positive-only-p | -171.3853 | -2.4154 | 2.8309 | -0.6072 |



**Symmetric-p/full-n:** lower AIC than Full BEGE in 3 of the four means and lower BIC in 4. Its lowest AIC is 368.3545 with ARX(2,1).

**Positive-only-p/full-n:** lower AIC than Full BEGE in 2 of the four means and lower BIC in 4. Its lowest AIC is 368.8385 with ARX(2,1).

Some restricted fits exceed the saved Full BEGE likelihood. Because the restricted parameter vectors are feasible for Full BEGE, this identifies an incomplete unrestricted numerical search. These differences compare saved fits and must not be interpreted as likelihood-ratio tests at global maxima.

An archived, admissible Full BEGE ARX(2,2) fit has log likelihood −168.8770, compared with −168.9699 in the current reported search. Using that archived benchmark increases the ARX(2,2) criterion differences above by 0.1858. Reconciliation of the unrestricted archived searches remains separate from these newly completed restricted searches.

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

## Monthly

Sample: 1969M2--2026M9; $N=692$.

| Mean | Restriction | LogLik | $\Delta\ell$ | $\Delta$AIC | $\Delta$BIC |
|---|---|---:|---:|---:|---:|
| Constant | Symmetric-p | 50.2694 | -7.7229 | 13.4458 | 8.9062 |
| ARX(1,1) | Symmetric-p | 104.4706 | -2.2766 | 2.5531 | -1.9865 |
| ARX(2,1) | Symmetric-p | 105.2829 | -2.2274 | 2.4548 | -2.0847 |
| ARX(2,2) | Symmetric-p | 106.0133 | -2.0549 | 2.1099 | -2.4297 |
| Constant | Positive-only-p | 45.3225 | -12.6698 | 23.3397 | 18.8001 |
| ARX(1,1) | Positive-only-p | 106.7471 | -0.0000 | -1.9999 | -6.5395 |
| ARX(2,1) | Positive-only-p | 107.5095 | -0.0008 | -1.9983 | -6.5379 |
| ARX(2,2) | Positive-only-p | 108.0674 | -0.0008 | -1.9984 | -6.5380 |



**Symmetric-p/full-n:** lower AIC than Full BEGE in 0 of the four means and lower BIC in 3. Its lowest AIC is -184.9413 with ARX(1,1).

**Positive-only-p/full-n:** lower AIC than Full BEGE in 3 of the four means and lower BIC in 3. Its lowest AIC is -189.4943 with ARX(1,1).

The model-specific pages select the likelihood-best estimate across the four reported means and provide parameter standard errors and path quantiles. The comparison above also shows criterion differences within each mean; likelihood-based selection across means need not coincide with AIC- or BIC-based selection. Boundary coefficients and rough standard errors do not support ordinary Wald significance claims.
