```{raw:typst}
#set page(margin: auto)
```

# Regime-Switching Inflation Models

This folder estimates regime-switching models using the project effective sample and pre-created lag variables.


## Mean Process Specifications

I estimate all four mean-process specifications from `MeanProcess/README.md`.

### 1) Constant

$$
\pi_{t} = SPF_{t} + u_t
$$
 
with regime-switching distributional parameters.

### 2) ARX(1,1)

$$
\pi_t = \rho_1 \pi_{t-1} + \phi_1 SPF_t + u_t
$$

### 3) ARX(2,1)

$$
\pi_t = \rho_1 \pi_{t-1} + \rho_2 \pi_{t-2} + \phi_1 SPF_t + u_t
$$

### 4) ARX(2,2)

$$
\pi_t = \rho_1 \pi_{t-1} + \rho_2 \pi_{t-2} + \phi_1 SPF_t + \phi_2 SPF_{t-1} + u_t
$$

## Regime-Switching Structure

Let latent state be

$$
s_t \in \{1,\dots,K\}, \qquad \Pr(s_t=j | s_{t-1}=i)=p_{ij}.
$$

Switching settings evaluated:

| Error Distribution | #Regime | Switching AR | Switching SPF | Switching Distribution |
|--------------------|---------|--------------|---------------|------------------------|
| Normal / Student t | 2       | Y            | Y             | Y                      |
| Normal / Student t | 2       | Y            | N             | Y                      |
| Normal / Student t | 2       | N            | N             | Y                      |
| Normal / Student t | 3       | Y            | N             | Y                      |
| Normal / Student t | 3       | N            | N             | Y                      |

For Student's t, `nu` is estimated and allowed to switch by regime.

