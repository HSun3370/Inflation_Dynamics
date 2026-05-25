```{raw:typst}
#set page(margin: auto)
```

# Regime-Switching Inflation Models

 
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

