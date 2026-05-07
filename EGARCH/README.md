
```{raw:typst}
#set page(margin: auto)
```




EGARCH model specification:
$$
 \ln\sigma_{t}^{2}=\omega +\sum_{i=1}^{p}\alpha_{i}  \left(\left|e_{t-i}\right|-\sqrt{2/\pi}\right)
        +\sum_{j=1}^{o}\gamma_{j} e_{t-j} +\sum_{k=1}^{q}\beta_{k}\ln\sigma_{t-k}^{2} 
$$
where  $e_{t}=\epsilon_{t}/\sigma_{t}$