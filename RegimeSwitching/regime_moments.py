"""Gaussian predictive-mixture moments and their distributions over sample dates.

Reporting only: predicted state probabilities are used, never smoothed weights.
Kurtosis is Pearson kurtosis (a Gaussian has 3), not excess kurtosis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gaussian_mixture_moments(weights, means, variances):
    """Rows are dates; columns are regimes. Return mean and central moments."""
    w, m, v = np.broadcast_arrays(np.asarray(weights, float), np.asarray(means, float),
                                   np.asarray(variances, float))
    if w.ndim != 2 or not np.all(np.isfinite(w+m+v)) or np.any(w < 0) or np.any(v <= 0):
        raise ValueError('Mixture inputs must be finite with valid weights and positive variances')
    if not np.allclose(w.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError('Regime probabilities must sum to one')
    w = w/w.sum(axis=1, keepdims=True)
    # Center relative to a component to avoid cancellation for large common means.
    origin = m[:, :1]
    shifts = m-origin
    mean_shift = (w*shifts).sum(axis=1)
    mean = origin[:, 0]+mean_shift
    delta = shifts-mean_shift[:, None]
    c2 = (w*(v+delta**2)).sum(axis=1)
    c3 = (w*(delta**3+3*delta*v)).sum(axis=1)
    c4 = (w*(delta**4+6*delta**2*v+3*v**2)).sum(axis=1)
    return pd.DataFrame(dict(mean=mean, variance=c2, third_central=c3, fourth_central=c4,
                             skewness=c3/c2**1.5, kurtosis=c4/c2**2))


def average_predictive_moments(path, centered=False):
    """Moments of a uniform mixture over dates, optionally of centered residuals."""
    mean = 0.0 if centered else float(path['mean'].mean())
    shift = np.zeros(len(path)) if centered else path['mean'].to_numpy()-mean
    v = path.variance.to_numpy()
    c3 = path.third_central.to_numpy()
    c4 = path.fourth_central.to_numpy()
    variance = np.mean(v+shift**2)
    third = np.mean(c3+3*shift*v+shift**3)
    fourth = np.mean(c4+4*shift*c3+6*shift**2*v+shift**4)
    return dict(mean=mean, variance=variance, skewness=third/variance**1.5,
                kurtosis=fourth/variance**2)


def predictive_moment_path(result, inflation_offset=0.0):
    model = result.model
    params = np.asarray(result.params, float)
    means = np.asarray(model.predict_conditional(params), float)[:, 0, :].T
    means = means+np.broadcast_to(np.asarray(inflation_offset), (model.nobs,))[:, None]
    weights = np.asarray(result.predicted_marginal_probabilities, float)
    variances = np.array([params[model.parameters[i,'variance']][0] for i in range(model.k_regimes)])
    if means.shape != weights.shape:
        raise ValueError('Predictive means and probabilities do not align')
    return gaussian_mixture_moments(weights, means, variances)


def stationary_shock_moments(result):
    p = np.asarray(result.regime_transition[:, :, 0], float).T
    k = len(p)
    q = np.linalg.lstsq(np.vstack([p.T-np.eye(k), np.ones(k)]),
                         np.r_[np.zeros(k), 1.], rcond=None)[0]
    if q.min() < -1e-12:
        raise ValueError('Invalid stationary probabilities')
    q = np.maximum(q, 0)
    q /= q.sum()
    params = np.asarray(result.params, float)
    variances = np.array([params[result.model.parameters[i,'variance']][0] for i in range(k)])
    return gaussian_mixture_moments(q[None,:], np.zeros((1,k)), variances).iloc[0], q


def markdown_table(frame):
    header='| '+' | '.join(map(str, frame.columns))+' |'
    sep='| '+' | '.join(['---']*len(frame.columns))+' |'
    rows=['| '+' | '.join(str(v).replace('|',r'\|') for v in row)+' |'
          for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header,sep,*rows])


def fmt(x):
    return f'{0.0 if abs(x)<0.00005 else x:.4f}'


def build_moment_section(result, data, out_dir, anchored=False):
    out_dir = Path(out_dir)
    path = predictive_moment_path(result, data.SPF.to_numpy() if anchored else 0.0)
    path.index = data.index
    path.index.name='period'
    path.to_csv(out_dir/'best_model_conditional_moments.csv')
    metrics=['mean','variance','skewness','kurtosis']
    summary=[dict(distribution='Inflation: sample-averaged predictive mixture',
                  **average_predictive_moments(path)),
             dict(distribution='Residual: sample-averaged predictive mixture',
                  **average_predictive_moments(path, centered=True))]
    shock, q = stationary_shock_moments(result)
    summary.append(dict(distribution='Regime shock: stationary mixture',
                        **{name:shock[name] for name in metrics}))
    table=pd.DataFrame(summary)
    table.to_csv(out_dir/'best_model_moment_summary.csv',index=False)
    pd.DataFrame({'regime':np.arange(len(q)),'stationary_probability':q}).to_csv(
        out_dir/'best_model_stationary_probabilities.csv',index=False)
    quantiles=pd.DataFrame([
        dict(moment=metric, average=path[metric].mean(), q05=path[metric].quantile(.05),
             q25=path[metric].quantile(.25), median=path[metric].quantile(.5),
             q75=path[metric].quantile(.75), q95=path[metric].quantile(.95))
        for metric in metrics])
    quantiles.to_csv(out_dir/'best_model_conditional_moment_quantiles.csv',index=False)
    figure, axes=plt.subplots(2,2,figsize=(10,6.6),constrained_layout=True)
    labels=['Conditional mean (percentage points)','Conditional variance (percentage points squared)',
            'Conditional skewness','Conditional kurtosis (Gaussian = 3)']
    for ax, metric, label in zip(axes.flat, metrics, labels):
        values=np.sort(path[metric].to_numpy())
        unique,counts=np.unique(values,return_counts=True)
        if len(unique)==1:
            ax.axvline(unique[0],color='#245b86',linewidth=2)
            ax.set_xlim(unique[0]-.1,unique[0]+.1)
            ax.text(.5,.15,'All dates have the same value',ha='center',transform=ax.transAxes,fontsize=9)
        else:
            pad=.025*(unique[-1]-unique[0])
            ax.step(np.r_[unique[0]-pad,unique,unique[-1]+pad],
                    np.r_[0,np.cumsum(counts)/len(values),1],where='post',color='#245b86',linewidth=1.8)
        ax.set_xlabel(label,fontsize=9)
        ax.set_ylabel('Fraction of sample dates')
        ax.set_ylim(0,1)
        ax.grid(alpha=.22)
    figure.savefig(out_dir/'best_model_conditional_moment_ecdfs.png',dpi=180)
    figure.savefig(out_dir/'best_model_conditional_moment_ecdfs.pdf')
    plt.close(figure)
    display=table.copy()
    for key in metrics:
        display[key]=display[key].map(fmt)
    display.columns=['Distribution','Mean','Variance','Skewness','Kurtosis']
    quantile_display=quantiles.copy()
    for key in quantile_display.columns[1:]:
        quantile_display[key]=quantile_display[key].map(fmt)
    quantile_display.columns=['Conditional moment','Average','5%','25%','Median','75%','95%']
    explanation = (
        'The mean is shared across regimes in this fit, so conditional skewness is exactly zero '
        'at every date. Variance mixing can still produce kurtosis above three. '
        'Inflation averaged over dates can be skewed because its conditional mean varies over time.'
        if not np.any(result.model.parameters.switching['exog']) else
        'The regime means differ, so the predictive distribution can be skewed even though each '
        'regime is Gaussian. Its variance includes uncertainty about the regime mean.')
    return '\n\n'.join([
        '## Unconditional and Sample-Averaged Moments',
        'The first two rows average the fitted predictive distributions uniformly over the '
        'observed sample. They are not stationary long-run moments of inflation: the model '
        'does not specify the joint stationary law of inflation lags and SPF. '
        'The third row is the stationary distribution of the regime shock '
        '$\\varepsilon_{t+1}=\\sigma_iz_{t+1}$, weighted by $\\bar Q$. '
        'When means switch, this shock differs from the forecast residual '
        '$u_{t+1}=\\pi_{t+1}-\\hat{\\pi}_{t+1}$.',
        markdown_table(display),
        'Mean is measured in percentage points per period and variance in squared percentage '
        'points. Skewness and kurtosis are dimensionless. Kurtosis is Pearson kurtosis '
        '(Gaussian = 3); excess kurtosis is this value minus 3.',
        '## Empirical Distributions of Conditional Moments',
        'Each observation contributes one value of the predictive mean, variance, skewness, '
        'and kurtosis, calculated using $Q_t$ before observing inflation. The table and '
        'empirical CDFs describe variation over dates, not sampling uncertainty about parameters. '
        'The formulas are given in the regime-switching methodology section.',
        markdown_table(quantile_display), explanation,
        'The average conditional skewness or kurtosis generally differs from the skewness or '
        'kurtosis of the sample-averaged distribution above.',
        '![Empirical CDFs of the four conditional moments](best_model_conditional_moment_ecdfs.png)',
    ])
