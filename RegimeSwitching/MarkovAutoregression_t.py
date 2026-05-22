"""
Markov switching autoregression with t distribution models

Author: Haoyang Sun

"""


import numpy as np
import statsmodels.base.wrapper as wrap

from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
    markov_switching, markov_regression,markov_autoregression)

from scipy.stats import t, kurtosis
from scipy.special import gammaln #student t distribution



def nu_(data):
    # Check if the input data is a 2D array-like structure
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)     
    nu_values = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        nu = t.fit(data[:, i])[0]
        if nu > 100:
            nu = 100
        nu_values[i] = nu
    return nu_values    
    
class MarkovAutoregression_t(markov_autoregression.MarkovAutoregression):
    r"""
    Markov switching regression with student t distribution model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : int
        The number of regimes.
    order : int
        The order of the autoregressive lag polynomial.
    trend : {'n', 'c', 't', 'ct'}
        Whether or not to include a trend. To include an constant, time trend,
        or both, set `trend='c'`, `trend='t'`, or `trend='ct'`. For no trend,
        set `trend='n'`. Default is a constant.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    exog_tvtp : array_like, optional

    switching_ar : bool or iterable, optional

    switching_trend : bool or iterable, optional

    switching_exog : bool or iterable, optional

    switching_variance : bool, optional
    
    switching_nu : bool, optional

    """

    def __init__(self, endog, k_regimes, order, trend='c', exog=None,
                 exog_tvtp=None, switching_ar=True, switching_trend=True,
                 switching_exog=False, switching_variance=False,switching_nu=False,
                 dates=None, freq=None, missing='none'):
        '''
            Including a new parameter nu 
            Using student t distribution to calculate log likelihood function
        
         '''

        # Properties
        self.switching_nu = switching_nu

        # Initialize the base model
        super().__init__(
            endog, k_regimes, order=order,trend=trend, exog=exog,
            exog_tvtp=exog_tvtp,switching_ar=switching_ar, switching_trend=switching_trend,
            switching_exog=switching_exog,
            switching_variance=switching_variance, dates=dates, freq=freq,
            missing=missing)

       
        # Parameters
        self.parameters['nu'] = [1] if self.switching_nu  else [0]
        
        

    def _conditional_loglikelihoods(self, params):
        """
        Compute loglikelihoods conditional on the current period's regime and
        the last `self.order` regimes.
        """
        # Get the residuals
        resid = self._resid(params)

        # Compute the conditional likelihoods
        variance = params[self.parameters['variance']].squeeze()
        nu = params[self.parameters['nu']].squeeze()
        nu = nu.astype(np.float64)
        variance = variance.astype(np.float64)
        if self.switching_variance:
            variance = np.reshape(variance, (self.k_regimes, 1, 1))
        if self.switching_nu:
            nu = np.reshape(nu, (self.k_regimes, 1, 1))
        #conditional_loglikelihoods = (-0.5 * resid**2 / variance - 0.5 * np.log(2 * np.pi * variance))
        # lls = -0.5 * (log(2 * pi) + log(sigma2) + resids ** 2.0 / sigma2)
        
        #lls = gammaln((nu + 1) / 2) - gammaln(nu / 2) - log(pi * (nu - 2)) / 2
        #lls -= 0.5 * (log(sigma2))
        #lls -= ((nu + 1) / 2) * (log(1 + (resids ** 2.0) / (sigma2 * (nu - 2))))

        conditional_loglikelihoods = (
            gammaln(( nu+ 1) / 2) - gammaln(nu / 2) - np.log(np.pi * (nu - 2)) / 2- 0.5 * (np.log(variance)) 
         - ((nu + 1) / 2) * (np.log(1 + (resid ** 2.0) / (variance * (nu - 2))))
         )
        return conditional_loglikelihoods


    @property
    def _res_classes(self):
        return {'fit': (MarkovAutoregressiontResults,
                        MarkovAutoregressiontResultsWrapper)}

    @property
    def start_params(self):
        """
         Starting parameters for maximum likelihood estimation.
        """
        # Inherited parameters
        params = markov_switching.MarkovSwitching.start_params.fget(self)

        # OLS for starting parameters
        endog = self.endog.copy()
        if self._k_exog > 0 and self.order > 0:
            exog = np.c_[self.exog, self.exog_ar]
        elif self._k_exog > 0:
            exog = self.exog
        elif self.order > 0:
            exog = self.exog_ar

        if self._k_exog > 0 or self.order > 0:
            beta = np.dot(np.linalg.pinv(exog), endog)
            variance = np.var(endog - np.dot(exog, beta))
        else:
            variance = np.var(endog)


        # Regression coefficients
        if self._k_exog > 0:
            if np.any(self.switching_coeffs):
                for i in range(self.k_regimes):
                    params[self.parameters[i, 'exog']] = (
                        beta[:self._k_exog] * (i / self.k_regimes))
            else:
                params[self.parameters['exog']] = beta[:self._k_exog]

        # Autoregressive
        if self.order > 0:
            if np.any(self.switching_ar):
                for i in range(self.k_regimes):
                    params[self.parameters[i, 'autoregressive']] = (
                        beta[self._k_exog:] * (i / self.k_regimes))
            else:
                params[self.parameters['autoregressive']] = beta[self._k_exog:]

        # Variance
        if self.switching_variance:
            params[self.parameters['variance']] = (
                np.linspace(variance / 10., variance, num=self.k_regimes))
        else:
            params[self.parameters['variance']] = variance
            
        # nu
        if self.switching_nu:
            params[self.parameters['nu']] = (
                4+2*np.cumsum(np.random.rand(self.k_regimes))
                )
        else:
            params[self.parameters['nu']] = 5           

        return params
    
    def _em_nu(self, result, endog, exog, betas, tmp=None):
        """
        EM step for nu
        
        I use residuals to estimate nu in each iteration
        """
        k_exog = 0 if exog is None else exog.shape[1]

        if self.switching_nu:
            nu = np.zeros(self.k_regimes)
            for i in range(self.k_regimes):
                if k_exog > 0:
                    resid = endog - np.dot(exog, betas[i])
                else:
                    resid = endog
                #variance = (
                #    np.sum(resid**2 *
                #           result.smoothed_marginal_probabilities[i]) /
                #    np.sum(result.smoothed_marginal_probabilities[i]))
                        
                nu[i] = nu_(resid )#/ np.sqrt(variance)) 
            
        else:
            nu = 5
            #variance=0
            if tmp is None:
                tmp = np.sqrt(result.smoothed_marginal_probabilities)
            for i in range(self.k_regimes):
                tmp_endog = tmp[i] * endog
                if k_exog > 0:
                    tmp_exog = tmp[i][:, np.newaxis] * exog
                    resid = tmp_endog - np.dot(tmp_exog, betas[i])
                else:
                    resid = tmp_endog
                #variance += np.sum(resid**2)

            nu = nu_(resid) #/ np.sqrt(variance))
        return nu
    
    def _em_iteration(self, params0):
        """
            EM iteration
         """
        # Inherited parameters
        result, params1 = markov_switching.MarkovSwitching._em_iteration(
            self, params0)

        tmp = np.sqrt(result.smoothed_marginal_probabilities)

        # Regression coefficients
        coeffs = None
        if self._k_exog > 0:
            coeffs = self._em_exog(result, self.endog, self.exog,
                                   self.parameters.switching['exog'], tmp)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'exog']] = coeffs[i]

        # Autoregressive
        if self.order > 0:
            if self._k_exog > 0:
                ar_coeffs, variance = self._em_autoregressive(
                    result, coeffs)
            else:
                ar_coeffs = self._em_exog(
                    result, self.endog, self.exog_ar,
                    self.parameters.switching['autoregressive'])
                variance = self._em_variance(
                    result, self.endog, self.exog_ar, ar_coeffs, tmp)
                
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'autoregressive']] = ar_coeffs[i]
            params1[self.parameters['variance']] = variance
            params1[self.parameters['nu']] =params0[self.parameters['nu']] 
            #params1[self.parameters['nu']] = self._em_nu( result, self.endog, self.exog_ar, ar_coeffs, tmp)
        return result, params1
    
    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        # Inherited parameters
        param_names = np.array(
            markov_autoregression.MarkovAutoregression.param_names.fget(self),
            dtype=object)

        # nu
        if self.switching_nu:
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'nu']] = 'nu[%d]' %i
        else:
            param_names[self.parameters['nu']] = 'nu'

        return param_names.tolist()

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.
        """
        # Inherited parameters
        constrained = super().transform_params(unconstrained)

        # Autoregressive
        # TODO may provide unexpected results when some coefficients are not
        # switching

        constrained[self.parameters['nu']] = (2.05+np.exp(unconstrained[self.parameters['nu']]))

        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        # Inherited parameters
        unconstrained = super().untransform_params(
            constrained)

        # Autoregressive
        # TODO may provide unexpected results when some coefficients are not
        # switching
        unconstrained[self.parameters['nu']]=(np.log(constrained[self.parameters['nu']] -2.05 )) 
        
        return unconstrained


class MarkovAutoregressiontResults(markov_autoregression.MarkovAutoregressionResults):
    r"""
    Class to hold results from fitting a Markov switching autoregression model

    Parameters
    ----------
    model : MarkovAutoregression instance
        The fitted model instance
    params : ndarray
        Fitted parameters
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    cov_type : str
        The type of covariance matrix estimator to use. Can be one of 'approx',
        'opg', 'robust', or 'none'.

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    """
    pass


class MarkovAutoregressiontResultsWrapper(
        markov_autoregression.MarkovAutoregressionResultsWrapper):
    pass
wrap.populate_wrapper(MarkovAutoregressiontResultsWrapper,  # noqa:E305
                      MarkovAutoregressiontResults)




