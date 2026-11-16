"""Check the notes' recursions by enumerating latent paths, independent of EM."""
from itertools import product
import unittest
import numpy as np
from scipy.special import logsumexp
import estimate_regime_switching_models as rs
from audit_derivation import forward_backward, model_inputs, check_sharing
from test_estimation_methodology import synthetic_data


def enumerate_paths(densities, transition, initial):
    n, k = densities.shape
    paths = np.array(list(product(range(k), repeat=n)))
    weights = np.log(initial[paths[:,0]]) + densities[0,paths[:,0]]
    for t in range(1,n):
        weights += np.log(transition[paths[:,t-1],paths[:,t]]) + densities[t,paths[:,t]]
    llf = logsumexp(weights)
    weights = np.exp(weights-llf)
    marginal = np.array([[weights[paths[:,t]==i].sum() for i in range(k)] for t in range(n)])
    pairs = np.array([[[weights[(paths[:,t]==i)&(paths[:,t+1]==j)].sum()
                       for j in range(k)] for i in range(k)] for t in range(n-1)])
    return llf, marginal, pairs


class DerivationTests(unittest.TestCase):
    def test_every_block_configuration_matches_exhaustive_path_sums(self):
        specs, _ = rs.build_model_specs()
        self.assertEqual(len(specs), 17)
        self.assertTrue(all(s.switch_spec.switching_distribution for s in specs))
        mean = rs.mean_processes()[-1]
        # Illustrative restrictions, deliberately outside the production menu.
        specs += [rs.ModelSpec(mean, 'Normal', rs.SwitchingSpec(k, True, False, False))
                  for k in [2,3]]
        data = synthetic_data(5)
        for spec in specs:
            with self.subTest(model=rs.model_label(spec), variance_switch=spec.switch_spec.switching_distribution):
                model, _, _ = rs.build_model(data, spec)
                check_sharing(model, spec)
                params = model.start_params.copy()
                k = model.k_regimes
                transition = (np.array([[.86,.14],[.23,.77]]) if k==2 else
                              np.array([[.82,.13,.05],[.10,.74,.16],[.21,.08,.71]]))
                params[model.parameters['regime_transition']] = transition.T[:-1].ravel()
                for i in range(k):
                    for j, index in enumerate(model.parameters[i,'exog']):
                        params[index] = .08*(j+1) + .15*i
                params[model.parameters['variance']] = ([.09,.7] if k==2 else [.09,.3,.7]) if spec.switch_spec.switching_distribution else .3
                result = model.smooth(params, cov_type='none')
                density, p, prior, _, _ = model_inputs(model,params)
                ref = forward_backward(density,p,prior)
                llf,s,joint = enumerate_paths(density,p,prior)
                self.assertAlmostEqual(ref['loglike'],llf,places=11)
                self.assertAlmostEqual(result.llf,llf,places=11)
                np.testing.assert_allclose(ref['S'],s,atol=2e-12)
                np.testing.assert_allclose(ref['J'],joint,atol=2e-12)
                np.testing.assert_allclose(result.smoothed_marginal_probabilities,s,atol=2e-12)
                np.testing.assert_allclose(ref['Q'],result.predicted_marginal_probabilities,atol=2e-12)
                np.testing.assert_allclose(ref['R'],result.filtered_marginal_probabilities,atol=2e-12)
                np.testing.assert_allclose(joint.sum(axis=2),s[:-1],atol=2e-12)
                np.testing.assert_allclose(joint.sum(axis=1),s[1:],atol=2e-12)
                for t in range(len(data)):
                    _, prefix, _ = enumerate_paths(density[:t+1],p,prior)
                    np.testing.assert_allclose(ref['R'][t],prefix[-1],atol=2e-12)

    def test_arx22_block_split_preserves_filter_and_smoother(self):
        spec=rs.ModelSpec(rs.mean_processes()[-1],'Normal',rs.SwitchingSpec(2,True,False))
        data=synthetic_data(20)
        model,_,_=rs.build_model(data,spec)
        params=model.start_params.copy()
        for i in range(2):
            params[model.parameters[i,'exog']]=[.2+.1*i,.3-.1*i,.05+.01*i,.4,.15]
        params[model.parameters['variance']]=[.2,.8]
        densities,p,prior,means,variances=model_inputs(model,params)
        x_s=np.column_stack([np.ones(len(data)),data[['Inflation_lag_1','Inflation_lag_2']].to_numpy()])
        x_n=data[['SPF','SPF_lag_1']].to_numpy()
        adjusted=data.Inflation.to_numpy()-x_n@np.array([.4,.15])
        split=np.column_stack([adjusted-x_s@np.array([.2+.1*i,.3-.1*i,.05+.01*i]) for i in range(2)])
        split_density=-.5*(np.log(2*np.pi*variances)+split**2/variances)
        np.testing.assert_allclose(densities,split_density,atol=1e-13)
        full=forward_backward(densities,p,prior)
        partial=forward_backward(split_density,p,prior)
        for key in ['Q','R','S','J']:
            np.testing.assert_allclose(full[key],partial[key],atol=1e-13)

    def test_identical_densities_convey_no_regime_information(self):
        p=np.array([[.9,.1],[.2,.8]])
        prior=np.array([2/3,1/3])
        density=np.repeat(np.array([-.2,-1.5,-.7])[:,None],2,axis=1)
        ref=forward_backward(density,p,prior)
        for name in ['Q','R','S']:
            np.testing.assert_allclose(ref[name],np.tile(prior,(3,1)),atol=1e-14)
        self.assertAlmostEqual(ref['loglike'],density[:,0].sum())

    def test_joint_em_variance_updates_follow_smoothed_weight_formula(self):
        mean=rs.mean_processes()[-1]
        for switch_variance in [True,False]:
            for switch_ar,switch_spf in [(False,False),(True,False),(True,True)]:
                spec=rs.ModelSpec(mean,'Normal',rs.SwitchingSpec(2,switch_ar,switch_spf,switch_variance))
                model,_,_=rs.build_model(synthetic_data(50),spec)
                old=model.start_params.copy()
                old[model.parameters['variance']]=[.1,.8] if switch_variance else .4
                smoothed,new=model._em_iteration(old)
                gamma=smoothed.smoothed_marginal_probabilities
                coefficients=np.stack([new[model.parameters[i,'exog']] for i in range(2)])
                residuals=model.endog[None,:]-coefficients@model.exog.T
                variance=(gamma*residuals**2).sum(axis=1)/gamma.sum(axis=1) if switch_variance else (gamma*residuals**2).sum()/model.nobs
                np.testing.assert_allclose(new[model.parameters['variance']],variance,atol=1e-12)
                scores=np.stack([model.exog.T@(gamma[i]/old[model.parameters[i,'variance']][0]*residuals[i]) for i in range(2)])
                for col,switches in enumerate(model.parameters.switching['exog']):
                    if switches:
                        np.testing.assert_allclose(scores[:,col],0,atol=1e-9)
                    else:
                        self.assertAlmostEqual(scores[:,col].sum(),0,places=9)
                        self.assertEqual(coefficients[0,col],coefficients[1,col])


if __name__=='__main__':
    unittest.main()
