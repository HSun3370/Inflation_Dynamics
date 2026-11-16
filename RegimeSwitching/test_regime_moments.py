"""Validate mixture moments by exact Gaussian quadrature and limiting cases."""
import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
from regime_moments import gaussian_mixture_moments, average_predictive_moments


def quadrature(weights,means,variances):
    nodes,node_weights=hermgauss(5)
    points=np.asarray(means)[...,None]+np.sqrt(2*np.asarray(variances))[...,None]*nodes
    probabilities=np.asarray(weights)[...,None]*node_weights/np.sqrt(np.pi)
    mu=np.sum(points*probabilities)
    centered=points-mu
    v=np.sum(probabilities*centered**2)
    return np.array([mu,v,np.sum(probabilities*centered**3)/v**1.5,
                     np.sum(probabilities*centered**4)/v**2])


class MomentTests(unittest.TestCase):
    def test_normal_and_shared_mean_limits(self):
        normal=gaussian_mixture_moments([[1]],[[2]],[[.4]]).iloc[0]
        np.testing.assert_allclose(normal[['mean','variance','skewness','kurtosis']],[2,.4,0,3])
        mixed=gaussian_mixture_moments([[.7,.3]],[[2,2]],[[.1,.8]]).iloc[0]
        self.assertEqual(mixed.skewness,0)
        self.assertGreater(mixed['kurtosis'],3)
        shifted=gaussian_mixture_moments([[.7,.3]],[[1000002,1000002]],[[.1,.8]]).iloc[0]
        np.testing.assert_allclose(shifted[['variance','skewness','kurtosis']],mixed[['variance','skewness','kurtosis']])

    def test_conditional_and_averaged_moments_match_gaussian_quadrature(self):
        weights=np.array([[.8,.2],[.35,.65],[.5,.5]])
        means=np.array([[.1,1.2],[-.3,1.1],[1.4,2.1]])
        variances=np.array([[.1,.7],[.1,.7],[.1,.7]])
        path=gaussian_mixture_moments(weights,means,variances)
        keys=['mean','variance','skewness','kurtosis']
        for t in range(3):
            np.testing.assert_allclose(path.loc[t,keys],quadrature(weights[t],means[t],variances[t]),atol=1e-13)
        for centered in [False,True]:
            target_means=means-path['mean'].to_numpy()[:,None] if centered else means
            exact=quadrature(weights/3,target_means,variances)
            actual=average_predictive_moments(path,centered=centered)
            np.testing.assert_allclose([actual[k] for k in keys],exact,atol=1e-13)
        self.assertNotAlmostEqual(path['kurtosis'].mean(),average_predictive_moments(path)['kurtosis'])


if __name__=='__main__':
    unittest.main()
