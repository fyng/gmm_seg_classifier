import pytest
import numpy as np
from gmm_seg_classifier.gmm import GaussianMixtureModel as GMM
from sklearn.mixture import GaussianMixture

class TestGMM:
    @pytest.fixture
    def means_fixture(self):
        return np.array([(0, 0), (3, 3), (0, 4)]) # Mean of each Gaussian (3D)
    
    @pytest.fixture
    def covariance_fixture(self):
        return np.array([  # Covariance matrices for each Gaussian (3D)
            [[0.5, 0],[0, 0.5]],
            [[0.5, 0],[0, 0.5]],
            [[0.5, 0],[0, 0.5]]
        ])
    
    @pytest.fixture
    def weights_fixture(self):
        return np.array([0.3, 0.4, 0.3])

    @pytest.fixture
    def data_fixture(self, means_fixture, covariance_fixture, weights_fixture):
        n_samples = 10000
        n_components = len(weights_fixture)

        component_indices = np.random.choice(n_components, size=n_samples)
        samples = np.array([
            np.random.multivariate_normal(means_fixture[i], covariance_fixture[i]) 
            for i in component_indices
            ]) 
        return samples
    
    def test_fit_gmm(self, data_fixture, means_fixture, covariance_fixture, weights_fixture):
        model = GMM(n_components=3, tol=1e-5)
        model.fit(data_fixture)

        print(model.means)
        print(model.covariance)
        print(model.mixing_coeff)
        
        assert np.allclose(model.means, means_fixture, rtol = 0.1, atol = 0.1)
        assert np.allclose(model.mixing_coeff, weights_fixture, rtol = 0.1, atol = 0.1)
        assert np.allclose(model.covariance, covariance_fixture, rtol = 0.1, atol = 0.1)
    
    def test_sklearn_gmm(self, data_fixture, means_fixture, covariance_fixture, weights_fixture):
        model = GaussianMixture(n_components=3, tol=1e-5)
        model.fit(data_fixture)
        
        print(model.means_)
        print(model.covariances_)
        print(model.weights_)

        assert np.allclose(model.means_, means_fixture, rtol = 0.1, atol = 0.1)
        assert np.allclose(model.weights_, weights_fixture, rtol = 0.1, atol = 0.1)
        assert np.allclose(model.covariances_, covariance_fixture, rtol = 0.1, atol = 0.1)
        
