import pytest
import numpy as np
from gmm_seg_classifier.gmm import GaussianMixtureModel as GMM

class TestGMM:
    @pytest.fixture
    def data_fixture(self):
        means = [[0, 1, 2], [3, 4, 5], [-2, -1, 0]]  # Mean of each Gaussian (3D)
        covariances = [  # Covariance matrices for each Gaussian (3D)
            [[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]],
            [[1, 0.2, 0.1], [0.2, 1, 0.2], [0.1, 0.2, 1]],
            [[1, 0.3, 0.1], [0.3, 1, 0.3], [0.1, 0.3, 1]]
        ]
        mixture_coeff = [0.3, 0.4, 0.3]  # Mixing coefficients (should sum to 1)
        n_samples = 1000
        n_components = len(mixture_coeff)

        weights = np.array(mixture_coeff) / np.sum(mixture_coeff)
        component_indices = np.random.choice(n_components, size=n_samples)
        samples = np.array([
            np.random.multivariate_normal(means[i], covariances[i]) 
            for i in component_indices
            ]) 
        return samples
    
    def test_fit_gmm(self, data_fixture):
        model = GMM(n_components=3)
        model.fit(data_fixture)

        print(model.means)
        print(model.covariance)
        print(model.mixture_coeff)

        assert False

        
