import pytest
import numpy as np
from gmm_seg_classifier import GaussianModel, GaussianMixtureModel
from sklearn.mixture import GaussianMixture

class TestGMM:
    @pytest.fixture
    def means_fixture(self):
        return np.array([(200, 50, 50), (50, 200, 50), (50, 50, 200)]) # Mean of each Gaussian (3D)
    
    @pytest.fixture
    def covariance_fixture(self):
        return np.array([
            [[10, 2, 5], [2, 10, 1], [5, 1, 10]], 
            [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            [[10, 5, 0], [5, 10, 0], [0, 0, 10]]
        ])
    
    @pytest.fixture
    def weights_fixture(self):
        return np.ones(3) / 3

    @pytest.fixture
    def data_fixture(self, means_fixture, covariance_fixture, weights_fixture):
        n_samples = 10
        n_components = len(weights_fixture)

        component_indices = np.random.choice(n_components, size=n_samples)
        samples = np.array([
            np.random.multivariate_normal(means_fixture[i], covariance_fixture[i]) 
            for i in component_indices
            ]) 
        return samples
    
    def test_fit_gmm(self, data_fixture, means_fixture, covariance_fixture, weights_fixture):
        model = GaussianMixtureModel(n_components=3)
        model.fit(data_fixture)

        # not sure how to test this

        print(model.means)
        print(model.covariance)
        print(model.mixing_coeff)
        
        assert np.allclose(model.means, means_fixture, rtol = 0.1, atol = 0.1)
        assert np.allclose(model.mixing_coeff, weights_fixture, rtol = 0.1, atol = 0.1)
        assert np.allclose(model.covariance, covariance_fixture, rtol = 0.1, atol = 0.1)


class TestGaussian:
    @pytest.fixture
    def mean_fixture(self):
        return np.array([100, 15, 150])
    
    @pytest.fixture
    def cov_fixture(self):
        return np.array([
            [1, 0.5, 0.1],
            [0.5, 1, 0.1],
            [0.1, 0.1, 1]
        ])
    
    @pytest.fixture
    def data_fixture(self, mean_fixture, cov_fixture):
        n_sample = 2000
        return (np.rint(np.random.multivariate_normal(mean_fixture, cov_fixture, size=n_sample))).astype(np.uint8)
    
    def test_fit_gaussian_model(self, mean_fixture, cov_fixture, data_fixture):
        model = GaussianModel()
        model.fit(data_fixture)

        assert np.allclose(model.mean_(), mean_fixture, rtol=1e-2, atol = 1e-2)
        assert np.allclose(model.variance_(), cov_fixture, rtol=0.1, atol=0.1)

    def test_predict_gaussian_model_quantized(self, data_fixture, mean_fixture):
        model = GaussianModel()
        model.fit(data_fixture)

        model_cached = GaussianModel(cache=True)
        model_cached.fit(data_fixture)

        p1 = model.predict(mean_fixture)
        p2 = model_cached.predict(mean_fixture)

        assert np.isclose(p1, p2)

    def test_gaussian_model_save_and_load(self, data_fixture):
        fp = 'test_files/model1_params.json'
        model1 = GaussianModel()
        model1.fit(data_fixture)
        model1.save_to_json(fp)

        model2 = GaussianModel()
        model2.load(fp)

        assert np.allclose(model1.mean_(), model2.mean_())
        assert np.allclose(model1.variance_(), model2.variance_())
