import numpy as np
import itertools
import json
import codecs

class GaussianModel:
    def __init__(self, cache=False, n_bits=6):
        self.cache = cache
        self.n_bits = n_bits
        self.mean = None 
        self.variance = None
        self.mean_quantized = None
        self.variance_quantized = None
        self.table = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.variance = np.cov(X, rowvar=0)

        if self.cache is True:
            max_value = (1 << self.n_bits) - 1
            max
            X_q = np.bitwise_and(X, max_value)
            self.mean_quantized = np.mean(X_q, axis=0)
            self.variance_quantized = np.cov(X_q, rowvar=0)

    def save_to_json(self, filepath):
        # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        params = {}
        params['mean'] = self.mean.tolist()
        params['variance'] = self.variance.tolist()
        if self.mean_quantized:
            params['mean_q'] = self.mean_quantized().tolist()
        if self.variance_quantized:
            params['variance_q'] = self.variance_quantized.tolist()

        # FIXME: probably a bad idea to serialize the table.
        # implement some checks to recache on first call to predict_cached()
        # if self.table:
        #     pass

        json.dump(params, codecs.open(filepath, 'w', encoding='utf-8'), 
            separators=(',', ':'), 
            sort_keys=True, 
            indent=4)

    def load(self, filepath):
        obj_text = codecs.open(filepath, 'r', encoding='utf-8').read()
        params = json.loads(obj_text)

        if 'mean' in params:
            self.mean = np.array(params['mean'])
        else:
            raise RuntimeWarning('Cannot load model: no param \"mean\"')

        if 'variance' in params:
            self.variance = np.array(params['variance'])
        else: 
            raise RuntimeWarning('Cannot load model: no param \"variance\"')
        
        if 'variance_q' in params:
            self.variance_quantized = np.array(params['variance_q'])
        else: 
            self.cache = False

        if 'mean_q' in params:
            self.mean_quantized = np.array(params['mean_q'])
        else: 
            self.cache = False


    def predict(self, X):
        # p(x|y), with full model
        X = X.reshape(1, -1)
        return self._multivariate_normal(X, self.mean, self.variance)
    
    def posterior(self, X, prior):
        pass


    def predict_cached(self, X):
        #FIXME: somehow it's really bad on real images even though test suite is passing
        if not self.cache:
            raise RuntimeError('Predictions are not cached. Use predict() or instantiate GaussianModel(cache=True) instead.')
        # predict with quantized model
        # build cache on first run
        if self.cache and self.n_bits > 0 and self.table is None:
            input = X.reshape(1, -1)
            _, n_features = input.shape
            self._cache(n_features)

        # quantize input 
        max_value = (1 << self.n_bits) - 1
        X_q = np.bitwise_and(X, max_value)
        return self.table[tuple(X_q)]

    def _cache(self, n_features):
        self.table = {}
        val_range = range(2 ** self.n_bits)
        keys = itertools.product(val_range, repeat=n_features)
        for k in keys:
            x = np.array(k).reshape(1, -1)
            self.table[k] = self._multivariate_normal(x, self.mean_quantized, self.variance_quantized)

    def _multivariate_normal(self, X, mean, covariance):
        # calculate pdf of multivariate normal
        _, n_features = X.shape
        z = (2*np.pi) ** (n_features/2) * (np.linalg.det(covariance) ** 0.5)

        x_centered = X - mean
        inv_cov = np.linalg.inv(covariance)
        exp_term = -0.5 * (x_centered @ inv_cov @ x_centered.T)

        return np.exp(exp_term) / z
        
    def mean_(self):
        return self.mean
    
    def variance_(self):
        return self.variance
    
