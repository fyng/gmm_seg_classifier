import numpy as np
import json
import codecs

class GaussianMixtureModel:
    def __init__(self, n_components, n_iter=500, tol=1e-8, verbose=True):
        self.n_components = n_components 
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.means = None # [n_components, n_features]
        self.covariance = None # [n_components, n_features, n_features]
        self.mixing_coeff = None # [n_components]
        
    def fit(self, X):
        if not isinstance(X,np.ndarray):
            X = np.array(X)

        n_samples, n_features = X.shape
        np.random.seed(0)  # For reproducibility

        # Initialize mean as some random datapoints; covarance as identity; equal mixing coefficient
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)] # k x n_features
        self.covariance = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.mixing_coeff = np.ones(self.n_components) / self.n_components

        log_likelihood = 0

        for i in range(self.n_iter):
            attribution = self._e_step(X)
            self._m_step(X, attribution)
            new_log_likelihood = self._get_log_likelihood(X)
            if np.abs(new_log_likelihood - log_likelihood) <= self.tol:
                print(f'Converged at iteration {i}: log-likelihood = {log_likelihood}')
                break
            log_likelihood = new_log_likelihood
            print(f'Iteration {i}: log-likelihood = {log_likelihood}')

    def posterior(self, X, prior):
        pass

    def save_to_json(self, filepath):
            # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
            params = {}
            params['means'] = self.means.tolist()
            params['covariance'] = self.covariance.tolist()
            params['mixing_coeff'] = self.mixing_coeff.tolist()

            json.dump(params, codecs.open(filepath, 'w', encoding='utf-8'), 
                separators=(',', ':'), 
                sort_keys=True, 
                indent=4)

    def load(self, filepath):
        obj_text = codecs.open(filepath, 'r', encoding='utf-8').read()
        params = json.loads(obj_text)

        if 'means' in params:
            self.means = np.array(params['means'])
        else:
            raise RuntimeWarning('Cannot load model: no param \"mean\"')

        if 'covariance' in params:
            self.covariance = np.array(params['covariance'])
        else: 
            raise RuntimeWarning('Cannot load model: no param \"covariance\"')
        
        if 'mixing_coeff' in params:
            self.mixing_coeff = np.array(params['mixing_coeff'])
        else: 
            raise RuntimeWarning('Cannot load model: no param \"mixing coefficients\"')


    def _e_step(self, X):
        n_samples, _ = X.shape
        attribution = np.zeros((n_samples, self.n_components))

        for k in range (self.n_components):
            attribution[:,k] = self.mixing_coeff[k] * self._multivariate_normal(X, self.means[k], self.covariance[k])
        attribution /= (attribution.sum(axis=1, keepdims=True))
        # print(attribution)
        return attribution

    def _m_step(self, X, attribution): 
        n_samples, _ = X.shape
        norm = attribution.sum(axis=0) # (n_components, )

        # update mixing coefficients
        self.mixing_coeff = norm / norm.sum()

        # update means
        # self.mean = attribution.T * X / norm[:, np.newaxis] # (n_components, n_features)
        # self.mean = np.matmul(attribution.T, X) / norm[:, np.newaxis] # (n_components, n_features)

        # update covariance
        for k in range(self.n_components):
            z_k = attribution[:,k][:, np.newaxis] # (n_samples, )
            self.means[k] = (z_k * X).sum(axis=0) / z_k.sum()

            x_centered = X - self.means[k]
            weighted_x = z_k * x_centered
            self.covariance[k] = weighted_x.T @ x_centered / norm[k]

    def _multivariate_normal(self, X, mean, covariance):
        # calculate pdf of multivariate normal
        n_samples, n_features = X.shape
        z = (2*np.pi) ** (n_features/2) * (np.linalg.det(covariance) ** 0.5)
        inv_cov = np.linalg.inv(covariance)
        x_centered = X - mean

        exp_term = -0.5 * np.diag(x_centered @ inv_cov @ x_centered.T)
        p = np.exp(exp_term) / z

        return p + 1e-15
    
    def _get_log_likelihood(self, X):
        likelihood = 0
        for k in range(self.n_components):
            likelihood += self.mixing_coeff[k] * self._multivariate_normal(X, self.means[k], self.covariance[k])

        return np.sum(np.log(likelihood) / self.n_components) 
    
