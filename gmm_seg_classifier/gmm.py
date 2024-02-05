import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components, n_iter=1000, tol=1e-6, verbose=True):
        self.n_components = n_components 
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.means = None # vector. len = self.n_components
        self.covariance = None # vector. len = self.n_components
        self.mixing_coeff = None # vector. len = self.n_components
        
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

    def _e_step(self, X):
        n_samples, _ = X.shape
        attribution = np.zeros((n_samples, self.n_components))
        for k in range (self.n_components):
            attribution[:,k] = self.mixing_coeff[k] * self._multivariate_normal(X, self.means[k], self.covariance[k])
        attribution /= attribution.sum(axis=1)[:,None]

        return attribution

    def _m_step(self, X, attribution): 
        n_samples, _ = X.shape
        norm = attribution.sum(axis=0)
        for k in range(self.n_components):
            norm_k = norm[k]
            z_k = attribution[:,k]
            self.means[k] = (np.transpose(X) @ z_k) / norm_k

            delta = X - self.means[k]
            self.covariance[k] = (np.transpose(delta * z_k[:,np.newaxis]) @ delta) / norm_k
        
        # self.mixing_coeff = norm / norm.sum()

    def _multivariate_normal(self, X, means, covariance):
        # calculate pdf of multivariate normal

        # means:  n_sample vector
        # covariance: n_sample x n_sample
        _, n_features = X.shape
        z = (2*np.pi) ** (n_features/2) * np.linalg.det(covariance)

        delta = X - means 
        inv_cov = np.linalg.inv(covariance)
        exp_term = -0.5 * np.sum((delta.dot(inv_cov)*delta), axis=1)
        
        return np.exp(exp_term) / z
    
    def _get_log_likelihood(self, X):
        ll = 0
        for k in range(self.n_components):
            ll += np.log(np.sum(self._multivariate_normal(X, self.means[k], self.covariance[k])))

        return ll
    
