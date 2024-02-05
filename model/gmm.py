import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components, n_iter=1000, tol=1e-3, verbose=True):
        self.n_components = n_components 
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.mean = None # vector. len = self.n_components
        self.covariance = None # vector. len = self.n_components
        self.mixing_coeff = None # vector. len = self.n_components
        
    def fit(self, X):
        n_samples, n_features = X.shape
        np.random.seed(0)  # For reproducibility

        # Initialize mean as some random datapoints; covarance as identity; equal mixing coefficient
        self.means = X[np.random.choice(n_samples, self.n_components, replacement=False)] # k x n_features
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.mixing_coefficients = np.ones(self.n_components) / self.n_components

        log_likelihood = 0

        for i in range(self.n_iter):
            attribution = self._e_step(X)
            self._m_step(X, attribution)
            new_log_likelihood = self._compute_log_likelihood(X)
            if np.abs(new_log_likelihood - log_likelihood) <= self.tol:
                print(f'Converged at iteration {i}: log-likelihood = {log_likelihood}')
                break
            log_likelihood = new_log_likelihood
            print(f'Iteration {i}: log-likelihood = {log_likelihood}')

    def _e_step(self, X):
        n_samples, _ = X.shape
        attribution = np.zeros(n_samples, self.n_components)
        for k in range (self.n_components):
            attribution[:,k] = self.mixing_coeff[k] * self._multivariate_normal(X, self.mean[k], self.covariance[k])
        attribution /= attribution.sum(axis=1)

        return attribution

    def _m_step(self, X, attribution): 
        n_samples, _ = X.shape
        z = attribution.sum(axis=0) # n_components length vector
        for k in range(self.n_components):
            z_k = z[k]
            self.mean[k] = (np.transpose(X) @ attribution[:,k]) / z_k

            delta_t = np.transpose(X - self.mean[k])
            self.covariance[k] = (delta_t @ attribution[:,k] @ delta_t) / z_k

        # n_samples = X.shape[0]
        #     for k in range(self.n_components):
        #         resp = responsibilities[:, k]
        #         total_resp = resp.sum()
        #         self.means[k] = (X * resp[:, np.newaxis]).sum(axis=0) / total_resp
        #         diff = X - self.means[k]
        #         self.covariances[k] = np.dot(resp * diff.T, diff) / total_resp
        #         self.mixing_coefficients[k] = total_resp / n_samples


    def _multivariate_normal(self, X, mean, covariance):
        _, n_features = X.shape
        z = (2*np.pi) ** (n_features/2) * np.linalg.norm(covariance)

        delta = X - mean 
        inv_cov = np.linalg.inv(covariance)
        exp_term = -0.5 * (np.transpose(delta) @ inv_cov @ delta)
        
        return np.exp(exp_term) / z

    def _get_log_likelihood(self, X):
        ll = 0
        for k in range(self.n_components):
            ll += np.log(np.sum(self._multivariate_normal(X, self.mean[k], self.covariance[k])))

        return ll
    
    # def mean_(self):
    #     return self.mean
    
    # def covariance_(self):
    #     return self.covariance
    
    # def mixing_coeff_(self):
    #     return self.mixing_coeff

