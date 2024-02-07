import numpy as np
from scipy.stats import multivariate_normal

# refer to https://github.com/mr-easy/GMM-EM-Python/blob/master/GMM.py#L58

# CLASS NOTES: restrict mixing component to 1/k. No learning on it

class GaussianMixtureModel:
    def __init__(self, n_components, n_iter=1000, tol=1e-3, verbose=True):
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
        self.covariance = np.array([np.eye(n_features) + np.random.randint((n_features, n_features)) for _ in range(self.n_components)])
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

            # #FIXME: run one iteration
            # break

    def predict(self, X):
        pass

    def _e_step(self, X):
        n_samples, _ = X.shape
        attribution = np.zeros((n_samples, self.n_components))
        for k in range (self.n_components):
            attribution[:,k] = self.mixing_coeff[k] * self._multivariate_normal(X, self.means[k], self.covariance[k])
        attribution /= attribution.sum(axis=1, keepdims=True)

        return attribution

    def _m_step(self, X, attribution): 
        n_samples, _ = X.shape
        norm = attribution.sum(axis=0) # (n_components, )

        # update mixing coefficients
        self.mixing_coeff = norm / norm.sum()

        # update means
        self.mean = attribution.T.dot(X) / norm[:, np.newaxis] # (n_components, n_features)
        # self.mean = np.matmul(attribution.T, X) / norm[:, np.newaxis] # (n_components, n_features)

        # update covariance
        for k in range(self.n_components):
            x_centered = X - self.means[k]

            z_k = attribution[:,k] # (n_samples, )
            weighted_x = z_k[:,np.newaxis] * x_centered
            self.covariance[k] = weighted_x.T.dot(x_centered) / norm[k]

            # x_centered = np.expand_dims(X, axis=1) - self.mean[k]
            # s = np.matmul(x_centered.transpose([0, 2, 1]), x_centered)
            # self.covariance[k] = np.matmul(s.transpose(1, 2, 0), attribution[:, k] )
            # self.covariance[k] /= norm[k]


    def _multivariate_normal(self, X, means, covariance):
        # calculate pdf of multivariate normal
        # _, n_features = X.shape
        # z = (2*np.pi) ** (n_features/2) * (np.linalg.det(covariance) ** 0.5)

        # x_centered = X - means 
        # inv_cov = np.linalg.inv(covariance)
        # exp_term = np.diag(-0.5 * (x_centered.dot(inv_cov).dot(x_centered.T)))

        # # print(np.allclose(exp_term, exp_term.T))

        # p_x = np.exp(exp_term) / z

        # # https://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
        # # `eigh` assumes the matrix is Hermitian.
        # vals, vecs = np.linalg.eigh(covariance)
        # logdet     = np.sum(np.log(vals))
        # valsinv    = np.array([1./v for v in vals])
        # # `vecs` is R times D while `vals` is a R-vector where R is the matrix 
        # # rank. The asterisk performs element-wise multiplication.
        # U          = vecs * np.sqrt(valsinv)
        # rank       = len(vals)
        # dev        = X - means
        # # "maha" for "Mahalanobis distance".
        # maha       = np.square(np.dot(dev, U)).sum()
        # log2pi     = np.log(2 * np.pi)
        # exp_term = -0.5 * (rank * log2pi + maha + logdet)
        # p_x = np.exp(exp_term)

        return multivariate_normal.pdf(X, means, covariance)
    
    
    def _get_log_likelihood(self, X):
        likelihood = 0
        for k in range(self.n_components):
            likelihood += self.mixing_coeff[k] * self._multivariate_normal(X, self.means[k], self.covariance[k])

        return np.log(np.sum(likelihood) / self.n_components)
    
