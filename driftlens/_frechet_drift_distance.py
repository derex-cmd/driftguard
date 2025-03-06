import numpy as np
from scipy.linalg import sqrtm

def get_covariance(E) -> np.ndarray:
    """ Computes the covariance matrix.

    Args:
        E (:obj:`numpy.ndarray`): Embedding matrix of shape *(n_samples, n_features)*.

    Returns:
        :obj:`numpy.ndarray` Covariance matrix of shape *(n_features, n_features)*.
    """
    return np.cov(E, rowvar=False)


def get_mean(E) -> np.ndarray:
    """ Compute the Mean vector.

    Args:
        E (:obj:`numpy.ndarray`): Embedding matrix of shape *(n_samples, n_features)*.
        
    Returns:
        :obj:`numpy.ndarray`: Mean vector of shape *(n_features)*.
    """
    return E.mean(0)

#Hellinger distance DriftGuard

def frechet_distance(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """ Computes the Mahalanobis distance between two multivariate Gaussian distributions.

    Args:
        mu_x (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features)*.
        mu_y (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features)*.
        sigma_x (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.
        sigma_y (:obj:`numpy.ndarray`): Covariance matrix of the second Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: Mahalanobis distance between the two Gaussian distributions.
    """
    diff = mu_x - mu_y
    pooled_cov = (sigma_x + sigma_y) / 2
    
    # Use pseudo-inverse to avoid singular matrix issues
    inv_cov = np.linalg.pinv(pooled_cov)
    
    distance = np.sqrt(diff.T @ inv_cov @ diff)
    return distance

