import numpy as np
from dtaidistance import dtw


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


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """ Computes the DTW distance between multivariate Gaussian distributions x and y, parameterized by their means and covariance matrices.

    Args:
        mu_x (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features)*.
        mu_y (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features)*.
        sigma_x (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.
        sigma_y (:obj:`numpy.ndarray`): Covariance matrix of the second Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: DTW distance between the two Gaussian distributions.

    """
    # Use only the mean vectors for DTW
    return dtw.distance(mu_x, mu_y)


# I swapped the Fréchet distance calculation with DTW, keeping the same argument structure!
# Let me know if you want me to fine-tune or optimize it further 🚀