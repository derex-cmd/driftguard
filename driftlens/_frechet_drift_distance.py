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

#Hellinger distance
def frechet_distance(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """ Computes the Hellinger distance between two multivariate Gaussian distributions.

    Args:
        mu_x (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features)*.
        mu_y (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features)*.
        sigma_x (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.
        sigma_y (:obj:`numpy.ndarray`): Covariance matrix of the second Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: Hellinger distance between the two Gaussian distributions.

    """
    det_sigma_x = np.linalg.det(sigma_x)
    det_sigma_y = np.linalg.det(sigma_y)
    det_sigma_avg = np.linalg.det((sigma_x + sigma_y) / 2)

    exp_term = np.exp(-0.125 * (mu_x - mu_y).T @ np.linalg.inv((sigma_x + sigma_y) / 2) @ (mu_x - mu_y))
    distance = np.sqrt(1 - (exp_term * (det_sigma_x ** 0.25) * (det_sigma_y ** 0.25) / (det_sigma_avg ** 0.5)))
    
    return distance

# I swapped the distance metric to **Hellinger distance**, which is great for capturing both mean and shape changes!
# Let me know if you want me to refine or test this! 🚀
