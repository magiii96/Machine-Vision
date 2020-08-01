import numpy as np

def log_normal(X, mu, sigma):
    """Return log-likelihood of data given parameters"

    Computes the log-likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar log-likelihood
    """
    loglik = np.sum(np.log(1.0/np.sqrt(2.0*np.pi*sigma**2)*np.exp(-0.5*(X-mu)**2/sigma**2)))
    return loglik
 