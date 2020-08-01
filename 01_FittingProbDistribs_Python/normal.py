import numpy as np

def normal(X, mu, sigma):
    """Return likelihood of data given parameters"

    Computes the likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar likelihood
    """
    lik = np.product((1/np.sqrt(2.0*np.pi*sigma**2))*np.exp(-0.5*((X-mu)**2)/(sigma**2)))
    return lik