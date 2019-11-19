import numpy as np
from scipy.spatial.distance import cdist

def sqexp(x1, x2, sigma, l):

    assert sigma > 0., "variance scale must be positive"
    assert l > 0., "length scale must be positive"

    x1 = np.array(x1)
    x2 = np.array(x2)

    return sigma**2*np.exp(-0.5*cdist(np.atleast_2d(x1), np.atleast_2d(x2))**2/l**2)