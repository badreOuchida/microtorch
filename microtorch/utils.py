from microtorch.Value import Value

import numpy as np

def generator(size ,mean = 0 , std = 1) :

    """
    Generate random numbers from a normal distribution.

    Args:
        size: Size of the random array.
        mean: Mean of the normal distribution (default is 0).
        std: Standard deviation of the normal distribution (default is 1).

    Returns:
        Numpy array of random numbers from the normal distribution.
    """

    return np.random.normal(mean, std, size) * Value(1)
