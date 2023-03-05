import numpy as np


def stratified_sampling(x, strata):
    x_split = np.array_split(x, strata)
    x_1 = np.array([np.random.choice(i) for i in x_split])
    x_2 = np.setdiff1d(x, x_1)
    assert np.intersect1d(x_1, x_2).size == 0
    return x_2, x_1
