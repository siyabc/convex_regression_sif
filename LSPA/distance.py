import numpy as np

def squared_distance(left, right, axis=1):
    diff = np.square(left - right)
    if len(diff.shape) == 1:
        return np.sum(diff)
    return np.sum(diff, axis=axis)
