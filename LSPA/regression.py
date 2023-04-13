import numpy as np
from estimator import EstimatorModel

def _prepare_prediction(weights, X, extend_X1):
    if isinstance(weights, EstimatorModel):
        weights = weights.weights
    if extend_X1:
        X = np.insert(X, 0, 1.0, axis=1)
    return weights, X

def max_affine_predict(weights, X, extend_X1=True):
    weights, X = _prepare_prediction(weights, X, extend_X1)
    # print("weights:", weights)
    return np.max(X.dot(weights.T), axis=1)


def max_affine_fit_partition(partition, X, y, extend_X1=True, rcond=None):
    if extend_X1:
        X = np.insert(X, 0, 1.0, axis=1)
    weights = np.empty((partition.ncells, X.shape[1]))
    for i, cell in enumerate(partition.cells):
        weights[i, :] = np.linalg.lstsq(X[cell, :], y[cell], rcond=rcond)[0]
    return weights

