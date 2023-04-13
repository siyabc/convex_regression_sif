import numpy as np
import pandas as pd
from functools import partial
from partition import rand_voronoi_partition, max_affine_partition
from distance import squared_distance
from regression import max_affine_predict, max_affine_fit_partition
from estimator import EstimatorModel, Estimator

class LSPAEstimatorModel(EstimatorModel):
    def __init__(self, weights, niters):
        EstimatorModel.__init__(self, weights)
        self.niters = niters


def lspa_train(
    X,
    y,
    coef_name,
    ncenters,
    nrestarts=1,
    nfinalsteps=None,
    obj_tol=1e-6,
):
    n, d = X.shape
    if isinstance(ncenters, str):
        ncenters = int(np.ceil(eval(ncenters)))
    if isinstance(nrestarts, str):
        nrestarts = int(np.ceil(eval(nrestarts)))
    if nfinalsteps is None:
        nfinalsteps = n
    elif isinstance(nfinalsteps, str):
        nfinalsteps = int(np.ceil(eval(nfinalsteps)))

    X1 = np.insert(X, 0, 1.0, axis=1)

    niters = []
    best_err = np.inf
    best_weights = None
    for restart in range(nrestarts):
        partition = rand_voronoi_partition(ncenters, X)
        niter = 0
        maxiter = nfinalsteps
        while niter < maxiter:
            niter += 1
            weights = max_affine_fit_partition(partition, X1, y, extend_X1=False)
            yhat = max_affine_predict(weights, X1, extend_X1=False)
            # print("yhat:", yhat)
            err = squared_distance(yhat, y, axis=0)
            if err < best_err - obj_tol:
                maxiter = niter + nfinalsteps
                best_err = err
                best_weights = weights

            induced_partition = max_affine_partition(X1, weights)
            if partition == induced_partition:
                break  
            partition = induced_partition

        niters.append(niter)
        print("==num_best_weights:", np.shape(best_weights))

    print(coef_name)
    print("best_weights:", best_weights)
    save_weights = -best_weights
    dataframe = pd.DataFrame(
        {'beta1': save_weights[:, 0], 'beta2': save_weights[:, 1], 'alpha': save_weights[:, 2]})
    dataframe.to_csv("../"+coef_name+".csv", index=False, sep=',')
    return LSPAEstimatorModel(
        weights=best_weights,
        niters=niters,
    )

        
class LSPAEstimator:
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(lspa_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
