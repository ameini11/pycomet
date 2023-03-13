import numpy as np
import joblib as jl
import tensorflow as tf

from src import utils
from src import reduction as red_mod

EPS = 1.e-10


# Error functions
# -----------------------------------------------------------------------------
def mrae(y_true, y_pred):
    return np.abs(y_true - y_pred) / (np.abs(y_true) + EPS)

def mse(y_true, y_pred):
    return (y_true - y_pred)**2

def get_err_fn(error):
    return {
        'mrae': mrae,
        'mse': mse
    }[error]

def get_red_fn(reduction):
    return {
        'max': np.amax,
        'mean': np.mean
    }[reduction]

def get_err(error, reduction):
    err = get_err_fn(error)
    red = get_red_fn(reduction)
    def err_fn(y_true, y_pred):
        return red(err(y_true, y_pred))
    return err_fn

# Error computations
# -----------------------------------------------------------------------------
def compute_err(
    y,
    pca,
    nb_pc,
    nb_cases,
    iremoved=None,
    iretained=None,
    errors=['mrae', 'mse'],
    reductions=['max', 'mean'],
    nb_jobs=4
):
    # Reconstruct dataset
    y_rec = red_mod.reconstruct(
        y,
        pca,
        nb_pc,
        iremoved,
        iretained
    )
    # Compute errors
    err = {}
    for e in errors:
        err[e] = {}
        for r in reductions:
            err[e][r] = tf.nest.map_structure(get_err(e, r), y, y_rec)
    err = tf.nest.map_structure(lambda x: x.reshape(1), err)
    return err