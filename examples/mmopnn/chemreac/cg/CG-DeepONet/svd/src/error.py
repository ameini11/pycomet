import numpy as np
import joblib as jl
import tensorflow as tf

from src import utils

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
def singlecase(
    y,
    y_rec,
    errors=['mrae', 'mse'],
    reductions=['max', 'mean'],
    ms_mep=None
):
    # Apply mep tranformation
    if ms_mep is not None:
        y = ms_mep.transform(y, "fwd")
        y_rec = ms_mep.transform(y_rec, "fwd")
    # Compute errors
    err = {}
    for e in errors:
        err[e] = {}
        for r in reductions:
            err[e][r] = tf.nest.map_structure(get_err(e, r), y, y_rec)
    return err

def multicases(
    y,
    y_rec,
    nb_cases,
    errors=['mare', 'mse'],
    reductions=['max', 'mean'],
    ms_mep=None,
    n_jobs=4
):
    # Compute
    err = jl.Parallel(n_jobs=n_jobs)(
        jl.delayed(singlecase)(
            utils.extract_case_from_struct(y, i, nb_cases),
            utils.extract_case_from_struct(y_rec, i, nb_cases),
            errors,
            reductions,
            ms_mep
        ) for i in range(nb_cases)
    )
    # Concatenate
    err = tf.nest.map_structure(lambda x: x.reshape(1), err)
    err = utils.concatenate_list_struct(err)
    # Compute errors
    for e in errors:
        for r in reductions:
            err[e][r] = tf.nest.map_structure(get_red_fn(r), err[e][r])
    return err