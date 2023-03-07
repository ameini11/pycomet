import numpy as np
import tensorflow as tf

from src import utils
from PCAfold import preprocess


def reconstruct(
    x,
    pca,
    nb_pc=None,
    iremoved=None,
    iretained=None
):
    # Remove constant columns
    if iremoved is not None:
        xr = tf.nest.map_structure(
            lambda y, i: np.take(y, i, 1), x, iretained
        )
    else:
        xr = x
    # Reconstruct dataset
    xr_rec = _reconstruct(
        xr,
        pca,
        nb_pc
    )
    # Merge constant columns to reconstructed dataset
    if iremoved is not None:
        x_rec = tf.nest.map_structure(np.zeros_like, x)
        # Add removed columns
        x_rem = tf.nest.map_structure(
            lambda y, i: np.take(y, i, 1) if len(i) > 0 else None,
            x, iremoved
        )
        x_rec = tf.nest.map_structure(utils.put, x_rec, iremoved, x_rem)
        # Add reconstructed columns
        x_rec = tf.nest.map_structure(utils.put, x_rec, iretained, xr_rec)
    else:
        x_rec = xr_rec
    return x_rec

def _reconstruct(
    x,
    pca,
    nb_pc=None
):
    x_rec = []
    vars_id = list(pca.keys())
    for i, x_i in enumerate(x):
        x_rec_i = []
        if i > len(vars_id)-1:
            x_rec.append(x_i)
            continue
        k = vars_id[i]
        for j, x_ij in enumerate(x_i):
            # PCA object
            pca_ij = pca[k][j]
            if nb_pc is not None:
                pca_ij.n_components = nb_pc
            # Reconstruct solution
            pc = pca_ij.transform(x_ij)
            x_rec_ij = pca_ij.reconstruct(pc)
            # Collect
            x_rec_i.append(x_rec_ij)
        # Collect
        x_rec.append(x_rec_i)
    # Return values
    return x_rec

def remove_constant(
    X
):
    Xr, iremoved, iretained = preprocess.remove_constant_vars(X)
    return [np.array(x) for x in (Xr, iremoved, iretained)]
