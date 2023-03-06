import numpy as np


def reconstruct(
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
