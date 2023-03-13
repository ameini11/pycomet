import itertools
import numpy as np
import tensorflow as tf


# Slicing/Concatenating
# -----------------------------------------------------------------------------
def extract_case_from_struct(x, case_index, nb_cases):
    shapes = tf.nest.map_structure(
        lambda z: int(z.shape[1]/nb_cases), x
    )
    extract = lambda z, s: np.take(
        z, np.arange(case_index*s, (case_index+1)*s), axis=1
    )
    y = tf.nest.map_structure(extract, x, shapes)
    return tf.nest.map_structure(extract, x, shapes)

def concatenate_list_struct(list_struct):
    struct = list_struct[0]
    for i in range(1, len(list_struct)):
        struct = tf.nest.map_structure(
            lambda a, b: np.concatenate([a, b]),
            struct,
            list_struct[i]
        )
    return struct

def put(x, indices, values, axis=1):
    if len(indices) > 0:
        shape = (1,-1) if axis == 1 else (-1,1)
        indices = indices.reshape(*shape)
        np.put_along_axis(x, indices, values, axis)
    return x

# Splitting
# -----------------------------------------------------------------------------
def splitting_indices(ls, indices=[]):
    try:
        sub_ls = []
        start = 0
        for elem in ls:
            sub_ls.append(list(range(start,start+len(elem))))
            start = sub_ls[-1][-1] + 1
        indices.append(sub_ls)
        ls = list(itertools.chain(*ls))
        return splitting_indices(ls, indices=indices)
    except:
        return indices

def split_singlecase(y, method, indices=None):
    splits = []
    for i, y_i in enumerate(y):
        if method == "single":
            splits_i = np.hsplit(y_i, y_i.shape[1])
        else:
            splits_i = [np.take(y_i, j, axis=1) for j in indices[i]]
        splits.append(splits_i)
    return splits

def concatenate_multicases(cases):
    z = list(zip(*cases))
    z = [list(zip(*zi)) for zi in z]
    return [[np.hstack(zij) for zij in zi] for zi in z]

# Dictionary
# -----------------------------------------------------------------------------
def depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0