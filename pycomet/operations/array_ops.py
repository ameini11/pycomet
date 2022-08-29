import numpy as np
import tensorflow as tf

from pycomet import utils
from tensorflow.python.keras import backend


def convert_struct_to_tensor(x):
    return tf.nest.map_structure(
        lambda y: tf.convert_to_tensor(y, dtype=backend.floatx()), x
    )

def get_cardinality(x):
    return get_cardinality(x[0]) if isinstance(x, (list,tuple)) else x.shape[0]

def get_struct_shape(structure=None, axis=None):
    if structure is not None:
        if axis:
            fn = lambda x: x.shape[axis] if x is not None else None
        else:
            fn = lambda x: x.shape if x is not None else None
        return tf.nest.map_structure(fn, structure)

def unpack_x_y_sample_weight_index(data):
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None, None, None)
    elif len(data) == 1:
        return (data[0], None, None, None)
    elif len(data) == 2:
        return (data[0], data[1], None, None)
    elif len(data) == 3:
        return (data[0], data[1], data[2], None)
    elif len(data) == 4:
        return (data[0], data[1], data[2], data[3])
    else:
        utils.raise_value_err(
            "Data is expected to be in format `x`, `(x,)`, "
            "`(x, y)`, `(x, y, sample_weight)`, or "
            f"`(x, y, sample_weight, sample_index)`, found: {data}"
        )

def extract_data_from_array(array, indices=None, axis=1):
    if indices is None:
        return array
    else:
        if any(isinstance(c, (list,tuple)) for c in indices):
            return [np.take(array, c, axis=axis) for c in indices]
        else:
            return np.take(array, indices, axis=axis)

def extract_data(container, indices=None, axis=1):
    if indices is not None:
        if not isinstance(indices, (dict, int)):
            utils.raise_value_err(
                "Indices not valid. It has to be a `int` or a "
                "`dict` with keys-values as integers."
            )
        if isinstance(container, (list, tuple)):
            k = list(indices.keys())[0]
            return extract_data(container[k], indices[k], axis)
        if isinstance(container, (np.ndarray, tf.Tensor)):
            return extract_data_from_array(container, indices, axis)
        else:
            utils.raise_value_err(
                "Container not valid. Valid options: "
                "[`np.ndarray`, `tf.Tensor`] or `list`/`tuple` of them."
            )
    else:
        return container
