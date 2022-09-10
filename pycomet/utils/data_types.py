import os
import h5py
import numpy as np
import tensorflow as tf

from pycomet import utils
from tensorflow.python.keras import backend


def make_dict(keys, values):
    """Convert two lists or two variables into a dictionary."""
    if isinstance(keys, (list, tuple)):
        if len(keys) != len(values):
            utils.raise_value_err("Keys and values have different length.")
        return dict(zip(keys, values))
    return {keys: values}

def numbers_to_str(numbers, precision=2):
    """Convert a list/tuple/array of numbers into a string."""
    if numbers is None:
        return ""
    if not isinstance(numbers, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(numbers, precision)
    else:
        return "[{:s}]".format(
            ", ".join(["{:.{}e}".format(x, precision) for x in numbers])
        )

def compute_dict_depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(compute_dict_depth, d.values())) if d else 0)
    return 0

def value_to_dict(value, keys):
    return value if isinstance(value, dict) else {k: value for k in keys}

def search_string(file, string):
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if string+' ' in line:
                return i, line

def check_file_extension(file, extension):
    file = os.path.abspath(file)
    if os.path.isfile(file):
        if os.path.splitext(file)[-1] != extension:
            utils.raise_value_err(
                f"The file '{file}' has to be a '{extension}' one."
            )

def zip_dict_of_lists(d):
    keys = d.keys()
    return tuple(dict(zip(keys, v)) for v in zip(*(d[k] for k in keys)))

def get_key_from_value(d, v):
    for key, value in d.items():
        if v == value:
            return key
    utils.raise_value_err(
        f"A key for value `{v}` doesn't exist in the dictionary."
    )

def get_keys(d):
    result = []
    for key, value in d.items():
        if isinstance(value, dict):
            new_keys = get_keys(value)
            result.append(key)
            for innerkey in new_keys:
                result.append(f'{key}/{innerkey}')
        else:
            result.append(key)
    return result

def check_dicts_consistency(self, list_of_dicts=[], idx_ref_dict=0):
    list_of_keys = tf.nest.map_structure(
        lambda x: (x[0], get_keys(x[1])), list_of_dicts
    )
    for i, (name, keys) in enumerate(list_of_keys):
        if i == idx_ref_dict:
            continue
        else:
            for k in keys[idx_ref_dict]:
                if k not in keys[i]:
                    utils.raise_value_err(
                        f"A structure discrepancy between '{name}' and "
                        f"'{list_of_dicts[idx_ref_dict][0]}' dictionaries "
                        "has been found."
                    )

def squeeze_list(iterable):
    iterable = list(iterable)
    if len(iterable) == 1:
        return iterable[-1]
    else:
        return iterable

class H5DataSet:
    def __init__(self, path, no_group=False):
        self.data = {}
        self.no_group = no_group
        self.df = h5py.File(path,'r')
        self.df.visititems(self)
        self.df.close()

    def __call__(self, name, node):
        if isinstance(node, h5py.Dataset):
            if self.no_group:
                name_ = os.path.join(*(name.split(os.path.sep)[1:]))
            else:
                name_ = name
            self.data[name_] = self.df[name][()]
