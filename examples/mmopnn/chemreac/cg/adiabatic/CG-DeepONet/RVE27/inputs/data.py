# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Handling data for training and testing phase.
# --------------------------------------------------------------------------- #


import numpy as np
import tensorflow as tf


# Data properties
# =============================================================================
data_type = 'DataSet'

physics_informed = False

# Data source -----------------------------------------------------------------
from_file = False

data_src = {
    'train': {
        'ics': None, #'./data/train/ics.csv',
        'pts': './data/train/pts.csv',
        'res': None #'./data/train/res.csv'
    },
    'valid': {
        'ics': None, #'./data/valid/ics.csv',
        'pts': './data/valid/pts.csv',
        'res': None #'./data/valid/res.csv'
    },
    'test': './data/test/'
}

cols = {
    'x': [np.arange(29),[29]],
    'y': [[30],31+np.arange(27),[58]]
    
}
columns = tf.nest.map_structure(lambda _: cols, data_src)

header = 0

def preproc_fn(x):
    if x.shape[1] == columns['y'][-1][-1]:
        s = np.sum(x[:,columns['y'][1]], axis=1, keepdims=1)
        x[:,columns['y'][0]] = 1.0 - s
    return x

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 128,
        'pts': 128,
        'res': 512
    },
    'valid': {
        'ics': 256,
        'pts': 256,
        'res': 256
    }
}