# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Handling data for training and testing phase.
# --------------------------------------------------------------------------- #


import numpy as np
import tensorflow as tf

from inputs.general import dim



# Data properties
# =============================================================================
data_type = 'ODE'

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
    'x': np.split(
        np.arange(np.sum(dim['inp'])),
        np.cumsum(dim['inp'])
    )[:-1],
    'y': np.split(
        np.sum(dim['inp']) + np.arange(np.sum(dim['out'])),
        np.cumsum(dim['out'])
    )[:-1]
}
cols = tf.nest.map_structure(list, cols)
columns = tf.nest.map_structure(lambda _: cols, data_src)

header = 0

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 256,
        'pts': 256,
        'res': 512
    },
    'valid': {
        'ics': 256,
        'pts': 256,
        'res': 256
    }
}