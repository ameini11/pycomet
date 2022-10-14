# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Handling data for training and testing phase.
# --------------------------------------------------------------------------- #


import tensorflow as tf


# Data properties
# =============================================================================
data_type = 'ODE'

physics_informed = False

# Data source -----------------------------------------------------------------
from_file = False

data_src = {
    'train': {
        'ics': './data/train/ics.csv',
        'pts': './data/train/pts.csv',
        'res': None #'./data/train/res.csv'
    },
    'valid': {
        'ics': './data/valid/ics.csv',
        'pts': './data/valid/pts.csv',
        'res': None #'./data/valid/res.csv'
    },
    'test': './data/test/'
}

cols = {
    'x': [[0,1,2,3],[4]],
    'y': [[5,6],[7,8]]
}
columns = tf.nest.map_structure(lambda _: cols, data_src)

header = 0

def preproc_fn(x):
    if x.shape[1] == cols['y'][-1][-1]+1:
        for c in cols['y']:
            x[:,c[0]:c[1]+1] = tf.nn.softmax(x[:,c[0]:c[1]+1])
    return x

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 64,
        'pts': 512,
        'res': 512
    },
    'valid': {
        'ics': 64,
        'pts': 512,
        'res': 512
    }
}
