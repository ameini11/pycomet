# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Handling data for training and testing phase.
# --------------------------------------------------------------------------- #


import tensorflow as tf


# Data properties
# =============================================================================
data_type = 'ODE'

physics_informed = True

# Data source -----------------------------------------------------------------
from_file = False

data_src = {
    'train': {
        'ics': './data/train/ics.csv',
        'pts': './data/train/pts.csv',
        'res': './data/train/res.csv'
    },
    'valid': {
        'ics': './data/valid/ics.csv',
        'pts': './data/valid/pts.csv',
        'res': './data/valid/res.csv'
    },
    'test': './data/test/'
}

columns = {
    'x': [[0,1],[2,3],[4]],
    'y': [3,4]
}
columns = tf.nest.map_structure(lambda _: columns, data_src)

header = 0

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 64,
        'pts': 256,
        'res': 256
    },
    'valid': {
        'ics': 64,
        'pts': 256,
        'res': 256
    }
}
