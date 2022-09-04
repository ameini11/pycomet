# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Handling data for training and testing phase.
# --------------------------------------------------------------------------- #


import tensorflow as tf


# Data properties
# =============================================================================
data_type = 'DataSet'

physics_informed = False

# Data source -----------------------------------------------------------------
from_file = True

data_src = {
    'train': './data/train/pts.csv',
    'valid': './data/valid/pts.csv'
}

columns = {
    'x': [[0,1,2,3],[4,5,6,7]],
    'y': [11],
    'sample_weight': [10]
}

header = 0

# Batching --------------------------------------------------------------------
batch_size = {
    'train': 64,
    'valid': 64
}
