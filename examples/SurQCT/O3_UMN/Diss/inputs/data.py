# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Handling data for training and testing phase.
# --------------------------------------------------------------------------- #


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
    'x': [0,1,2],
    'y': [5]
}

header = 0

# Batching --------------------------------------------------------------------
batch_size = {
    'train': 32,
    'valid': 32
}
