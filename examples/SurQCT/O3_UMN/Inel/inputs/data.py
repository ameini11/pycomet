# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Generating and/or handling data for training and testing phase.
# --------------------------------------------------------------------------- #

# Data properties
# =============================================================================
data_type = 'DataSet'

# Data source -----------------------------------------------------------------
from_file = True

# *** From file
path_to_data = 'data_gen/data/'
file = {
    'train': path_to_data + 'train/pts.csv',
    'valid': path_to_data + 'valid/pts.csv'
}

columns = {
    'x': [[0,5],[5,10]],
    'y': [12,13]
}

header = 0

# Batching --------------------------------------------------------------------
batch_size = {
    'train': 64,
    'valid': 64
}

