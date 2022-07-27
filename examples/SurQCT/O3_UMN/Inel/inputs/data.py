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
path_to_data = '../../datagen/data/'
file = {
    'train': path_to_data + 'train/pts.csv',
    'valid': path_to_data + 'valid/pts.csv'
}

columns = {
    'x': [[0,1,4,8],[9,10,13,17]],
    'y': [21,22],
    'sample_weight': [20,21]
}

header = 0

## *** From arrays
## Training data
#train_pts = {
#    'ics': 80,      # Number of training scenarios
#    'pts': 50,       # Number of total anchor pts for each scenario
#    'res': 160      # Number of total collocation pts for each scenario
#}
#validation_split = 0.2                 # % of Validation pts
#distribution = 'uniform'    # Training pts distribution
#
## Testing data
#test_cases = 5          # Number of testing cases
#test_training = True    # Test training cases

# Batching --------------------------------------------------------------------
batch_size = {
    'train': 64,
    'valid': 64
}

