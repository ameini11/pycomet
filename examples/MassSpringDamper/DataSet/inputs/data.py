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
from_file = False

# *** From file
file = {
    'train': './data/train/pts.csv',
    'valid': './data/valid/pts.csv',
    'test': './data/test/ics.csv'
}

columns = {
    'x': [[0,2],[2,3]],
    'y': [3,5],
    # 'sample_weight': [5,6]
}

header = 0

# *** From arrays
# Training data
train_pts = {
    'ics': 100,      # Number of training scenarios
    'pts': 160      # Number of total collocation pts for each scenario
}
validation_split = 0.2                 # % of Validation pts
distribution = 'uniform'    # Training pts distribution

# Testing data
test_cases = 10          # Number of testing cases
test_training = True    # Test training cases

# Batching --------------------------------------------------------------------
batch_size = {
    'train': 64,
    'valid': 64
}
