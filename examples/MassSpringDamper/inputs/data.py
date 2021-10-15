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
data_type = 'ODE'

# Data source -----------------------------------------------------------------
from_file = False

# *** From file
train_file = {
    'ics': './data/train/ics.csv',
    'pts': None,
    'res': './data/train/res.csv'
}

valid_file = {
    'ics': './data/valid/ics.csv',
    'pts': None,
    'res': './data/valid/res.csv'
}

test_file = './data/test/ics.csv'

col_x = [[0,2],[2,3]]
col_y = [3,5]

header = 0

# *** From arrays
# Training data
train_pts = {
    'ics': 80,      # Number of training scenarios
    'pts': 0,       # Number of total anchor pts for each scenario
    'res': 160      # Number of total collocation pts for each scenario
}
valid = 0.2                 # % of Validation pts
distribution = 'uniform'    # Training pts distribution

# Testing data
test_cases = 5          # Number of testing cases
test_training = True    # Test training cases

# Batching --------------------------------------------------------------------
batch_size = {
    'ics': 16,
    'pts': 16,
    'res': 64
}

valid_batch_size = {
    'ics': 32,
    'pts': 32,
    'res': 32
}
