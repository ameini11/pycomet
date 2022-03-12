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
file = {
    'train': {
        'ics': './data/train/ics.csv',
        'pts': './data/train/pts.csv'
    },
    'valid': {
        'ics': './data/valid/ics.csv',
        'pts': './data/train/pts.csv'
    },
    'test': './data/test/ics.csv'
}

columns = {
    'x': [[0,29],[29,30]],
    'y': [[30,31],[31,34],[34,43]]
}

header = 0

# *** From arrays
# Training data
train_pts = {
    'ics': 10,            # Number of training scenarios
    'pts': 160               # Number of total anchor pts for each scenario
}
validation_split = 0.2      # % of Validation pts
distribution = 'uniform'    # Training pts distribution

# Testing data
test_cases = 1             # Number of testing cases
test_training = True        # Test training cases

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 16,
        'pts': 64
    },
    'valid': {
        'ics': 16,
        'pts': 64
    }
}
