# *************************************************************************** #
#                                   PyCOMET                                   #
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
from_file = True

# *** From file
file = {
    'train': {
        'ics': './data/train/ics.csv',
        'pts': './data/train/pts.csv'
    },
    'valid': {
        'ics': './data/valid/ics.csv',
        'pts': './data/valid/pts.csv'
    },
    'test': './data/test/'
}

columns = {
    'x': [[0,8],[8,9]],
    'y': [11,12]
}

header = 0

# def preproc_fn(x):
#     """Manipulates data extracted from file.

#     Args:
#         x: Array or list of arrays.

#     Returns:
#         An array or list of arrays.
#     """
#     if x.shape[1] == columns['y'][-1][-1]:
#         x[:,columns['y'][0][0]+1] = 1. - x[:,columns['y'][0][0]]
#     return x

# *** From arrays
# Training data
train_pts = {
    'ics': 2048,            # Number of training scenarios
    'pts': 1,               # Number of total anchor pts for each scenario
    'res': 160              # Number of total collocation pts for each scenario
}
validation_split = 0.2      # % of Validation pts
distribution = 'uniform'    # Training pts distribution

# Testing data
test_cases = 50             # Number of testing cases
test_training = True        # Test training cases

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 32,
        'pts': 128
    },
    'valid': {
        'ics': 64,
        'pts': 64
    }
}
