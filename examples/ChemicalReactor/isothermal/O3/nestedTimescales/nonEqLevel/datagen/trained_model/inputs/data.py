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

ode_order = 1

# Data source -----------------------------------------------------------------
from_file = False

# *** From file
file = {
    'train': {
        'ics': './data/train/ics.csv',
        'pts': './data/train/pts.csv',
        'res': './data/train/res.csv'
    },
    'valid': {
        'ics': './data/valid/ics.csv',
        'pts': './data/train/pts.csv',
        'res': './data/valid/res.csv'
    },
    'test': './data/test/ics.csv'
}

columns = {
    'x': [[0,29],[29,30]],
    'y': [[30,32],[32,35],[35,44],[44,71]]
}

header = 0

def preproc_fn(x):
    """Manipulates data extracted from file.

    Args:
        x: Array or list of arrays.

    Returns:
        An array or list of arrays.
    """
    if x.shape[1] == columns['y'][-1][-1]:
        x[:,columns['y'][0][0]+1] = 1. - x[:,columns['y'][0][0]]
    return x

# *** From arrays
# Training data
train_pts = {
    'ics': 2048,            # Number of training scenarios
    'pts': 160,               # Number of total anchor pts for each scenario
    'res': 0              # Number of total collocation pts for each scenario
}
validation_split = 0.2      # % of Validation pts
distribution = 'uniform'    # Training pts distribution

# Testing data
test_cases = 50             # Number of testing cases
test_training = True        # Test training cases

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 128,
        'pts': 512,
        'res': 512
    },
    'valid': {
        'ics': 256,
        'pts': 256,
        'res': 256
    }
}
