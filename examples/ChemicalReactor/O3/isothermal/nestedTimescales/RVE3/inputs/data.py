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
train_file = {
    'ics': './data/train/ics.csv',
    'pts': './data/train/pts.csv',
    'res': './data/train/res.csv'
}

valid_file = {
    'ics': './data/valid/ics.csv',
    'pts': './data/valid/pts.csv',
    'res': './data/valid/res.csv'
}

test_file = './data/test/ics.csv'

col_x = [[0,5],[5,6]]
col_y = [[6,8],[8,11]]

header = 0

def preproc_fn(x):
    """Manipulates data extracted from file.

    Args:
        x: Array or list of arrays.

    Returns:
        An array or list of arrays.
    """
    if x.shape[1] == col_y[1][1]:
        x[:,col_y[0][0]+1] = 1. - x[:,col_y[0][0]]
    return x

# *** From arrays
# Training data
train_pts = {
    'ics': 2048,            # Number of training scenarios
    'pts': 1,               # Number of total anchor pts for each scenario
    'res': 160              # Number of total collocation pts for each scenario
}
valid = 0.2                 # % of Validation pts
distribution = 'uniform'    # Training pts distribution

# Testing data
test_cases = 50             # Number of testing cases
test_training = True        # Test training cases

# Batching --------------------------------------------------------------------
batch_size = {
    'ics': 128,
    'pts': 128,
    'res': 512
}

valid_batch_size = {
    'ics': 128,
    'pts': 128,
    'res': 128
}
