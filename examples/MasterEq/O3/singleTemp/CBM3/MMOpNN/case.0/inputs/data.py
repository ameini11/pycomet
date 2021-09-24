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


import os
import numpy as np

import inputs.network as inp_net


# Paths
# =============================================================================
paths = {
    'database': os.path.normpath(os.getcwd()+'/../../../../database'),
    'utils':    os.path.normpath(os.getcwd()+'/../../../utils'),
    'ode':      os.path.normpath(os.getcwd()+'/../../')
}

# Data properties
# =============================================================================
data_type = 'ODE'

# Data source -----------------------------------------------------------------
from_file = True

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
col_y = [[6,9],[9,10]]

header = 0

def preproc_fn(x):
    """Manipulates data extracted from file.

    Args:
        x: Array or list of arrays.

    Returns:
        An array or list of arrays.
    """
    if x.shape[1] == col_y[1][0]:
        x[:,-1] = 1. - np.sum(x[:,col_y[0][0]:col_y[0][1]-1], axis=1)
    return x

# *** From arrays
# Training data
train_pts = {
    'ics': 10,#2048,    # Number of training scenarios
    'pts': 1,       # Number of total anchor pts for each scenario
    'res': 160      # Number of total collocation pts for each scenario
}
valid = 0.2                 # % of Validation pts
distribution = 'uniform'  # Training pts distribution

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

# Testing data
test_cases = 5#40         # Number of testing cases
test_training = True    # Test training cases

# ODE properties
# =============================================================================
ode_order = 1
grouping = 'CBM3'                                   # ROM: ['CBM3', 'DPM10']
processes = {'excit': 1, 'diss': 1, 'recomb': 1}    # Reactive processes

# Input space (limit values)
input_space = {
    'Ttr':  [1.e4,   1.e4],     # Tanslational temperature [K]
    'T0':   [1.e3,   1.e4],     # Initial temperature [K]
    'P0':   [1.e2,   1.e4],     # Initial pressure [Pa]
    'XO2':  [0.05e0, 1.e0]      # Initial O2 mole fraction
}

# Time limits for testing
t_limit = [1.e-14, 1.e-3]
