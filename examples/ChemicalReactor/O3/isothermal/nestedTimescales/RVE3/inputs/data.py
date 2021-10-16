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
import inputs.general as inp_gen


# Data properties
# =============================================================================
data_type = 'ODE'

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
    if x.shape[1] == col_y[-1][-1]:
        x[:,col_y[0][0]] = 1. - x[:,col_y[0][0]+1]
    return x

# *** From arrays
# Training data
train_pts = {
    'ics': 10,#4096,    # Number of training scenarios
    'pts': 1,       # Number of total anchor pts for each scenario
    'res': 160      # Number of total collocation pts for each scenario
}
valid = 0.2                 # % of Validation pts
distribution = 'random' #'uniform'    # Training pts distribution

batch_size = {
    'ics': 256,
    'pts': 256,
    'res': 1024
}

valid_batch_size = {
    'ics': 256,
    'pts': 256,
    'res': 256
}

# Testing data
test_cases = 5#40      # Number of testing cases
test_training = True    # Test training cases

# ODE properties
# =============================================================================
ode_order = 1
grouping = inp_gen.grouping[-1]                     # ROM: ['CBM3', 'DPM10']
processes = {'excit': 1, 'diss': 0, 'recomb': 0}    # Reactive processes
# processes = {'excit': 1, 'diss': 1, 'recomb': 1}    # Reactive processes
groups_graph = inp_gen.groups_graph

# Input space (limit values)
input_space = inp_gen.input_space

# Time limits for testing
time_limit = inp_gen.time_limit
