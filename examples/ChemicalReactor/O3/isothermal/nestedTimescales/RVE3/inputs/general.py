# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Global input parameters
# --------------------------------------------------------------------------- #


import os
import numpy as np


# Paths
# =============================================================================
paths = {
    'database': os.path.normpath(os.getcwd()+'/../../../../database'),
    'utils':    os.path.normpath(os.getcwd()+'/utils'),
    'ode':      os.path.normpath(os.getcwd()+'/')
}

# Architecture properties
# =============================================================================
architecture = 'IsoThermChemReacNN'

dim = {
    "inp": [5, 1],
    "out": [2, 3]
}

# Physical quantities
# =============================================================================
# Input space (limit values)
# input_space = {
#     'Ttr':  [5.e3,   1.5e4],    # Tanslational temperature [K]
#     'T0':   [1.e3,   5.e3],     # Initial temperature [K]
#     'P0':   [1.e3,   1.e4],     # Initial pressure [Pa]
#     'XO2':  [0.05e0, 1.e0]      # Initial O2 mole fraction
# }
input_space = {
    'Ttr':  [1.e4,   1.e4],    # Tanslational temperature [K]
    'T0':   [5.e3,   5.e3],     # Initial temperature [K]
    'P0':   [1.e4,   1.e4],     # Initial pressure [Pa]
    'XO2':  [0.5e0,  0.5e0]      # Initial O2 mole fraction
}


# Time (limit values)
time_limit = [1.e-14, 1.e-3]

# Grouping
# grouping = [None, 'RVE3']
grouping = [None, 'CBM3']

# groups_graph = [0]

groups_graph = {
    0: [0,1,2]
}

var_names = [r'$O$', r'$O_2$'] \
    + [r'$O_2^{(1,{%s})}$' % (i+1) for i in range(dim['out'][1])]
