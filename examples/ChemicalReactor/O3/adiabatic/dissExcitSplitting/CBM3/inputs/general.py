# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining global input parameters and model characteristics 
# --------------------------------------------------------------------------- #


import os


# Model/architecture properties
# =============================================================================
model_name = 'IsoThermChemReact_MMOpNN'

dim = {
    "inp": [5, 1],
    "out": [2, 3]
}

# Execution Flags
# =============================================================================
flags = {
    'load':  0,       # Loading model       0: no, 1: saved model, 2: ckpt
    'train': 1,       # Training model      0: no, 1: yes
    'test':  1,       # Evaluating          0: no, 1: yes
    'plot':  2,       # Plotting            0: no, 1: offline, 2: online
    'save':  1        # Saving model        0: no, 1: yes
}

# Physical model
# =============================================================================
# Input space (limit values)
# input_space = {
#     'Ttr':  [8.e3,   1.5e4],        # Tanslational temperature [K]
#     'T0':   [1.e3,   8.e3],         # Initial temperature [K]
#     'P0':   [1.e3,   1.e4],         # Initial pressure [Pa]
#     'XO2':  [0.05e0, 1.e0]          # Initial O2 mole fraction
# }
input_space = {
    'Ttr':  [1.e4,   1.e4],        # Tanslational temperature [K]
    'T0':   [5.e3,   5.e3],         # Initial temperature [K]
    'P0':   [1.e4,   1.e4],         # Initial pressure [Pa]
    'XO2':  [0.5e0,  0.5e0]          # Initial O2 mole fraction
}
temp_limits = input_space['Ttr']

# Time (limit values)
time_limit = [1.e-14, 1.e-3]

# Thermochemistry -------------------------------------------------------------
# *** Species
species = {
    'O2': {
        'rovib': {
            'system': 'O3',
            'PES': 'UMN',
            'grouping': 'CBM3'
        }
    },
    'O': None
}

processFlags = {      # Reactive processes
    'excit':  1,
    'diss':   1
}

groups_graph = {
    0: [0,1,2]
}
