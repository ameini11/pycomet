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
input_space = {
    'Ttr':  [8.e3,   1.5e4],        # Tanslational temperature [K]
    'T0':   [1.e3,   8.e3],         # Initial temperature [K]
    'P0':   [1.e3,   1.e4],         # Initial pressure [Pa]
    'XO2':  [0.05e0, 1.e0]          # Initial O2 mole fraction
}
temp_limits = input_space['Ttr']

# Time (limit values)
time_limit = [1.e-14, 1.e-3]

# Timescales thermochemistry --------------------------------------------------
ts_thermochem = {

    'TS0': {
        'species': {
            'O2': {
                'rovib': {
                    'system': 'O3',
                    'PES': 'UMN',
                    'grouping': None
                }
            },
            'O': None
        },
        'reaction': 'diss'
    },

    'TS1': {
        'species': {
            'O2': {
                'rovib': {
                    'system': 'O3',
                    'PES': 'UMN',
                    'grouping': 'RVE3'
                }
            }
        },
        'reaction': 'excit'
    }

}

groups_graph = {
    0: [0,1,2]
}

# Thermochemistry -------------------------------------------------------------
processes_mask = {      # Reactive processes
    'excit':  1,
    'diss':   1,
    'recomb': 1
}

# *** Thermo
mixture = {
    'var': 'Y',
    'type': 'MultiComponent',
    'species': ['O2', 'O']
}

thermo = 'CoupledEnergyModes'

# *** Species
species = {
    'O2': {
        'rovib': {
            'system': 'O3',
            'PES': 'UMN',
            'grouping': 'RVE3'
        }
    },
    'O': None
}
