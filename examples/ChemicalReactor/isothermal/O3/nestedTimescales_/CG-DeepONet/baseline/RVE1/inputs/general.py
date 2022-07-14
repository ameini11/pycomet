# *************************************************************************** #
#                                    PyCOMET                                    #
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
    "inp": [2+1, 1],
    "out": 2
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
physics_informed = True

ode_order = 2

# Input space (limit values)
input_space = {
    'Ttr':  [8.e3,   1.5e4],        # Tanslational temperature [K]
    'T0':   [1.e3,   8.e3],         # Initial temperature [K]
    'P0':   [1.e3,   1.e4],         # Initial pressure [Pa]
    'XO2':  [0.05e0, 1.e0]          # Initial O2 mole fraction
}

# Temperature (limit values)
temperature_limits = input_space['Ttr']

# Input log-scaled mass fractions 
log_mass_frac = False

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
        'reaction': None #'diss'
    }

}

groups_graph = [0]

# Thermochemistry -------------------------------------------------------------
# *** Species
species = {
    'O2': {
        'rovib': {
            'system': 'O3',
            'PES': 'UMN',
            'grouping': 'RVE1'
        }
    },
    'O': None
}

processFlags = {      # Reactive processes
    'excit': 0,
    'diss':  1
}
