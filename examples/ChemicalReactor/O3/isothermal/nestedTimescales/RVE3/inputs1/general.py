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
    "inp": [11, 1],
    "out": [2, 3, 9]
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
ode_order = 1
processes = {                   # Reactive processes
    'excit':  1,
    'diss':   1,
    'recomb': 1
}

# Input space (limit values)
input_space = {
    'Ttr':  [5.e3,   1.5e4],    # Tanslational temperature [K]
    'T0':   [1.e3,   5.e3],     # Initial temperature [K]
    'P0':   [1.e3,   1.e4],     # Initial pressure [Pa]
    'XO2':  [0.05e0, 1.e0]      # Initial O2 mole fraction
}
temp_limits = input_space['Ttr']

# Time (limit values)
time_limit = [1.e-14, 1.e-3]

# Grouping for each timescale
grouping = [None, 'RVE3', 'RVE9']

# groups_graph = [0]

# groups_graph = {
#     0: [0,1,2]
# }

groups_graph = {
    0: {
        0: [0,1,2],
        1: [3,4,5],
        2: [6,7,8]
    }
}


# Database --------------------------------------------------------------------
database = os.path.normpath(os.getcwd()+'/../../../database')

levels = database + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'

lev_bin_map = {
    n: database + '/grouping/O3_UMN/O2/LevelsMap_'+g+'.csv' \
        if g != None else g for g in grouping
}
