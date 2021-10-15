# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import tensorflow as tf
import inputs.general as inp_gen


# Architecture properties
# =============================================================================
architecture = 'OpNN'

end_scale = 'bias'

# DeepONet characterization ---------------------------------------------------
layer_size = {
    "branch": [inp_gen.dim['inp'][0], 32, 32, 16],
    "trunk":  [inp_gen.dim['inp'][1], 32, 32, 16]
}

activation = {
    "branch": 'elu',
    "trunk": 'tanh'
}

kernel_initializer = 'GlorotNormal'

block = 'ResPIBlock' #'FFBlock'

stacked = True

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name': 'l1+l2',
    'l1': 1.e-4,
    'l2': 1.e-4
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
input_fn = None
output_fn = None

