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
p = 2 # Number of modes

layer_size = {
    "branch": [inp_gen.dim['inp'][0], 16, 16, p],
    "trunk":  [inp_gen.dim['inp'][1], 16, 16, p]
}

activation = {
    "branch": 'sigmoid',
    "trunk": 'sigmoid'
}

kernel_initializer = 'GlorotNormal'

block = 'FFBlock' #'FFBlock' 'ResPIBlock'

stacked_blocks = {
    "branch": True,
    "trunk": False
}

# Shifter
shifter = {
    "layer_size": [inp_gen.dim['inp'][0], 8, 8, 8, p],
    "activation": 'sigmoid'
}

# Augmentation
augmented = False

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name': 'l1+l2',
    'l1':   1.e-4,
    'l2':   1.e-4
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
input_fn = None
output_fn = None
