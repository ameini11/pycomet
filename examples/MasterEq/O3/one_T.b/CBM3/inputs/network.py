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

from tensorflow.keras import layers as tf_layers
from prode.utilities.model import layers as pro_layers


# Architecture properties
# =============================================================================
architecture = 'MMOpNN'

dim = {
    "inp": [5, 1],
    "out": 4
}

end_scale = 'dense'

# DeepONets characterization --------------------------------------------------
names = ['O2', 'O']

opnn_dim = {
    names[0]: {
        "inp": dim["inp"],
        "out": 3
    },
    names[1]: {
        "inp": dim["inp"],
        "out": 1
    }
}

layer_size = {
    names[0]: {
        "branch": [opnn_dim[names[0]]['inp'][0], 16, 16, 32],
        "trunk":  [opnn_dim[names[0]]['inp'][1], 16, 16, 32]
    },
    names[1]: {
        "branch": [opnn_dim[names[1]]['inp'][0], 16, 16, 32],
        "trunk":  [opnn_dim[names[1]]['inp'][1], 32, 32, 32]
    }
}

activation = {
    names[0]: {
        "branch": 'elu',
        "trunk":  'sigmoid'
    },
    names[1]: {
        "branch": 'elu',
        "trunk":  'sigmoid'
    }
}

kernel_initializer = 'GlorotNormal'

block = 'ResPIBlock'

opnn_end_scale = None

stacked = True

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name':       'l1+l2',                          # 'fine_tuning'
    'l1':         1.e-4,
    'l2':         1.e-4,
    'weights_SP': './saving/MEq_OpNN_weights.h5'    # Path to for `FineTuningRegularizer`
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
input_fn = {
    "branch": tf_layers.Lambda(lambda x: x, name='branch_fn'),
    "trunk" : pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
}

# *** Outputs
output_fn = tf_layers.Softmax(axis=-1, name='output_fn')