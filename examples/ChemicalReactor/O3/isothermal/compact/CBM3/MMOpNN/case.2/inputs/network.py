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
from tensorflow.keras import activations as tf_act
from prode.utilities.model import layers as pro_layers


# Architecture properties
# =============================================================================
architecture = 'MMOpNN'

dim = {
    "inp": [4, 1],
    "out": [3, 2],
}

end_scale = None

# DeepONets characterization --------------------------------------------------
names = ['exc', 'dis']

opnn_dim = {
    names[0]: {
        "inp": dim["inp"],
        "out": 3
    },
    names[1]: {
        "inp": dim["inp"],
        "out": 2
    }
}

layer_size = {
    names[0]: {
        "branch": [opnn_dim[names[0]]['inp'][0], 64, 64, 64],
        "trunk":  [opnn_dim[names[0]]['inp'][1], 128, 128, 64]
    },
    names[1]: {
        "branch": [opnn_dim[names[1]]['inp'][0], 16, 16, 32],
        "trunk":  [opnn_dim[names[1]]['inp'][1], 64, 64, 32]
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

opnn_end_scale = 'dense'

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
# >> Output
output_fn = tf_layers.Lambda(
    lambda y: [ y[0]*y[1][:,0:1], y[1] ], name='output_fn'
)

# Each single DeepONet --------------------------------------------------------
opnn_inp_trans = None
opnn_out_trans = {
    names[0]: tf_layers.Softmax(axis=-1, name='output_fn_'+names[0]),
    names[1]: tf_layers.Softmax(axis=-1, name='output_fn_'+names[1])
}
