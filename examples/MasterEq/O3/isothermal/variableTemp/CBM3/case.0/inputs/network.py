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
import inputs.data as inp_data

from tensorflow.keras import layers as tf_layers
from prode.utilities.model import layers as pro_layers


# Architecture properties
# =============================================================================
architecture = 'MMOpNN'

dim = {
    "inp": [5, 1],
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
        "branch": [opnn_dim[names[1]]['inp'][0], 64, 64, 64],
        "trunk":  [opnn_dim[names[1]]['inp'][1], 64, 64, 64]
    }
}

activation = {
    names[0]: {
        "branch": 'gelu',
        "trunk":  'sigmoid'
    },
    names[1]: {
        "branch": 'gelu',
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
    'l1':         1.e-3,
    'l2':         1.e-3,
    'weights_SP': './saving/MEq_OpNN_weights.h5'    # Path to for `FineTuningRegularizer`
}

dropout_rate = 0


# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
def _branch_fn(x):
    T = inp_data.input_space['Ttr']
    T_norm = (x[:,0:1] - T[0]) / (T[1] - T[0])
    return tf.concat([T_norm, x[:,1:]], 1)

input_fn = {
    "branch": tf_layers.Lambda(_branch_fn, name='branch_fn'),
    "trunk" : pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
}

# *** Outputs
output_fn = tf_layers.Lambda(
    lambda y: [ y[0]*y[1][:,0:1], y[1] ], name='output_fn'
)

# Each single DeepONet --------------------------------------------------------
opnn_inp_trans = None
opnn_out_trans = {
    names[0]: tf_layers.Softmax(axis=-1, name='output_fn_'+names[0]),
    names[1]: tf_layers.Softmax(axis=-1, name='output_fn_'+names[1])
}
