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

from tensorflow.keras import layers
from prode.utilities.model import layers as pro_layers


# Architecture properties
# =============================================================================
architecture = 'MMOpNN'

# Variables dimensionality ----------------------------------------------------
dim = {
    "inp": [6, 1],  # Branch-Trunk input dimensionality
    "out": [3, 1]   # Output dimensionality
}

output_names = ['O2', 'O']

# Structure -------------------------------------------------------------------
layer_size = {
    output_names[0]: {
        "branch": [dim['inp'][0], 32, 32, 32],
        "trunk":  [dim['inp'][1], 32, 32, 32]
    },
    output_names[1]: {
        "branch": [dim['inp'][0], 32, 32, 32],
        "trunk":  [dim['inp'][1], 64, 64, 32]
    }
}

activation = {
    output_names[0]: {
        "branch": 'elu',
        "trunk":  'sigmoid'
    },
    output_names[1]: {
        "branch": 'elu',
        "trunk":  'sigmoid'
    }
}

regularization = {
    'name':         'l1+l2',                                # 'fine_tuning'
    'l1':           1.e-4,
    'l2':           1.e-4,
    'weights_SP':   None #'./saving/MEq_OpNN_weights.h5'    # Path to for `FineTuningRegularizer`
}

batch_norm = {
    "branch": None,
    "trunk":  None
}

dropout_rate = 0
kernel_initializer = "GlorotNormal"
block = 'ResPIBlock'

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# >> Input
def branch_fn(x):
    T = inp_data.inp_space['Ttr']
    if T[1] != T[0]:
        T_norm = (x[:,0:1] - T[0]) / (T[1] - T[0])
    else:
        T_norm = x[:,0:1]/T[0]
    return tf.concat([T_norm, x[:,1:]], 1)

input_fn = {
    "branch": layers.Lambda(branch_fn, name='branch_fn'),
    "trunk" : pro_layers.Scaling(
        init_scale=1.e7, name='trunk_fn', trainable=False
    )
}
# >> Output
output_fn = layers.Lambda(
    lambda y: [ y[0]*(1.-y[1]), y[1] ], name='output_fn'
)

# Each single DeepONet --------------------------------------------------------
opnn_inp_trans = None
opnn_out_trans = {
    output_names[0]: layers.Softmax(axis=-1, name=output_names[0]+'_output_fn'),
    output_names[1]: layers.Lambda(
        lambda x: tf.clip_by_value(x, 0., 1.), name=output_names[1]+'_output_fn'
    )
}
