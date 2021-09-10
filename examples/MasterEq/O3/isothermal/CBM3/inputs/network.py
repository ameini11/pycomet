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
    "inp": [6, 1],
    "out": 4
}

# DeepONets characterization --------------------------------------------------
names = ['O2', 'O']

opnn_dim = {
    names[0]: {
        "inp": [6, 1],
        "out": 3
    },
    names[1]: {
        "inp": [6, 1],
        "out": 1
    }
}

layer_size = {
    names[0]: {
        "branch": [opnn_dim[names[0]]['inp'][0], 32, 32, 32],
        "trunk":  [opnn_dim[names[0]]['inp'][1], 32, 32, 32]
    },
    names[1]: {
        "branch": [opnn_dim[names[1]]['inp'][0], 32, 32, 32],
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

kernel_initializer = "GlorotNormal"

block = 'ResPIBlock'

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name':         'l1+l2',                                # 'fine_tuning'
    'l1':           1.e-4,
    'l2':           1.e-4,
    'weights_SP':   None #'./saving/MEq_OpNN_weights.h5'    # Path to for `FineTuningRegularizer`
}

dropout_rate = 0

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
    "branch": tf_layers.Lambda(branch_fn, name='branch_fn'),
    "trunk" : pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
}
# >> Output
output_fn = tf_layers.Lambda(
    lambda y: tf_layers.Softmax(axis=-1, name='output_fn')(tf.concat(y, 1))
)

# Each single DeepONet --------------------------------------------------------
opnn_inp_trans = None
opnn_out_trans = None
