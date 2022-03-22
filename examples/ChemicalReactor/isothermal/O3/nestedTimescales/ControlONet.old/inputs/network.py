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

from tensorflow.python.keras import layers as tf_layers
from prode.utilities.model import layers as pro_layers


# Architecture properties
# =============================================================================
architecture = 'MMOpNN'

# TimeScaleNNs characterization -----------------------------------------------
names = ['NEQ1', 'NEQ2', 'NEQ3_1', 'NEQ3_2', 'NEQ3_3']

opnn_dim = {
    names[i]: {
        "inp": inp_gen.dim['inp'],
        "out": d
    } for i, d in enumerate(inp_gen.dim['out'])
}

layer_size = {
    names[0]: {
        "branch": [2+27, 32, 32, 8],
        "trunk":  [1, 32, 32, 8]
    }
}
layer_size.update({
    name: {
        "branch": [2+27, 32, 32, 16],
        "trunk":  [1, 32, 32, 16]
    } for name in names[1:]
})

activation = {
    name: {
        "branch": 'tanh',   # elu/gelu
        "trunk":  'sigmoid'
    } for name in names
}

kernel_initializer = 'GlorotNormal' #'GlorotNormal' 'zeros'

block = 'FFBlock' # 'FFBlock' 'ResPIBlock'
block_linear_out = False

opnn_end_scale = 'dense'

stacked = True

cross_encoder = {
    "branch": False,
    "trunk": False
}

augmented = True

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name':         'l1+l2',                                      # 'fine_tuning'
    'l1':           1.e-4,
    'l2':           1.e-4,
    'weights_path': './saving/'+inp_gen.model_name+'_weights.h5'  # For `FineTuningRegularizer`
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
def branch_fn(x):
    T = inp_gen.input_space['Ttr']
    if T[1] != T[0]:
        T_norm = (x[:,0:1] - T[0]) / (T[1] - T[0])
    else:
        T_norm = x[:,0:1]/T[0]
    return tf.concat([T_norm, x[:,1:]], 1)

input_fn = {
    "branch": tf_layers.Lambda(branch_fn, name='branch_fn'),
    "trunk": pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
    # "trunk": tf_layers.Lambda(lambda x: tf.math.reciprocal((x+1.e-15)*1.e7))
    # "trunk": pro_layers.Log(
    #     base='e', slope=1.e7, offset=1.e-15, name='trunk_fn', trainable=False
    # )
}