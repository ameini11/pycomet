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
    name: {
        "branch": [2+27, 32, 32, 16],
        "trunk":  [1, 64, 64, 16]
    } for name in names
}

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

augmented = False

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name':         'l1+l2',                                      # 'fine_tuning'
    'l1':           1.e-3,
    'l2':           1.e-3,
    'weights_path': './saving/'+inp_gen.model_name+'_weights.h5'  # For `FineTuningRegularizer`
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
def branch_fn_(x):
    T_norm = pro_layers.LinearNorm(
        *inp_gen.input_space['Ttr'], trainable=False, name='branch_fn'
    )(tf.gather(x, indices=[0], axis=1))
    y = tf_layers.Concatenate(axis=1)([T_norm, x[:,1:]])
    return y

# input_fn = {
#     # "branch": branch_fn_,
#     "branch": tf_layers.Lambda(lambda x: x, name='branch_fn'),
#     "trunk": pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
#     # "trunk": tf_layers.Lambda(lambda x: tf.math.reciprocal((x+1.e-15)*1.e7))
#     # "trunk": pro_layers.Log(
#     #     base='e', slope=1.e7, offset=1.e-15, name='trunk_fn', trainable=False
#     # )
# }