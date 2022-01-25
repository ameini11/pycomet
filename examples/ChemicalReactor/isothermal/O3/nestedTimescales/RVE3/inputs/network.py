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
architecture = 'IsoThermChemReacNN'

# TimeScaleNNs characterization -----------------------------------------------
names = ['TS0', 'TS1']

tsnn_dim = {
    names[0]: {
        "inp": inp_gen.dim['inp'],
        "out": 2
    },
    names[1]: {
        "inp": inp_gen.dim['inp'],
        "out": 3
    }
}

layer_size = {
    names[0]: {
        "branch": [2+1, 32, 32, 32],
        "trunk":  [1, 64, 64, 32]
    },
    names[1]: {
        "branch": [2+3, 32, 32, 32],
        "trunk":  [1, 64, 64, 32]
    }
}

activation = {
    name: {
        "branch": 'sigmoid',   # elu/gelu
        "trunk":  'sigmoid'
    } for name in names
}

kernel_initializer = 'GlorotNormal' #'GlorotNormal' 'zeros'

block = 'ResPIBlock' # 'FFBlock' 'ResPIBlock'

opnn_end_scale = 'bias' # 'dense'

stacked = True

cross_encoder = {
    "branch": False,
    "trunk": False
}

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name':         'fine_tuning',            # 'l1+l2', 'fine_tuning'
    'l1':           1.e-3,
    'l2':           1.e-3,
    'weights_path': './saving/'+inp_gen.model_name+'_weights.h5'  # For `FineTuningRegularizer`
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
input_fn = {
    "branch": tf_layers.Lambda(lambda x: x, name='branch_fn'),
    # "trunk": tf_layers.Lambda(lambda x: tf.math.reciprocal((x+1.e-15)*1.e7))
    "trunk": pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
    # "trunk": pro_layers.Log(
    #     base='e', slope=1.e7, offset=1.e-15, name='trunk_fn', trainable=False
    # )
}
