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
import inputs.model as inp_mdl

from tensorflow.keras import layers as tf_layers
from prode.utilities.model import layers as pro_layers


# Architecture properties
# =============================================================================
architecture = 'IsoTChemicalReactor'

# groups_graph = [0]

# groups_graph = {
#     0: [0,1,2]
# }

groups_graph = {
    0: {
        0: [0,1,2],
        1: [3,4,5],
        2: [6,7,8]
    }
}

dim = {
    "inp": [11, 1],
    "out": [2, 3, 9]
}

# DeepONets characterization --------------------------------------------------
names = ['TS0', 'TS1', 'TS2']

ts_layer_dim = {
    names[0]: {
        "inp": dim['inp'],
        "out": 2
    },
    names[1]: {
        "inp": dim['inp'],
        "out": 3
    },
    names[3]: {
        "inp": dim['inp'],
        "out": [3,3,3]
    }
}

layer_size = {
    names[0]: {
        "branch": [opnn_dim[names[0]]['inp'][0], 64, 64, 64],
        "trunk":  [opnn_dim[names[0]]['inp'][1], 64, 64, 64]
    },
    names[1]: {
        "branch": [opnn_dim[names[1]]['inp'][0], 64, 64, 64],
        "trunk":  [opnn_dim[names[1]]['inp'][1], 128, 128, 64]
    },
    names[2]: {
        "branch": [opnn_dim[names[2]]['inp'][0], 64, 64, 64],
        "trunk":  [opnn_dim[names[2]]['inp'][1], 128, 128, 64]
    }
}

act_baseline = {
    "branch": 'tanh',
    "trunk":  'sigmoid'
}
activation = {name: act_baseline for name in names}

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
    'weights_SP': './saving/'+inp_mdl.name+'_weights.h5'    # Path to for `FineTuningRegularizer`
}

dropout_rate = 0


# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
input_fn = {
    "branch": tf_layers.Lambda(lambda x: x, name='branch_fn'),
    "trunk": pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
}
