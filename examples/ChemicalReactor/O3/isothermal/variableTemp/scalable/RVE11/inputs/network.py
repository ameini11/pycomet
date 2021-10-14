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
architecture = 'IsoThermChemReacNN'

dim = {
    "inp": [11, 1],
    "out": [2, 3, 9]
}


# TimeScaleNNs characterization -----------------------------------------------
names = ['TS0', 'TS1', 'TS2']

tsnn_dim = {
    names[0]: {
        "inp": [2+1, 1],
        "out": 2
    },
    names[1]: {
        "inp": [2+3, 1],
        "out": 3
    },
    names[3]: {
        "inp": [2+9, 1],
        "out": [3,3,3]
    }
}

layer_size = {
    names[0]: {
        "branch": [tsnn_dim[names[0]]['inp'][0], 64, 64, 64],
        "trunk":  [tsnn_dim[names[0]]['inp'][1], 64, 64, 64]
    },
    names[1]: {
        "branch": [tsnn_dim[names[1]]['inp'][0], 64, 64, 64],
        "trunk":  [tsnn_dim[names[1]]['inp'][1], 128, 128, 64]
    },
    names[2]: {
        "branch": [tsnn_dim[names[2]]['inp'][0], 64, 64, 64],
        "trunk":  [tsnn_dim[names[2]]['inp'][1], 128, 128, 64]
    }
}

activation = {
    name: {
        "branch": 'tanh',
        "trunk":  'sigmoid'
    } for name in names
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
    'name':       'l1+l2',                                  # 'fine_tuning'
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


# Physical quantities
# =============================================================================
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

temp_limits = inp_data.input_space['Ttr']

grouping = {k: v for k, v in zip(names, [None, 'RVE3', 'RVE9'])}

levels = inp_data.paths['database']+'/levels/O3_UMN/O2/FromUMN_Sorted.inp'

lev_bin_map = {
    n: inp_data.paths['database']+'/grouping/O3_UMN/O2/LevelsMap_'+g+'.csv' \
        if g != None else g for n, g in grouping.items()
}
