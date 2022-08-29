# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import inputs
import tensorflow as tf

from tensorflow.python.keras import layers as tf_layers
from pycomet.utilities.model import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'OpNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1':   1.e-4,
        'l2':   1.e-4
    },
    'dropout_rate': 0,
    'batch_norm': None,
    'weight_norm': False
}

lays_propty = {

    'input': {
        'branch': {
            'index': 0,
            'xform': lambda x: x
        },
        'trunk': {
            'index': 1,
            'xform': lambda x: x
        }
    },

    'prenet': {
        'block': 'PreNetBlock',
        'layer_size': [inp_gen.dim['inp'][0], 8, 8, 8, 2],
        'activation': ['sigmoid']*3 + ['linear'],
        'out_xform': None,
        'transformation': 'projective',
        'num_coordinates': 1,
        'trainable': True,
        **common
    },

    'branch': [{
        'block': 'FFBlock',
        'layer_size': [inp_gen.dim['inp'][0], 16, 16, 2],
        'activation': ['sigmoid']*2 + ['linear'],
        'trainable': True,
        **common
    }]*2,

    'trunk': {
        'block': 'FFBlock',
        'layer_size': [inp_gen.dim['inp'][1], 16, 16, 2],
        'activation': ['sigmoid']*2 + ['linear'],
        'trainable': True,
        **common
    },

    'end_scale': {
        'type': 'bias'
    }

}
