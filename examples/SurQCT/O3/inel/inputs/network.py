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


import inputs.general as inp_gen

from pycomet.utilities.model import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'OpNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1': 1.e-5,
        'l2': 1.e-6
    },
    'trainable': True,
    'dropout_rate': 1.e-3,
    'batch_norm': None,
    'weight_norm': False
}

lays_propty = {

    'input': {
        'branch': {
            'index': 0,
            'xform': pcm_layers.Normalization(
                file=['data/train/pts.csv', 'data/valid/pts.csv'],
                columns=[0,3]
            )
        },
        'trunk': {
            'index': 1,
            'xform': pcm_layers.Normalization(
                file=['data/train/pts.csv', 'data/valid/pts.csv'],
                columns=[6,9]
            )
        }
    },

    'branch': {
        'block': 'FFBlock',
        'layer_size': [inp_gen.dim['inp'][0], 64, 64, 32],
        'activation': ['selu']*3,
        **common
    },

    'trunk': {
        'block': 'FFBlock',
        'layer_size': [inp_gen.dim['inp'][1], 64, 64, 64, 32],
        'activation': ['tanh']*3 + ['linear'],
        **common
    },

    'end_scale': {
        'type': 'dense'
    }

}
