# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


from inputs.general import dim
from inputs.data import data_src, columns
from pycomet.networks.elements import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'OpNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1': 1.e-4,
        'l2': 1.e-4
    },
    'trainable': True,
    'dropout_kwargs': {'rate': 1.e-3},
    'batch_norm': None,
    'weight_norm': False
}

lays_propty = {

    'input': {
        'branch': {
            'index': 0,
            'xform': pcm_layers.Normalization(
                mu_sigma_file='./saving/normalization_mu_sigma.txt',
                data_files=list(data_src.values()),
                columns=columns['x'][0]
            )
        },
        'trunk': {
            'index': 1,
            'xform': pcm_layers.Normalization(
                mu_sigma_file='./saving/normalization_1_mu_sigma.txt',
                data_files=list(data_src.values()),
                columns=columns['x'][1]
            )
        }
    },

    'branch': {
        'block': 'FFBlock',
        'layer_size': [dim['inp'][0], 32, 32, 32],
        'activation': ['selu']*3,
        **common
    },

    'trunk': {
        'block': 'FFBlock',
        'layer_size': [dim['inp'][1], 32, 32, 32],
        'activation': ['tanh']*2 + ['linear'],
        **common
    },

    'end_scale': {
        'type': 'dense'
    }

}
