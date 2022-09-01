# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


from inputs.general import dim
from inputs.data import data_src, columns
from pycomet.networks.elements import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'FFNN'

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
        'xform': pcm_layers.Normalization(
            mu_sigma_file='./saving/normalization_mu_sigma.txt',
            data_files=list(data_src.values()),
            columns=columns['x']
        )
    },

    'hidden_block': {
        'block_type': 'FFBlock',
        'layer_size': [dim['inp'], 16, 16, 16, 1],
        'activation': ['tanh']*3 + ['linear'],
        **common
    }

}
