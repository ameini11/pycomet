# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import inputs


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
    'weight_norm': False,
    'trainable': True
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
        'block_type': 'PreNetBlock',
        'layer_size': [inputs.dim['inp'][0], 8, 8, 8, 2],
        'activation': ['sigmoid']*3 + ['linear'],
        'out_xform': None,
        'transformation': 'projective',
        'num_coordinates': 1,
        **common
    },

    'branch': [{
        'block_type': 'FFBlock',
        'layer_size': [inputs.dim['inp'][0], 16, 16, 2],
        'activation': ['sigmoid']*2 + ['linear'],
        **common
    }]*2,

    'trunk': {
        'block_type': 'FFBlock',
        'layer_size': [inputs.dim['inp'][1], 16, 16, 2],
        'activation': ['sigmoid']*2 + ['linear'],
        **common
    },

    'end_scale': {
        'type': 'bias'
    }

}
