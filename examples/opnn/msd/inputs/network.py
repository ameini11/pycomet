# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


from inputs.general import dim


# Architecture properties
# =============================================================================
architecture = 'FlexOpNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1':   1.e-4,
        'l2':   1.e-4
    },
    'dropout_kwargs': {'rate': 0.},
    'batch_norm': None,
    'weight_norm': False,
    'trainable': True
}

lays_propty = {

    'input': {
        'branch': {
            'index': 0,
            'xform': None
        },
        'trunk': {
            'index': 1,
            'xform': None
        }
    },

    'prenet': {
        'from': 'branch',
        'to': 'trunk',
        'kwargs': {
            'layer_size': [dim['inp'][0], 8, 8, 8],
            'activation': ['sigmoid']*3 + ['linear'],
            'out_xform': None,
            'transformation': 'projective',
            'num_coords': 1,
            **common
        }
    },

    'branch': [{
        'lay_id': 'FFBlock',
        'kwargs': {
            'layer_size': [dim['inp'][0], 16, 16, 2],
            'activation': ['sigmoid']*2 + ['linear'],
            **common
        }
    } for _ in range(2)],

    'trunk': {
        'lay_id': 'FFBlock',
        'kwargs': {
            'layer_size': [dim['inp'][1], 16, 16, 2],
            'activation': ['sigmoid']*2 + ['linear'],
            **common
        }
    },

    # 'end_scale': {
    #     'lay_id': 'bias',
    #     'kwargs': {}
    # }

}
