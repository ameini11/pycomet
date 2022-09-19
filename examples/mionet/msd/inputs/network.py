# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


from inputs.general import dim


# Architecture properties
# =============================================================================
architecture = 'MIONet'

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
        'params': {
            'index': 0,
            'xform': None
        },
        'ics': {
            'index': 1,
            'xform': None
        },
        'time': {
            'index': 2,
            'xform': None
        }
    },

    'prenet':  {

        'shifter': {
            'from': ['params', 'ics'],
            'to': 'time',
            'lay_id': 'PreNetBlock',
            'kwargs': {
                'layer_size': [sum(dim['inp'][:2]), 8, 8, 8, 2],
                'activation': ['sigmoid']*3 + ['linear'],
                'out_xform': None,
                'transformation': 'projective',
                'num_coords': 1,
                **common
            }
        }

    },

    'branch': {

        'params': {
            'lay_id': 'FFBlock',
            'kwargs': {
                'layer_size': [dim['inp'][0], 16, 16, 2],
                'activation': ['sigmoid']*2 + ['linear'],
                **common
            }
        },

        'ics': {
            'lay_id': 'FFBlock',
            'kwargs': {
                'layer_size': [dim['inp'][1], 16, 16, 2],
                'activation': ['sigmoid']*2 + ['linear'],
                **common
            }
        }

    },

    'trunk': {

        'time': {
            'lay_id': 'FFBlock',
            'kwargs': {
                'layer_size': [dim['inp'][2], 16, 16, 2],
                'activation': ['sigmoid']*2 + ['linear'],
                **common
            }
        }

    },

    'end_scale': {
        'lay_id': 'bias',
        'kwargs': {}
    }

}
