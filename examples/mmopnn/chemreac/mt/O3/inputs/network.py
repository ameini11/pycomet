# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import tensorflow as tf
from inputs.general import dim
from inputs.data import data_src, columns
from pycomet.networks.elements import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'MMOpNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1': 1.e-5,
        'l2': 1.e-5
    },
    'trainable': True,
    'dropout_kwargs': {'rate': 1.e-2},
    'batch_norm': None,
    'weight_norm': False
}

# Outputs transformations
def arrhenius(x, y):
    K = y[0] * tf.math.pow(x[-1], y[1]) * tf.math.exp(-y[2]/x[-1])
    return tf.math.log(K*1.e9)

opnns_out_xform = {
    'A': lambda y: tf.math.pow(10., y),
    'beta': lambda y: y,
    'Ta': lambda y: y*1.e4
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
        },
        'T': {
            'index': 2,
            'xform': pcm_layers.Normalization(
                mu_sigma_file='./saving/normalization_2_mu_sigma.txt',
                data_files=list(data_src.values()),
                columns=columns['x'][2]
            )
        }
    },

    'opnns': {
        name: {
            'lay_id': 'FlexOpNN',
            'dim': {
                "inp": dim['inp'][:2],
                "out": 1
            },
            'lays_propty': {

                'prenet': {
                    'from': 'branch',
                    'to': 'trunk',
                    'kwargs': {
                        'layer_size': [3, 32, 32, 32],
                        'activation': ['tanh']*3 + ['linear'],
                        'out_xform': None,
                        'transformation': 'affine',
                        'num_coords': 3,
                        **common
                    }
                },

                'branch': {
                    'lay_id': 'FFBlock',
                    'kwargs': {
                        'layer_size': [dim['inp'][0], 32, 32, 32],
                        'activation': ['tanh']*2 + ['linear'],
                        **common
                    }
                },

                'trunk': {
                    'lay_id': 'FFBlock',
                    'kwargs': {
                        'layer_size': [dim['inp'][1], 32, 32, 32],
                        'activation': ['tanh']*2 + ['linear'],
                        **common
                    }
                },

                'output': {
                    'xform': out_xform
                }
            }
        } for name, out_xform in opnns_out_xform.items()
    },

    'output': {
        'xform': arrhenius
    }

}
