# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import tensorflow as tf
from inputs.general import dim
from inputs.data import data_src, columns
from inputs.physics import specie_thermo, process
from pycomet.networks.elements import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'TwoTempOneElemNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1': 1.e-3,
        'l2': 1.e-3
    },
    'trainable': True,
    'dropout_kwargs': {'rate': 0.},
    'batch_norm': None,
    'weight_norm': False
}

def output_xform(y):
    y =  tf.math.sigmoid(y)
    return tf.concat([1.-y, y],1)

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
            'xform': pcm_layers.Log()
        }
    },

    'opnns': {
        name: {
            'lay_id': 'FlexOpNN',
            'dim': {
                "inp": dim['inp'],
                "out": 2
            },
            'lays_propty': {

                'prenet': {
                    'from': 'branch',
                    'to': 'trunk',
                    'kwargs': {
                        'layer_size': [dim['inp'][0], 32, 32],
                        'activation': ['tanh']*2 + ['linear'],
                        'out_xform': None,
                        'transformation': 'similarity',
                        'num_coords': 1,
                        **common
                    }
                },

                'branch': {
                    'lay_id': 'FFBlock',
                    'kwargs': {
                        'layer_size': [dim['inp'][0], 32, 32, 16],
                        'activation': ['tanh']*3,
                        **common
                    }
                },

                'trunk': {
                    'lay_id': 'FFBlock',
                    'kwargs': {
                        'layer_size': [dim['inp'][1], 32, 32, 16],
                        'activation': ['tanh']*3,
                        **common
                    }
                },

                'output': {
                    'xform': output_xform
                }
            }
        } for name in ('mass', 'temp')
    },

    'output': {
        'lay_id': 'TwoTempThermoLay',
        'kwargs': {
            'process': process,
            'specie_thermo': specie_thermo,
            'out_indices': {'Y': {0: [0,1]}, 'T': {1: [0,1]}},
            'inp_indices': {'e': {0: -1}}
        }
    }

}
