# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import numpy as np
import tensorflow as tf

from inputs.general import dim
from inputs.data import data_src, cols
from inputs.physics import specie_thermo, process
from pycomet.networks.elements import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'MMOpNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1': 5.e-4,
        'l2': 5.e-4
    },
    'trainable': True,
    'dropout_kwargs': {'rate': 0.},
    'batch_norm': None,
    'weight_norm': False
}

lays_propty = {

    'input': {
        'branch': {
            'index': 0,
            'xform': {
                'lay_id': 'StdNormalization',
                'kwargs': {
                    'mu_sigma_file': './saving/normalization_mu_sigma.txt',
                    'data_files': [data_src[k]['ics'] for k in ('train','valid')],
                    'columns': cols['x'][0],
                    'trainable': False
                }
            }
        },
        'trunk': {
            'index': 1,
            'xform': {
                'lay_id': 'Log',
                'kwargs': {
                    'base': 'e',
                    'slope': 1.e7,
                    'offset': 1.e-14,
                    'trainable': False
                }
            }
        }
    },

    'opnns': {
        name: {
            'lay_id': 'FlexOpNN',
            'dim': {
                "inp": dim['inp'],
                "out": 1
            },
            'lays_propty': {

                'prenet': {
                    'from': 'branch',
                    'to': 'trunk',
                    'kwargs': {
                        'layer_size': [dim['inp'][0], 24, 24],
                        'activation': ['tanh']*2 + ['linear'],
                        'out_xform': None,
                        'transformation': 'affine',
                        'num_coords': 1,
                        **common
                    }
                },

                'branch': {
                    'lay_id': 'FFBlock',
                    'kwargs': {
                        'layer_size': [dim['inp'][0], 64, 64, 24],
                        'activation': ['tanh']*2 + ['linear'],
                        **common
                    }
                },

                'trunk': {
                    'lay_id': 'FFBlock',
                    'kwargs': {
                        'layer_size': [dim['inp'][1], 64, 64, 24],
                        'activation': ['sigmoid']*2 + ['linear'],
                        **common
                    }
                }
            }
        } for name in ('mass', 'temp')
    },

    'output': {
        'xform': {
            'lay_id': 'TwoTempThermoLay',
            'kwargs': {
                'process': process,
                'specie_thermo': specie_thermo,
                'coding_type': 'temperature'
            }
        },
        'with_inputs': True
    }

}
