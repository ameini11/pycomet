# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import numpy as np
import tensorflow as tf

from inputs.general import dim
from inputs.physics import bins
from pycomet import config as cfg
from keras import layers as tf_layers


# Architecture properties
# =============================================================================
architecture = "NeqOpNN"

force_siso = False

# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------

# Layers - Common properties
# -------------------------------------
common = {
    "kernel_initializer": "GlorotNormal",
    "regularizer": {
        "name": "l1+l2",
        "l1": 1.e-3,
        "l2": 1.e-3
    },
    "trainable": True,
    "dropout_kwargs": {"rate": 0.0},
    "batch_norm": None,
    "weight_norm": False
}

# -----------------------------------------------------------------------------
# MMOpNN
# -----------------------------------------------------------------------------

# MMOpNN - Simple properties
# -------------------------------------
names = ["NEQ0", "NEQ1", "NEQ2"]

dims = {
    names[0]: {
        "inp": dim["inp"],
        "out": 1
    },
    names[1]: {
        "inp": dim["inp"],
        "out": 3
    },
    names[2]: {
        "inp": dim["inp"],
        "out": [3]*3
    }
}

layer_sizes = {
    names[0]: {
        "branch": [2+27, 32, 32, 16],
        "trunk":  [   1, 32, 32, 16]
    },
    names[1]: {
        "branch": [2+27, 32, 32, 16],
        "trunk":  [   1, 32, 32, 16]
    },
    names[2]: {
        "branch": [2+27, 32, 32, 16],
        "trunk":  [   1, 32, 32, 16]
    }
}

# MMOpNN - Single OpNN config
# -------------------------------------
def get_opnn_cfg(
    dim,
    layer_size
):
    nb_lays = {k: len(l) for k, l in layer_size.items()}
    cfg = {
        "lay_id": "FlexOpNN",
        "dim": {
            "inp": dim[0],
            "out": dim[1]
        },
        "lays_propty": {
            "branch": [{
                'lay_id': 'FFBlock',
                'kwargs': {
                    "layer_size": layer_size["branch"],
                    "activation": ["tanh"]*(nb_lays["branch"]-2) + ["linear"],
                    **common
                }
            } for _ in range(dim[1])],
            "trunk": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": layer_size["trunk"],
                    "activation": ["sigmoid"]*(nb_lays["trunk"]-2) + ["linear"],
                    **common
                }
            }
        }
    }
    return cfg

# MMOpNN - Full network config
# -------------------------------------
def get_mmopnn_cfg(
    dim,
    layer_size
):
    dim["inp"] = tf.nest.flatten(dim["inp"])
    dim["out"] = tf.nest.flatten(dim["out"])
    cfg = {
        "dim": {
            "inp": dim["inp"],
            "out": np.sum(dim["out"])
        },
        "lays_propty": {
            "prenet": {
                'from': 'branch',
                'to': 'trunk',
                "kwargs": {
                    'layer_size': [dim['inp'][0], 16, 16],
                    'activation': ['tanh']*2 + ['linear'],
                    'out_xform': None,
                    'transformation': 'affine',
                    'num_coords': 1,
                    **common
                }
            },
            "opnns": {
                f"opnn_{i+1}": get_opnn_cfg(
                    dim=[dim["inp"], d],
                    layer_size=layer_size
                ) for i, d in enumerate(dim["out"])
            }
        }
    }
    return cfg

# -----------------------------------------------------------------------------
# Complete network characterization
# -----------------------------------------------------------------------------
branch_multilays = {
    "lay_id": "MultiLays",
    "kwargs": {
        "lays_propty": {
            0: {
                "lay_id": "Log",
                "kwargs": {
                    "base": 10,
                    "offset": 1.e-14,
                    "trainable": False
                }
            },
            1: {
                "lay_id": "ZeroToOneNormalization",
                "kwargs": {
                    "xform": np.log10,
                    "mn": [3.0e-3] + [1.0e-14]*bins[-1] + [2.9e3],
                    "mx": [1.7e-2] + [9.99e-1]*bins[-1] + [9.0e3],
                    "trainable": False
                }
            }
        }
    }
}

lays_propty = {
    
    # 'preinput': {
    #     'xform': {
    #         'lay_id': 'CGThermoEncoder',
    #         'kwargs': {
    #             'bins[-1]': 27
    #         }
    #     }
    # },

    "input": {
        "branch": {
            "index": 0,
            "xform": branch_multilays
        },
        "trunk": {
            "index": 1,
            "xform": {
                "lay_id": "Log",
                "kwargs": {
                    "base": "e",
                    "slope": 1.e8,
                    "offset": 1.e-12,
                    "trainable": False
                }
            }
        }
    },

    "mmopnns": {
        name: get_mmopnn_cfg(
            dim=dims[name],
            layer_size=layer_sizes[name]
        ) for name in names
    }
    
}
