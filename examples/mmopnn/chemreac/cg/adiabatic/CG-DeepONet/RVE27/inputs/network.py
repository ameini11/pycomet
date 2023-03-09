# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import numpy as np
import tensorflow as tf

from inputs.general import dim
from inputs.physics import bins, mol_specie_thermos


# Architecture properties
# =============================================================================
architecture = "CGOpNN"

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
# TimeScaleOpNN
# -----------------------------------------------------------------------------

# TimeScaleOpNN - Simple properties
# -------------------------------------
names = ["TS0", "TS1", "TS2", "TS3"]

dims = {
    names[0]: {
        "inp": dim["inp"],
        "out": 2
    },
    names[1]: {
        "inp": dim["inp"],
        "out": 3
    },
    names[2]: {
        "inp": dim["inp"],
        "out": [3]*3
    },
    names[3]: {
        "inp": dim["inp"],
        "out": [3]*9
    }
}

layer_sizes = {
    names[0]: {
        "branch": [ 2+1, 16, 16, 4],
        "trunk":  [   1, 16, 16, 4]
    },
    names[1]: {
        "branch": [ 2+3, 16, 16, 4],
        "trunk":  [   1, 16, 16, 4]

    },
    names[2]: {
        "branch": [ 2+9, 16, 16, 4],
        "trunk":  [   1, 16, 16, 4]
    },
    names[3]: {
        "branch": [2+27, 16, 16, 4],
        "trunk":  [   1, 16, 16, 4]
    }
}

# TimeScaleOpNN - Physics
# -------------------------------------
specie_thermos = {
    names[0]: None,
    **{n: sp for (n, sp) in zip(names[1:], mol_specie_thermos[1:])}
}

groups_graph = {
    0: {
        0: {
            0: [ 0, 1, 2],
            1: [ 3, 4, 5],
            2: [ 6, 7, 8]
        },
        1: {
            3: [ 9,10,11],
            4: [12,13,14],
            5: [15,16,17]
        },
        2: {
            6: [18,19,20],
            7: [21,22,23],
            8: [24,25,26]
        }
    }
}

# TimeScaleOpNN - Single OpNN config
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

# TimeScaleOpNN: Full network config
# -------------------------------------
prenet = {
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
}

branch_multilays = [{
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
                    "mn": [3.0e-3] + [1.0e-14]*nb_bins + [2.9e3],
                    "mx": [1.7e-2] + [9.99e-1]*nb_bins + [9.0e3],
                    "trainable": False
                }
            }
        }
    }
} for nb_bins in bins]

def get_ts_opnn_cfg(
    index,
    dim,
    layer_size,
    specie_thermo=None
):
    dim["inp"] = tf.nest.flatten(dim["inp"])
    dim["out"] = tf.nest.flatten(dim["out"])
    cfg = {
        "dim": {
            "inp": dim["inp"],
            "out": np.sum(dim["out"])
        },
        "lays_propty": {
            "input": {
                "multilays": branch_multilays[index]
            },
            "mmopnn": {
                "dim": {
                    "inp": dim["inp"],
                    "out": dim["out"]
                },
                "lays_propty": {
                    "prenet": prenet,
                    "opnns": {
                        f"opnn_{i+1}": get_opnn_cfg(
                            dim=[dim["inp"], d],
                            layer_size=layer_size
                        ) for i, d in enumerate(dim["out"])
                    }
                }
            },
            "output": {
                "xform": {
                    "lay_id": "CGEquilibrium",
                    "kwargs": {
                        "output_dim": dim["out"],
                        "use_pf_int": True if specie_thermo is not None else False,
                        "specie_thermo": specie_thermo,
                        "process": ["adiabatic", "isochoric"]
                    }
                }
            }
        }
    }
    return cfg

# -----------------------------------------------------------------------------
# Complete network characterization
# -----------------------------------------------------------------------------
lays_propty = {
    
    # 'preinput': {
    #     'xform': {
    #         'lay_id': 'CGThermoEncoder',
    #         'kwargs': {
    #             'nb_bins': 27
    #         }
    #     }
    # },
    
    "input": {
        "branch": {
            "index": 0,
            "xform": None
        },
        "trunk": {
            "index": 1,
            "xform": {
                "lay_id": "Log",
                "kwargs": {
                    "base": 10,
                    "slope": 1.e8,
                    "offset": 1.e-12,
                    "trainable": False
                }
            }
        }
    },
    
    "mass": {
        name: get_ts_opnn_cfg(
            index=i,
            dim=dims[name],
            layer_size=layer_sizes[name],
            specie_thermo=specie_thermos[name]
        ) for i, name in enumerate(names)
    },
    
    "temp": {
        "lay_id": "FlexOpNN",
        "dim": {
            "inp": dim["inp"],
            "out": 1
        },
        "lays_propty": {
            "input": {
                "branch": {
                    "index": 0,
                    "xform": branch_multilays[-1]
                },
                "trunk": {
                    "index": 1,
                    "xform": None
                }
            },
            "prenet": prenet,
            "branch": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": [dim["inp"][0], 16, 16, 4],
                    "activation": ["tanh"]*2 + ["linear"],
                    **common
                }
            },
            "trunk": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": [dim["inp"][0], 16, 16, 4],
                    "activation": ["sigmoid"]*2 + ["linear"],
                    **common
                }
            },
            "output": {
                "xform": {
                    "lay_id": "Limiter",
                    "kwargs": {
                        "limits": [200.0, 10000.0],
                        "xform_in": tf.math.exp
                    }
                }
            }
        }
    }
    
}
