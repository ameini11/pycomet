# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import numpy as np
import tensorflow as tf

from inputs.general import dim


# Architecture properties
# =============================================================================
architecture = "CGOpNN"

force_siso = True

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

# Layers ----------------------------------------------------------------------
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

# TimeScaleOpNNs characterization
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
        "branch": [ 2+1, 32, 32,  8],
        "trunk":  [   1, 32, 32,  8]
    },
    names[1]: {
        "branch": [ 2+3, 48, 48, 16],
        "trunk":  [   1, 48, 48, 16]

    },
    names[2]: {
        "branch": [ 2+9, 48, 48, 16],
        "trunk":  [   1, 48, 48, 16]
    },
    names[3]: {
        "branch": [2+27, 48, 48, 16],
        "trunk":  [   1, 48, 48, 16]
    }
}

specie_thermos = {
    names[0]: None,
    **{
        names[i+1]: {
            "specie": {
                "name": "O2",
                "model": {
                   "levels": "O3_UMN/O2.csv",
                   "lev_mapping": f"O3_UMN/O2/LevelsMap_RVE{nb_bins}.csv"
                }
            },
            "thermo": {
                "model": "CoupledModes"
            }
        } for i, nb_bins in enumerate([3,9,27])
    }
}

def get_opnn_cfg(
    dim,
    layer_size
):
    cfg = {
        "lay_id": "AugOpNN",
        "dim": {
            "inp": dim[0],
            "out": dim[1]
        },
        "lays_propty": {
            "branch": [{
                'lay_id': 'FFBlock',
                'kwargs': {
                    "layer_size": layer_size["branch"],
                    "activation": ["tanh"]*(len(layer_size["branch"])-1),
                    **common
                }
            } for _ in range(dim[1])],
            "trunk": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": layer_size["trunk"],
                    "activation": ["sigmoid"]*(len(layer_size["trunk"])-1),
                    **common
                }
            },
            "end_scale": {
                "lay_id": "Dense"
            }
        }
    }
    return cfg

def get_ts_opnn_cfg(
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
                "xform_temp": {
                    "lay_id": "ZeroToOneNormalization",
                    "kwargs": {
                        "mn": 200.0,
                        "mx": 10000.0,
                        "trainable": False
                    }
                },
                # "xform_mass": {
                #     "lay_id": "Log",
                #     "kwargs": {
                #         "base": 10,
                #         "trainable": False
                #     }
                # }
            },
            "mmopnn": {
                "dim": {
                    "inp": dim["inp"],
                    "out": dim["out"]
                },
                "lays_propty": {
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

# Full network characterization
lays_propty = {
    'preinput': {
        'xform': {
            'lay_id': 'CGThermoEncoder',
            'kwargs': {
                'nb_bins': 27
            }
        }
    },
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
                    "base": "e",
                    "slope": 1.e7,
                    "offset": 1.e-14,
                    "trainable": False
                }
            }
        }
    },
    "temp": {
        "lay_id": "AugOpNN",
        "dim": {
            "inp": dim["inp"],
            "out": 1
        },
        "lays_propty": {
            "branch": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": [dim["inp"][0], 32, 32, 8],
                    "activation": ["tanh"]*2 + ["linear"],
                    **common
                }
            },
            "trunk": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": [dim["inp"][1], 32, 32, 8],
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
    },
    "mass": {
        name: get_ts_opnn_cfg(
            dim=dims[name],
            layer_size=layer_sizes[name],
            specie_thermo=specie_thermos[name]
        ) for name in names
    }
}
