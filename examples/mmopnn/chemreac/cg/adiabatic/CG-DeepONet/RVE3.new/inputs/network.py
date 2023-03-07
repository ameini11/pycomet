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
names = ["TS0", "TS1"]

dims = {
    names[0]: {
        "inp": dim["inp"],
        "out": 2
    },
    names[1]: {
        "inp": dim["inp"],
        "out": 3
    }
}

layer_sizes = {
    names[0]: {
        "branch": [ 2+1, 32, 32, 16],
        "trunk":  [   1, 32, 32, 16]
    },
    names[1]: {
        "branch": [ 2+3, 32, 32, 16],
        "trunk":  [   1, 32, 32, 16]

    }
}

# TimeScaleOpNN - Physics
# -------------------------------------
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
        } for i, nb_bins in enumerate((3,))
    }
}

groups_graph = {0: [0, 1, 2]}

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
branch_multilays = [{
    "lay_id": "MultiLays",
    "kwargs": {
        "lays_propty": {
            0: {
                "lay_id": "Log",
                "kwargs": {
                    "base": 'e',
                    "trainable": False
                }
            },
            1: {
                "lay_id": "ZeroToOneNormalization",
                "kwargs": {
                    "xform": np.log,
                    "mn": [3.0e-3] + [1.0e-14]*nb_bins + [2.9e3],
                    "mx": [1.7e-2] + [  1.0e0]*nb_bins + [9.0e3],
                    "trainable": False
                }
            }
        }
    }
} for nb_bins in (1,3)]

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
                    "base": "e",
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
            'prenet': {
                'from': 'branch',
                'to': 'trunk',
                'kwargs': {
                    'layer_size': [dim['inp'][0], 16, 16],
                    'activation': ['tanh']*2 + ['linear'],
                    'out_xform': None,
                    'transformation': 'affine',
                    'num_coords': 1,
                    **common
                }
            },
            "branch": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": [dim["inp"][0], 32, 32, 16],
                    "activation": ["tanh"]*2 + ["linear"],
                    **common
                }
            },
            "trunk": {
                "lay_id": "FFBlock",
                "kwargs": {
                    "layer_size": [dim["inp"][1], 32, 32, 16],
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
