import tensorflow as tf

from inputs.data import data_src, _path_to_dtb, _path_to_data, \
                        _species, _indices, _nlin_vars

# Architecture properties
# =====================================
architecture = "MENOPlasmaCRReactor"

# masking = True
force_siso = True

# Extra inputs
# -------------------------------------
callable_lays = {k: True for k in _nlin_vars+("states",)}
callable_lays["pe"] = False

return_full = True

# Layers
# -------------------------------------
# noisy_io = {
#   "inputs": {
#     "lay_id": "GaussianNoise",
#     "kwargs": {
#       "stddev": 0.04,
#       "seed": 0
#     }
#   },
#   "outputs": {
#     "lay_id": "GaussianNoise",
#     "kwargs": {
#       "stddev": 0.04,
#       "seed": 0
#     }
#   }
# }

_common = {
  "kernel_initializer": "GlorotNormal",
  "regularizer": {
    "name": "l1+l2",
    "l1": 1e-3,
    "l2": 1e-3
  },
  "dropout": {"rate": 1e-2},
  "layer_norm": False
}


lays_propty = {

  "log_t_ph": {
    "lay_id": "Log",
    "kwargs": {
      "base": "e",
      "offset": 1e-8,
      "trainable": False
    }
  },

  "lin_autoenc": {
    "lay_id": "LinAutoencoder",
    "kwargs": {
      "phi": _path_to_data + "/rom/phi.txt",
      "psi": _path_to_data + "/rom/phi.txt",
      "xref": _path_to_data + "/rom/xref.txt",
      "xscale": _path_to_data + "/rom/xscale.txt",
      "use_log": True,
      "trainable": False
    }
  },

  "mole_mass": {
    "lay_id": "PlasmaCRMoleMass",
    "kwargs": {
      "species": _species,
      "trainable": False
    }
  },

  "ic_xform": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/normalization_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": _indices["x0_norm"],
      "trainable": False
    }
  },

  "fcorr_compute": {
    "lay_id": "StateCorrection",
    "kwargs": {
      "layer_size": [1, 32, 32, 8],
      "activation": ["tanh"]*2 + ["linear"],
      "state_prop": _path_to_data + "/states.txt",
      "trainable": False,
      **_common
    }
  },

  "states_update": {
    "lay_id": "PlasmaCRSolUpdate",
    "kwargs": {
      "species": _species,
      "rates": _path_to_dtb + "/rates/rates.hdf5",
      "use_corr": True,
      "scale_corr": None,
      "epsilon": 1e-15,
      "use_pade": True,
      "use_expmv": False,
      "max_degree": None,
      "max_nb_squarings": None,
      "expm_dtype": None,
      "add_inv_loss": False,
      "inv_loss_weight": 1e-3,
      "trainable": False
    }
  }

}

for _name in _nlin_vars:
  lays_propty[_name] = {
    "lay_id": "FlexOpNN",
    "dim": {
      "inp": [8, 1],
      "out": [1]
    },
    "lays_propty": {
      "prenet": {
        "from": "branch",
        "to": "trunk",
        "kwargs": {
          "layer_size": [8, 128, 128],
          "activation": ["tanh"]*2 + ["linear"],
          "out_xform": None,
          "transformation": "affine",
          "num_coords": 1,
          "name": _name + "_prenet",
          "trainable": False,
          **_common
        }
      },
      "branch": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [8, 128, 128, 64],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_branch",
          "trainable": False,
          **_common
        }
      },
      "trunk": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [1, 128, 128, 64],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_trunk",
          "trainable": False,
          **_common
        }
      },
      "output": {
        "xform": {
          "lay_id": "Transform",
          "kwargs": {
            "xform": tf.math.exp,
            "name": _name + "_xform",
            "trainable": False
          }
        },
        "with_inputs": False
      }
    }
  }
