import json
import tensorflow as tf

from inputs.data import data_src, _path_to_data, _species, _indices

# Architecture properties
# =====================================
architecture = "MENOPlasmaCRReactor"
# Variable name
_name = "n_Ar1"

# Extra inputs
# -------------------------------------
callable_lays = {
  "n_em": False,
  "n_Ar1": False,
  "Th": False,
  "Te": False,
  "pe": False,
  "states": False
}
callable_lays[_name] = True

# Time basis
# -------------------------------------
_path_to_basis = _path_to_data + "/../tools/output/lowrank_time/eps1/"
with open(_path_to_basis + "/rdim.json") as file:
  _rdim = json.load(file)

# Layers
# -------------------------------------
noisy_io = {
  "inputs": {
    "lay_id": "GaussianNoise",
    "kwargs": {
      "stddev": 0.04,
      "seed": 0
    }
  },
  "outputs": {
    "lay_id": "GaussianNoise",
    "kwargs": {
      "stddev": 0.04,
      "seed": 0
    }
  }
}

_common = {
  "kernel_initializer": "GlorotNormal",
  "regularizer": {
    "name": "l1+l2",
    "l1": 1e-4,
    "l2": 1e-4
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

  _name: {
    "lay_id": "PODOpNN",
    "dim": {
      "inp": [8, 1],
      "out": [1]
    },
    "lays_propty": {
      "branch": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [8, 128, 128, _rdim[_name]],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_branch",
          "trainable": True,
          **_common
        }
      },
      "trunk": {
        "lay_id": "FFBlock",
        "kwargs": {
          "interpolating": {
            "train": _path_to_basis + f"/basis_{_name}.hdf5",
            "order": 2,
            "smoothing": 0,
            "rank": 2,
            "tol": 1e-4
          },
          "name": _name + "_trunk",
          "trainable": False
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

}
