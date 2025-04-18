import numpy as np
import tensorflow as tf

from inputs.data import data_src, _path_to_data

# Architecture properties
# =====================================
architecture = "MENONonEqReac"
# Variable name
_name = "temp"

# Extra inputs
# -------------------------------------
qoi_indices = {
  "mass": 0,
  "temp": 1
}
callable_lays = {
  "mass": False,
  "temp": False,
  "dist": False
}
callable_lays[_name] = True

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
    "l1": 1e-3,
    "l2": 1e-3
  },
  "dropout": {"rate": 1e-2},
  "layer_norm": False
}

lays_propty = {

  "log_t_pa": {
    "lay_id": "Log",
    "kwargs": {
      "base": "e",
      "offset": 1e-12,
      "trainable": False
    }
  },

  "thermo_update": {
    "lay_id": "ThermoUpdate",
    "kwargs": {
      "specie_thermos": {
        "filename": _path_to_data + "/../datagen/physics/specie_thermos.json",
        "config": None
      },
      "trainable": False
    }
  },

  "ic_xform": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/normalization_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": (96 + np.arange(4)).tolist(),
      "xform": {"dir": tf.math.log, "inv": tf.math.exp},
      "trainable": False
    }
  },

  _name: {
    "lay_id": "FlexOpNN",
    "dim": {
      "inp": [4, 1],
      "out": [1]
    },
    "lays_propty": {
      "prenet": {
        "from": "branch",
        "to": "trunk",
        "kwargs": {
          "layer_size": [4, 128, 64],
          "activation": ["tanh"]*2 + ["linear"],
          "out_xform": None,
          "transformation": "affine",
          "num_coords": 1,
          "name": _name + "_prenet",
          "trainable": True,
          **_common
        }
      },
      "branch": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [4, 128, 64, 32],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_branch",
          "trainable": True,
          **_common
        }
      },
      "trunk": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [1, 128, 64, 32],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_trunk",
          "trainable": True,
          **_common
        }
      },
      "output": {
        "xform": {
          "lay_id": "Transform",
          "kwargs": {
            "xform": tf.math.sigmoid,
            "name": _name + "_xform",
            "trainable": False
          }
        },
        "with_inputs": False
      }
    }
  }

}
