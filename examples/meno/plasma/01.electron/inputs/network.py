import tensorflow as tf

from inputs.data import data_src

# Architecture properties
# =====================================
architecture = "MENOPlasmaReac"
# Variable name
_name = "electron"

# Extra inputs
# -------------------------------------
callable_lays = {
  "argon": False,
  "electron": True
}

# Layers
# -------------------------------------
noisy_io = {
  "inputs": {
    "lay_id": "GaussianNoise",
    "kwargs": {
      "stddev": 0.02,
      "seed": 0
    }
  },
  "outputs": {
    "lay_id": "GaussianNoise",
    "kwargs": {
      "stddev": 0.02,
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
  "dropout": {"rate": 1e-3},
  "layer_norm": False
}

lays_propty = {

  "log_t_pe": {
    "lay_id": "Log",
    "kwargs": {
      "base": "e",
      "offset": 1e-15,
      "trainable": False
    }
  },

  "ic_xform": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/normalization_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": list(range(5)),
      "xform": {"dir": tf.math.log, "inv": tf.math.exp},
      "trainable": False
    }
  },

  _name: {
    "lay_id": "FlexOpNN",
    "dim": {
      "inp": [5, 1],
      "out": [1]
    },
    "lays_propty": {
      "prenet": {
        "from": "branch",
        "to": "trunk",
        "kwargs": {
          "layer_size": [5, 32, 16],
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
          "layer_size": [5, 64, 32, 16],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_branch",
          "trainable": True,
          **_common
        }
      },
      "trunk": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [1, 64, 32, 16],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_trunk",
          "trainable": True,
          **_common
        }
      }
    }
  }

}
