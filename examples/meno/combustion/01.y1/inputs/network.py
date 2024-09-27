import tensorflow as tf

from inputs.data import data_src, _columns

# Architecture properties
# =====================================
architecture = "MENOPOLLUReac"

# Extra inputs
# -------------------------------------
callable_lays = {
  "y_lin": False,
  "y_nlin": True
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
    "l1": 1e-3,
    "l2": 1e-3
  },
  "dropout": {"rate": 1e-2},
  "layer_norm": False
}

lays_propty = {

  "log_t": {
    "lay_id": "Log",
    "kwargs": {
      "base": "e",
      "offset": 1e-9,
      "trainable": False
    }
  },

  "ic_xform": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/normalization_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": _columns["x"][2],
      "xform": {"dir": tf.math.log, "inv": tf.math.exp},
      "trainable": False
    }
  }

}

for i in (1,):
  _name = f"y{i}"
  lays_propty[_name] = {
    "lay_id": "FlexOpNN",
    "dim": {
      "inp": [7, 1],
      "out": [1]
    },
    "lays_propty": {
      "prenet": {
        "from": "branch",
        "to": "trunk",
        "kwargs": {
          "layer_size": [7, 64, 32],
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
          "layer_size": [7, 128, 64, 32],
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
          "lay_id": "StdNormalization",
          "kwargs": {
            "mean_var_file": "./saving/std_normalization_1_mean_var.txt",
            "data_files": [data_src[k] for k in ("train", "valid")],
            "columns": _columns["y"][0],
            "xform": {"dir": tf.math.log, "inv": tf.math.exp},
            "trainable": False,
            "invert": True
          }
        },
        "with_inputs": False
      }
    }
  }
