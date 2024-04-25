import numpy as np

from inputs.data import data_src

# Architecture properties
# =====================================
architecture = "LinNonlinME"

# Extra inputs
# -------------------------------------
_name = "mass"
qoi_indices = {
  "mass": 0,
  "temp": 1
}
callable_lays = (_name,)

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
  "layer_norm": False,
  "trainable": True
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
        "filename": "./../../../physics/specie_thermos.json",
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
      "columns": [95, 96, 97, 98],
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
      "branch": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [4, 64, 32, 12],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_branch",
          **_common
        }
      },
      "trunk": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [1, 32, 32, 12],
          "activation": ["tanh"]*2 + ["linear"],
          "interpolating": {
            "file": f"./../../datagen/decomp/tensors/{_name}.txt",
            "columns": {"x": [0], "y": np.arange(1,13).tolist()},
            "axis": 1,
            "order": 3
          },
          "trainable": False,
          "name": _name + "_trunk"
        }
      }
    }
  }

}
