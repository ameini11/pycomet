import numpy as np
import tensorflow as tf

from inputs.data import data_src

# Architecture properties
# =====================================
architecture = "LinNonlinME"

# Extra inputs
# -------------------------------------
callable_lays = {
  "mass": True,
  "temp": True,
  "dist": True
}
# Paths
_paths = {
  "data": "./../../datagen/data",
  "physics": "./../../../physics"
}

# Layers
# -------------------------------------
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
        "filename": _paths["physics"] + "/specie_thermos.json",
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
      "trainable": False
    }
  },

  "fcorr_compute": {
    "lay_id": "Correction",
    "kwargs": {
      "layer_size": [1, 32, 16, 4],
      "activation": ["tanh"]*2 + ["linear"],
      "state_prop": _paths["data"] + "/physics/RVE45/states/energies_norm.txt",
      "trainable": False,
      **_common
    }
  },

  "dist_update": {
    "lay_id": "SolUpdate",
    "kwargs": {
      "rates": _paths["data"] + "/physics/RVE45/rates.hdf5",
      "use_corr": True,
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

for _name in ("mass", "temp"):
  lays_propty[_name] = {
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
          'layer_size': [4, 24, 24],
          'activation': ['tanh']*2 + ['linear'],
          'out_xform': None,
          'transformation': 'affine',
          'num_coords': 1,
          "name": _name + "_prenet",
          "trainable": True,
          **_common
        }
      },
      "branch": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [4, 64, 32, 8],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_branch",
          "trainable": True,
          **_common
        }
      },
      "trunk": {
        "lay_id": "FFBlock",
        "kwargs": {
          "layer_size": [1, 64, 32, 8],
          "activation": ["tanh"]*2 + ["linear"],
          "name": _name + "_trunk",
          "trainable": True,
          **_common
        }
      },
      "output": {
        "lay_id": "Transform",
        "kwargs": {
          "xform": tf.math.exp if (_name == "temp") else tf.math.sigmoid,
          "name": _name + "_xform",
          "trainable": False
        }
      }
    }
  }
