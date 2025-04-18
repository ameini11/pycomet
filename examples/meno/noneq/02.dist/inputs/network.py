import numpy as np
import tensorflow as tf

from inputs.data import data_src, _path_to_data

# Architecture properties
# =====================================
architecture = "MENONonEqReac"
# Variable name
_name = "dist"

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

  "fcorr_compute": {
    "lay_id": "StateCorrection",
    "kwargs": {
      "layer_size": [1, 32, 32, 8],
      "activation": ["tanh"]*2 + ["linear"],
      "state_prop": _path_to_data + "/physics/states/energies_norm.txt",
      "trainable": True,
      **_common
    }
  },

  "dist_update": {
    "lay_id": "NonEqSolUpdate",
    "kwargs": {
      "rates": _path_to_data + "/physics/rates.hdf5",
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
