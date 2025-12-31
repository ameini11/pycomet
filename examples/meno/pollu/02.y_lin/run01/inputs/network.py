from inputs.data import data_src, _columns

# Architecture properties
# =====================================
architecture = "MENOPOLLUReac"
# Path to data
_path_to_data = "./../../00.datagen/data/"

# Extra inputs
# -------------------------------------
callable_lays = {
  "y_lin": True,
  "y_nlin": False
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
      "trainable": False
    }
  },

  "fcorr_compute": {
    "lay_id": "StateCorrection",
    "kwargs": {
      "layer_size": [1, 64, 32, 16],
      "activation": ["tanh"]*2 + ["linear"],
      "state_prop": _path_to_data + "/physics/species_feat.txt",
      "trainable": True,
      **_common
    }
  },

  "y_lin_update": {
    "lay_id": "POLLUSolUpdate",
    "kwargs": {
      "rates": _path_to_data + "/physics/rates.hdf5",
      "species_indices": _path_to_data + "/physics/species_indices.hdf5",
      "use_corr": True,
      "scale_corr": 1e0,
      "epsilon": 1e-12,
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
