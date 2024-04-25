from inputs.data import data_src

# Architecture properties
# =====================================
architecture = "LinNonlinME"

# Extra inputs
# -------------------------------------
_name = "dist"
qoi_indices = {
  "mass": 0,
  "temp": 1
}
callable_lays = (_name,)
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
  "layer_norm": False,
  "trainable": True
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
      "columns": [95, 96, 97, 98],
      "trainable": False
    }
  },

  "fcorr_xform": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/std_normalization_1_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": [95, 96, 97, 98, 99],
      "trainable": False
    }
  },

  "fcorr_compute": {
    "lay_id": "Correction",
    "kwargs": {
      "layer_size": [1, 32, 16, 4],
      "activation": ["tanh"]*2 + ["linear"],
      "state_prop": _paths["data"] + "/physics/RVE45/states/energies_norm.txt",
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
      "inv_loss_weight": 1e-3
    }
  }

}
