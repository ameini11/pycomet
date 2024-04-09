from inputs.data import data_src, cols

# Architecture properties
# =====================================
architecture = "SinglePoleME"

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
_paths = {
  "data": "/home/zanardi/Codes/ML/PyCOMET/run/meno/adiabatic/datagen/data",
  "physics": "/home/zanardi/Codes/ML/PyCOMET/run/meno/physics/NRVE45"
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

  "pf_update": {
    "lay_id": "PFUpdate",
    "kwargs": {
      "specie_thermos": {
        "filename": _paths["physics"] + "/specie_thermos.json",
        "config": None
      },
      "trainable": False
    }
  },

  "fcorr_xform": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/normalization_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": cols["x"][-1],
      "trainable": False
    }
  },

  "fcorr_compute": {
    "lay_id": "Correction",
    "kwargs": {
      "layer_size": [1, 32, 16, 2],
      "activation": ["tanh"]*2 + ["linear"],
      "static_prop": _paths["data"] + "/physics/NRVE45/states/energies_norm.txt",
      **_common
    }
  },

  "operators": {
    "lay_id": "Operators",
    "kwargs": {
      "rates": _paths["data"] + "/physics/NRVE45/rates.hdf5",
      "use_corr": False
    }
  },

  "sol_update": {
    "lay_id": "SolUpdate",
    "kwargs": {
      "use_eig": False,
      "use_pinv": False,
      "expm_use_pade": True
    }
  }

}
