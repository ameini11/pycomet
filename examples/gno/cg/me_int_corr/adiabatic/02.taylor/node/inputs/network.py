from inputs.data import data_src, cols

# Architecture properties
# =====================================
architecture = "MEIntCorrFE"

# Layers
# -------------------------------------
_common = {
  "kernel_initializer": "GlorotNormal",
  "regularizer": {
    "name": "l1+l2",
    "l1": 1e-3,
    "l2": 1e-3
  },
  "dropout": {"rate": 1e-2},
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

  "input": {
    "xform": {
      "lay_id": "StdNormalization",
      "kwargs": {
        "mean_var_file": "./saving/normalization_mean_var.txt",
        "data_files": [data_src[k] for k in ("train", "valid")],
        "columns": cols["x"][-1],
        "trainable": False
      }
    }
  },

  "corr_factor": {
    "lay_id": "PFCorr",
    "kwargs": {
      "layer_size": [8+2, 128, 128, 16],
      "activation": ["tanh"]*2 + ["linear"],
      "static_prop": "./../../datagen/data/physics/states/energies_norm.txt",
      **_common
    }
  },

  "update": {
    "lay_id": "FwdEulerME",
    "kwargs": {
      "rates": "./../../datagen/data/physics/rates.hdf5",
      "molecule": "O2",
      "specie_thermos": {
        "Ng": {
          "filename": "./../../physics/thermo/specie_thermos_ng.json",
          "config": None
        },
        "1g": {
          "filename": "./../../physics/thermo/specie_thermos_1g.json",
          "config": None
        }
      },
      "use_corr": True,
      "use_tril_corr": False,
      "use_state_corr": True,
      "use_state_corr_lin": True,
      "boltz_rescale_order": 0
    }
  }

}
