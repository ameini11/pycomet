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
    "lay_id": "FwdRateCorr",
    "kwargs": {
      "layer_size": {
        "exc": [9+4, 32, 16, 1],
        "dis": [9+2, 32, 16, 1]
      },
      "activation": {
        "exc": ["tanh"]*2 + ["linear"],
        "dis": ["tanh"]*2 + ["linear"]
      },
      "static_prop": "./../datagen/data/physics/states/energies_norm.txt",
      **_common
    }
  },

  "update": {
    "lay_id": "MatExpME",
    "kwargs": {
      "rates": "./../datagen/data/physics/rates.hdf5",
      "specie_thermos": {
        "filename": "./../physics/thermo/specie_thermos_ng.json",
        "config": None
      }
    }
  }

}
