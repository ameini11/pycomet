from inputs.data import data_src, cols

# Architecture properties
# =====================================
architecture = "MultiPoleME"

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
_path_to_physics = "/home/zanardi/Codes/ML/PyCOMET/run/mp_meno/physics"

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

groups_graph = "./../datagen/data/groups_graph.json"

lays_propty = {

  "xform_fcorr": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/normalization_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": cols["x"][-1],
      "trainable": False
    }
  },

  "timescales": [
    {
      "dim": {
        "inp": [1, 1, 3, 60, 9],
        "out": [g]
      },
      "lays_propty": {
        "pf_update": {
          "lay_id": "PFUpdate",
          "kwargs": {
            "specie_thermos": {
              "filename": f"{_path_to_physics}/NRVE{g}/thermo/specie_thermos_ng.json",
              "config": None
            },
            "trainable": False
          }
        },
        "correct": {
          "lay_id": "Correction",
          "kwargs": {
            "layer_size": [1, 32, 16, 2],
            "activation": ["tanh"]*2 + ["linear"],
            "static_prop": f"./../datagen/data/physics/NRVE{g}/states/energies_norm.txt",
            **_common
          }
        },
        "sol_update": {
          "lay_id": "SolUpdate",
          "kwargs": {
            "rates": f"./../datagen/data/physics/NRVE{g}/rates.hdf5",
            "use_corr": False,
            "use_pinv": False,
            "trainable": False
          }
        }
      }
    } for g in (5, 20, 60)
  ]

}
