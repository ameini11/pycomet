from inputs.data import data_src, cols

# Architecture properties
# =====================================
architecture = "MEIntCorrFE"

# Layers
# -------------------------------------
_processes = ("exc", "dis")
_path_to_physics = "./../datagen/data/physics/"

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
    "lay_id": "PFCorr",
    "kwargs": {
      "layer_size": [1+2, 64, 32, 1],
      "activation": ["tanh"]*2 + ["linear"],
      "static_prop": _path_to_physics + "/states/energies_norm.txt",
      **_common
    }
  },

  "fwd_euler": {
    "lay_id": "FwdEulerIsoME",
    "kwargs": {
      "processes": _processes,
      "fwd_rates": {
        k: _path_to_physics + f"/rates/{k}.txt" for k in ("kef", "kd")
      },
      "part_fn": {
        "atom": _path_to_physics + "/pf/O.txt",
        "molecule": _path_to_physics + "/pf/O2.txt"
      },
      "use_pfcorr": True,
      "use_fully_nonlin": False,
      "trainable": False
    }
  }

}
