import numpy as np
from inputs.datagen import processes

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
  "dropout": {"rate": 1e-4},
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

def _get_log_xform(n=2):
  return {
    "base": "e",
    "slope":  np.array([  1e5]+[1e19]*n).reshape(1,1,-1),
    "offset": np.array([1e-10]+[   0]*n).reshape(1,1,-1),
    "scale":  np.array([  0.1]+[   1]*n).reshape(1,1,-1),
    "trainable": False
  }

lays_propty = {

  "corr_factor": {
    "lay_id": "PFCorr",
    "kwargs": {
      "layer_size": [3, 32, 32, 1],
      "activation": ["tanh"]*2 + ["linear"],
      "static_prop": "./../data/states/energies.txt",
      "features_log_xform": _get_log_xform(n=2),
      **_common
    }
  },

  "fwd_euler": {
    "lay_id": "FwdEulerIsoME",
    "kwargs": {
      "processes": processes,
      "use_pfcorr": True,
      "use_fully_nonlin": False,
      "fwd_rates": {k: f"./../data/rates/{k}.txt" for k in ("kef", "kd")},
      "part_fn": {
        "atom": "./../data/pf/O.txt", 
        "molecule": "./../data/pf/O2.txt"
       }
    }
  }

}
