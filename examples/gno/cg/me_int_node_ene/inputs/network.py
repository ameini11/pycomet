import numpy as np

# Architecture properties
# =====================================
architecture = "MEIntNodeEnergyGNet"

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

lays_propty = {

  "node_energy": {
    "lay_id": "NodeEnergy",
    "kwargs": {
      "layer_size": [4, 32, 32, 1],
      "activation": ["tanh"]*2 + ["linear"],
      "static_node_attr": "./data/node_energies.txt",
      "cond_attr_indices": [0,1],
      "features_log_xform": {
        "base": "e",
        "slope":  np.array([  1e8, 1e-22, 1e19, 1e19]).reshape(1,1,-1),
        "offset": np.array([1e-13,     0,    0,    0]).reshape(1,1,-1),
        "scale":  np.array([  0.1,     1,    1,    1]).reshape(1,1,-1),
        "trainable": False
      },
      **_common
    }
  },

  "node_thermo": {
    "lay_id": "NodeThermoUpdate",
    "kwargs": {
      "node_attr_index_temp": {
        "int": 1,
        "chem": 1,
        "trans": 1
      },
      "node_attr_index_energy": 2,
      "node_attr_indices_nd_temp": [0,1],
      "specie_thermos": {
        "filename": "./inputs/sp_th.json",
        "thermo_process": 0,
        "eos_model": "PerfectGas"
      }
    }
  },

  "edge_fwd_corr": {
    "lay_id": "EdgeFwdRateCorr",
    "kwargs": {
      "edge_attr_index_rate": 0,
      "node_attr_index_temp": 1,
      "node_attr_index_energy": 2,
      "cond_attr_indices_scaling": [0,1],
      "static_fwd_edge_index": "./data/fwd_edge_index.txt"
    }
  },

  "edge_bwd": {
    "lay_id": "EdgeBackward",
    "kwargs": {
      "node_attr_index_pf": 3
    }
  },

  "lin_me_gnet": {
    "lay_id": "LinearMEGraphNet",
    "kwargs": {
      "node_attr_indices": [0]
    }
  },

  "residual": {
    "lay_id": "GatheredAdd",
    "kwargs": {
      "gathered": [
        {"indices": [0], "axis": -1},
        {"indices": [0], "axis": -1}
      ]
    }
  }

}
