# Architecture properties
# =====================================
architecture = "LinMEIntGraphNet"

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

  "node_temp": {
    "lay_id": "NodeTemperature",
    "kwargs": {
      "layer_size": [4, 32, 32, 1],
      "activation": ["tanh"]*2 + ["exp"],
      "temp": {
        "limits": [3e2, 1e5],
        "xform": "exp",
        "index_in": [1]
      },
      "cond_attr_indices": None,
      "static_node_attr": "./data/node_energies.txt",
      **_common
    }
  },

  "node_pf": {
    "lay_id": "NodePartitionFunctions",
    "kwargs": {
      "node_index_temp": {
        "int": 2,
        "chem": 1,
        "trans": 1
      },
      "specie_thermos": {
        "filename": "./inputs/sp_th.json",
        "thermo_process": 0,
        "eos_model": "PerfectGas"
      }
    }
  },

  "edge_bwd": {
    "lay_id": "EdgeBackward",
    "kwargs": {
      "node_index_pf": [3],
      "static_fwd_edge_index": "./data/fwd_edge_index.txt"
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
        {'indices': [0], 'axis': -1},
        {'indices': [0], 'axis': -1}
      ]
    }
  }

}
