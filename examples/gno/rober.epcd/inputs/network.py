# Architecture properties
# =====================================
architecture = "EncodeProcessConditionDecode"

# Layers
# -------------------------------------
_latent_dim = 8
_hidden_dim = 32
_gnet_depth = 1
_layer_norm = True

_common = {
  "kernel_initializer": "GlorotNormal",
  "regularizer": {
    "name": "l1+l2",
    "l1": 1e-4,
    "l2": 1e-4
  },
  "dropout": {"rate": 1e-4},
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
    "node": {
      "index": 0,
      "xform": None
    },
    "edge": {
      "index": 1,
      "xform": None
    },
    "cond": {
      "index": 2,
      "xform": {
        "lay_id": "Log",
        "kwargs": {
          "base": 10,
          "slope": 1,
          "offset": 1e-6,
          "trainable": False
        }
      }
    }
  },

  "prenet": {
    "from": "node",
    "to": "cond",
    "kwargs": {
      "layer_size": [1, 16, 16],
      "activation": ["tanh"]*2 + ["linear"],
      "layer_norm": False,
      "out_xform": None,
      "transformation": "similarity",
      "num_coords": 1,
      "name": "prenet",
      **_common
    }
  },

  "cond_encoder": {
    "lay_id": "FFBlock",
    "kwargs": {
      "layer_size": [1, _hidden_dim, _hidden_dim, _latent_dim],
      "activation": ["tanh"]*2 + ["linear"],
      "layer_norm": True,
      "name": "cond_encoder",
      **_common
    }
  },

  "edge_encoder": {
    "lay_id": "EdgeEncoder",
    "kwargs": {
      "layer_size": [2, _hidden_dim, _hidden_dim, _latent_dim],
      "activation": ["tanh"]*2 + ["linear"],
      "layer_norm": _layer_norm,
      "static_edge_index": "./data/edge_index.txt",
      "static_edge_attr": None,
      "name": "edge_encoder",
      **_common
    }
  },

  "node_encoder": {
    "lay_id": "FFBlock",
    "kwargs": {
      "layer_size": [1, _hidden_dim, _hidden_dim, _latent_dim],
      "activation": ["tanh"]*2 + ["linear"],
      "layer_norm": _layer_norm,
      "name": "node_encoder",
      **_common
    }
  },

  "graphnet": [{
    "lay_id": "GraphNet",
    "kwargs": {
      "layer_size": {
        "edge": [3*_latent_dim, _hidden_dim, _hidden_dim, _latent_dim],
        "node": [2*_latent_dim, _hidden_dim, _hidden_dim, _latent_dim]
      },
      "activation": {
        "node": ["tanh"]*2 + ["linear"],
        "edge": ["tanh"]*2 + ["linear"]
      },
      "layer_norm": _layer_norm,
      "static_edge_index": "./data/edge_index.txt",
      "reduction": "mean",
      "name": f"graphnet_{i+1}",
      **_common
    }
  } for i in range(_gnet_depth)],

  "node_decoder": {
    "lay_id": "FFBlock",
    "kwargs": {
      "layer_size": [_latent_dim, _hidden_dim, _hidden_dim, 1],
      "activation": ["tanh"]*2 + ["linear"],
      "layer_norm": False,
      "name": "node_decoder",
      **_common
    }
  },

  "residual": {
    "lay_id": "GatheredAdd",
    "kwargs": {
      "gathered": None,
      "name": "residual"
    }
  }

}
