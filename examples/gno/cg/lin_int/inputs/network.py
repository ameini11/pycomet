# Architecture properties
# =====================================
architecture = "IntegralLinearMEGraphNet"

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

  "edge_encoder": {
    "lay_id": "FFBlock",
    "kwargs": {
      "layer_size": [2, 32, 32, 32, 1],
      "activation": ["tanh"]*3 + ["exp"],
      "layer_norm": False,
      "name": "edge_encoder",
      **_common
    }
  },

  "lin_me_gnet": {
    "lay_id": "LinearMEGraphNet",
    "kwargs": {
      "static_edge_index": "./data/edge_index.txt",
      "name": "lin_me_gnet"
    }
  }

}
