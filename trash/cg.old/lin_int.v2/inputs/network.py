# Architecture properties
# =====================================
architecture = "IntegralLinearMEGraphNetV2"

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

  "message": {
    "lay_id": "NonLinearMessage",
    "kwargs": {
      "layers": {
        "gather": {
          "lay_id": "GatherMessage",
          "kwargs": {
            "origin": "sender",
            "include_edge": True,
            "static_edge_index": "./data/edge_index.txt",
          }
        },
        # "xform": {
        #   "lay_id": "Log",
        #   "kwargs": {
        #     "base": 10,
        #     "slope": 1e7,
        #     "offset": 1e-13,
        #   }
        # },
        "nonlin": {
          "lay_id": "FFBlock",
          "kwargs": {
            "layer_size": [4, 32, 32, 1],
            "activation": ["tanh"]*2 + ["linear"],
            **_common
          }
        }
      }
    }
  }

}
