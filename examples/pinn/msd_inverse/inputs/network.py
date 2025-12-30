# Architecture properties
# =====================================
architecture = "FFNN"

# Layers
# -------------------------------------
common = {
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

lays_propty = {

  "hidden_block": {
    "lay_id": "FFBlock",
    "kwargs": {
      "layer_size": [1, 32, 32, 2],
      "activation": ["tanh"]*2 + ["linear"],
      **common
    }
  }

}
