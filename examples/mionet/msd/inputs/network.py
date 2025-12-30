# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


from inputs.general import dim


# Architecture properties
# =============================================================================
architecture = "MIONet"

# Layers ----------------------------------------------------------------------
common = {
  "kernel_initializer": "GlorotNormal",
  "regularizer": {
    "name": "l1+l2",
    "l1":   1.e-4,
    "l2":   1.e-4
  },
  "dropout": {"rate": 1e-3},
  "layer_norm": False,
  "trainable": True
}

lays_propty = {

  "input": {
    "params": {
      "index": 0,
      "xform": None
    },
    "ics": {
      "index": 1,
      "xform": None
    },
    "time": {
      "index": 2,
      "xform": None
    }
  },

  "prenet":  {

    "shifter": {
      "from": ["params", "ics"],
      "to": "time",
      "kwargs": {
        "layer_size": [sum(dim["inp"][:2]), 8, 8, 8],
        "activation": ["sigmoid"]*3 + ["linear"],
        "out_xform": None,
        "transformation": "similarity",
        "num_coords": 1,
        **common
      }
    }

  },

  "branch": {

    "params": [{
      "lay_id": "FFBlock",
      "kwargs": {
        "layer_size": [dim["inp"][0], 32, 32, 4],
        "activation": ["sigmoid"]*2 + ["linear"],
        **common
      }
    } for _ in range(2)],

    "ics": [{
      "lay_id": "FFBlock",
      "kwargs": {
        "layer_size": [dim["inp"][1], 32, 32, 4],
        "activation": ["sigmoid"]*2 + ["linear"],
        **common
      }
    } for _ in range(2)]

  },

  "trunk": {

    "time": {
      "lay_id": "FFBlock",
      "kwargs": {
        "layer_size": [dim["inp"][2], 32, 32, 4],
        "activation": ["sigmoid"]*2 + ["linear"],
        **common
      }
    }

  }

}
