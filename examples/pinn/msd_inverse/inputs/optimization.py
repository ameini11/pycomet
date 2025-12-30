from inputs.postproc import figures


# Training
# =====================================
epochs = 1000
optimizer = "adam"

# Learning rate scheduler
# -------------------------------------
lr = 1.e-3
scheduler = {
  "name": "exponential",
  "decay_steps": 8*50,
  "decay_rate": 0.98
}

# Loss and metrics
# =====================================
loss_info = {
  "loss": {
    "ics": "mse",
    "pts": "mse",
    "res": "mse"
  },
  "axis": 0, # -1 0 None
  "reduction": "auto"
}

# Callbacks
# =====================================
callbacks = {

  "early_stopping": {
     "monitor":              "val_loss",
     "min_delta":            1.e-8,
     "patience":             20000,
     "restore_best_weights": True,
     "verbose":              1
  },

  "lr_tracker": {
    "verbose": 1
  },

  "model_ckpt": {
    "monitor":           "val_loss",
    "save_best_only":    True,
    "formats":           ["h5"]
  }, 

  "plotter": {
    "figures": figures,
    "freq":    50
  },

  "tensorboard": {
    "histogram_freq": 0,
    "write_graph":    True,
    "write_images":   True,
    "profile_batch":  0
  },

  "variable": {
    "var_names": ["C", "K"]
  },

  # "weighted_loss": {
  #     "name":         "EmpiricalWeightsAdapter",
  #     "alpha":        0.9,
  #     "freq":         10,
  #     "max_samples":  20000
  # }

}
