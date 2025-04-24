from inputs.postproc import figures

# Training
# =====================================
epochs = int(2e4)
optimizer = "adam"

# Learning rate scheduler
# -------------------------------------
lr = 1e-3
scheduler = {
  "name": "exponential",
  "decay_steps": 75*50,
  "decay_rate": 0.98
}

# Loss and metrics
# =====================================
loss_info = {
  "loss": "mae",
  "axis": 0,
  "reduction": "auto"
}

# Callbacks
# =====================================
callbacks = {

  "early_stopping": {
    "monitor":              "val_loss",
    "min_delta":            1e-8,
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

  "tensorboard": {
    "histogram_freq": 0,
    "write_graph":    True,
    "write_images":   False,
    "profile_batch":  0
  },

  "plotter": {
    "figures": figures,
    "freq":    250
  }

}
