from inputs.postproc import figures

# Training
# =====================================
epochs = 10000
optimizer = "adam"

# Learning rate scheduler
# -------------------------------------
lr = 1e-3
scheduler = {
  "name": "exponential",
  "decay_steps": 100*50,
  "decay_rate": 0.97
}

# Loss and metrics
# =====================================
loss_info = {
  "loss": "mape",
  "axis": 0, # -1 0 None
  "reduction": "auto", # "auto" "sum_over_batch_size" "sum"
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

  "grads_tracker": {
    "max_samples":    None,
    "freq":           1,
    "histogram_freq": 1,
    "save":           True
  },

  'plotter': {
    'figures': figures,
    'freq':    250
  }

}
