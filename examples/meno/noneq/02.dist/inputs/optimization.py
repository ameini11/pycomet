import numpy as np

from inputs.postproc import figures
from scipy.constants import Avogadro as UNA

# Training
# =====================================
epochs = int(2e3)
optimizer = "adam"

# Learning rate scheduler
# -------------------------------------
lr = 1e-3
scheduler = {
  "name": "exponential",
  "decay_steps": 100*50,
  "decay_rate": 0.94
}

# Loss and metrics
# =====================================
loss_info = {
  "loss": "mape",
  "axis": 0,
  "reduction": "auto",
  "std_scaler": np.array({"mean": 0.0, "std": UNA})
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
