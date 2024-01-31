import numpy as np


# Model/architecture properties
# =====================================
model_name = 'epcd'

dim = {
  "inp": (
    np.array([3,1]),          # Node attributes
    np.array([4,2]),          # Edge attributes
    np.array([1,1])           # Conditioning input
  ),
  "out": np.array([3,1])      # Node attributes
}

# Execution Flags
# =====================================
flags = {
  'load':  0,       # Loading model       0: no, 1: saved model, 2: ckpt
  'train': 1,       # Training model      0: no, 1: yes
  'test':  0,       # Testing model       0: no, 1: yes
  'plot':  0,       # Plotting            0: no, 1: offline, 2: online
  'save':  1        # Saving model        0: no, 1: yes
}

run_eagerly = False
