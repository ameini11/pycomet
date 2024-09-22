# Model/architecture properties
# =====================================
model_name = "me_int"

dim = {
  "inp": [1, 45, 1, 1],
  "out": [1, 45, 1]
}

# Execution Flags
# =====================================
flags = {
  'load':  0,       # Loading model       0: no, 1: saved model, 2: ckpt
  'train': 0,       # Training model      0: no, 1: yes
  'test':  0,       # Testing model       0: no, 1: yes
  'plot':  0,       # Plotting            0: no, 1: offline, 2: online
  'save':  0        # Saving model        0: no, 1: yes
}

run_eagerly = False
