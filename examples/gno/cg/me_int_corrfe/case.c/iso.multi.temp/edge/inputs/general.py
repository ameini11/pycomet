# Model/architecture properties
# =====================================
model_name = "me_int"

dim = {
  "inp": [5, 3, 8],
  "out": [3]
}

# Execution Flags
# =====================================
flags = {
  'load':  0,       # Loading model       0: no, 1: saved model, 2: ckpt
  'train': 1,       # Training model      0: no, 1: yes
  'test':  1,       # Testing model       0: no, 1: yes
  'plot':  2,       # Plotting            0: no, 1: offline, 2: online
  'save':  1        # Saving model        0: no, 1: yes
}

run_eagerly = False
