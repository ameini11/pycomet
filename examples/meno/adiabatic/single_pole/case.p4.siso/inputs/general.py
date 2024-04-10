# Model/architecture properties
# =====================================
model_name = "sp_meno"

dim = {
  "inp": [1]*4 + [45, 9],
  "out": [45]
}

force_siso = True

# Execution Flags
# =====================================
flags = {
  'load':  2,       # Loading model       0: no, 1: saved model, 2: ckpt
  'train': 0,       # Training model      0: no, 1: yes
  'test':  0,       # Testing model       0: no, 1: yes
  'plot':  0,       # Plotting            0: no, 1: offline, 2: online
  'save':  1        # Saving model        0: no, 1: yes
}

run_eagerly = False
