# Model/architecture properties
# =====================================
model_name = "pollu"

dim = {
  "inp": [17, 3, 7, 1],
  "out": [17, 3]
}

# Execution Flags
# =====================================
flags = {
  "load":  0,       # Loading model       0: no, 1: saved model, 2: ckpt
  "train": 0,       # Training model      0: no, 1: yes
  "test":  0,       # Testing model       0: no, 1: yes
  "plot":  0,       # Plotting            0: no, 1: offline, 2: online
  "save":  0        # Saving model        0: no, 1: yes
}

run_eagerly = False
