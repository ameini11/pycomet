# Model/architecture properties
# =====================================
model_name = "plasma"

dim = {
  "inp": [4, 1, 1, 1],
  "out": [4, 1]
}

# Execution Flags
# =====================================
flags = {
  "load":  3,       # Loading model       0: no, 1: saved model, 2: ckpt
  "train": 1,       # Training model      0: no, 1: yes
  "test":  1,       # Testing model       0: no, 1: yes
  "plot":  2,       # Plotting            0: no, 1: offline, 2: online
  "save":  1        # Saving model        0: no, 1: yes
}

forced_loading = True
weights_to_load = [
  "./../../01.electron/saving/plasma_electron/model_weights.h5",
  "./../../02.argon/saving/plasma_argon/model_weights.h5"
]

run_eagerly = False
