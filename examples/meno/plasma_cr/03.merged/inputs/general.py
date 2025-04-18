# Model/architecture properties
# =====================================
model_name = "noneq"

dim = {
  "inp": [1, 45, 1, 1, 2],
  "out": [1, 45, 1]
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
  "./../../01.mass/saving/noneq_mass/model_weights.h5",
  "./../../01.temp/saving/noneq_temp/model_weights.h5",
  "./../../02.dist/saving/noneq_dist/model_weights.h5"
]

run_eagerly = False
