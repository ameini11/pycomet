# Model/architecture properties
# =====================================
model_name = "noneq"

dim = {
  "inp": [1, 45, 1, 1, 2],
  "out": [46, 1]
}

# Execution Flags
# =====================================
flags = {
  "load":  3,       # Loading model       0: no, 1: saved model, 2: ckpt
  "train": 0,       # Training model      0: no, 1: yes
  "test":  0,       # Testing model       0: no, 1: yes
  "plot":  0,       # Plotting            0: no, 1: offline, 2: online
  "save":  1        # Saving model        0: no, 1: yes
}

forced_loading = True
weights_to_load = [
  "./../../01.mass/case.r04/saving/noneq_mass/model_weights.h5",
  "./../../01.temp/case.r01/saving/noneq_temp/model_weights.h5"
]

run_eagerly = False
