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
  "load":  3,       # Loading model       0: no, 1: saved model, 2: ckpt
  "train": 1,       # Training model      0: no, 1: yes
  "test":  1,       # Testing model       0: no, 1: yes
  "plot":  2,       # Plotting            0: no, 1: offline, 2: online
  "save":  1        # Saving model        0: no, 1: yes
}

forced_loading = True
weights_to_load = [
  "./../../01.y1/saving/pollu_y1/model_weights.h5",
  "./../../01.y2/saving/pollu_y2/model_weights.h5",
  "./../../01.y6/saving/pollu_y6/model_weights.h5",
  "./../../02.y_lin/saving/pollu_y_lin/model_weights.h5"
]

run_eagerly = False
