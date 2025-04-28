from inputs.data import _nlin_vars

# Model/architecture properties
# =====================================
model_name = "plasma_cr"

dim = {
  "inp": [34, 1, 1, 1],
  "out": [34, 1, 1]
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
  f"./../../01.{k}/run01/saving/plasma_cr_{k}/model_weights.h5" for k in _nlin_vars
] + ["./../../02.states/run01/saving/plasma_cr_states/model_weights.h5"]

run_eagerly = False
