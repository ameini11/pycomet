# Data generator inputs
# =====================================
# Training/Validation data
scenarios = 200
train_pts = 160
validation_split = 0.2

# Testing data
test_cases = 10

# Time space
time_lim = [1e-12, 1e-3]
time_eps = 1e-13

# Input space
inp_space = {
  "X_O": [0.05, 0.95],
  "p":   [ 1e3,  1e4],
  "Ti":  [ 1e3,  9e3],
  "T":   [ 1e4,  1e4]
}

# Physics
path_to_physics = "./physics/"
processes = ("exc",)
free_atom = False
