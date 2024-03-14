# Data generator inputs
# =====================================
# Training/Validation data
scenarios = 400
train_pts = 160
validation_split = 0.2

# Testing data
test_cases = 20

# Time space
time_lim = [1e-12, 1e0]
time_eps = 1e-13

# Input space
inp_space = {
  "X_O": [0.05,  0.95],
  "p":   [ 1e3,   1e4],
  "Ti":  [ 1e3,   9e3],
  "T":   [ 1e4, 1.5e4]
}
read_init_sols = False

# Physics
path_to_physics = "./../physics/"
load_plato = "module load plato/dev-gcc-9.3.0-openmpi-3.1.3-sundials-5.8.0"

# SVD decomposition
# > Use only to generate data to perform SVD
save_sols_only = True
# > Path to SVD tensors
svd = None
# svd = {
#  # t*pO grid
#  "tp": "./decomp/tensors/tp.txt",
#  # Modes
#  "phi": "./decomp/tensors/phi.txt"
# }

# Multi-threading
nb_workers = 32
nb_threads = 4
