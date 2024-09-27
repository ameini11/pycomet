# Data generator inputs
# =====================================
# Training/Validation data
train_pts = 320
scenarios = 1000
validation_split = 0.2

# Testing data
test_cases = 100

# Time space
time_pts = 1000
time_lim = [1e-14, 1e-7]
time_eps = 1e-15
time_eval = None #[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

# Input space
inp_space = {
  "n":    [ 1e18,  1e20],
  "Ar":   [ 1e-1,  9e-1],
  "Ar4s": [ 1e-1,  9e-1],
  "Ar4p": [ 1e-1,  9e-1],
  "Ar+":  [1e-20, 1e-20],
  "e-":   [ 1e-3,  1e-1]
}
read_init_sols = False

# Database
energies_file = "./datagen/database/energies.json"
reactions_file = "./datagen/database/reactions.json"

# Multi-threading
nb_workers = 32
