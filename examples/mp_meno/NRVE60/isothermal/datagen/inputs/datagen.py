# Data generator inputs
# =====================================
# Training/Validation data
scenarios = 1000
train_pts = 160
validation_split = 0.2

# Testing data
test_cases = 20

# Time space
time_lim = [1e-12, 1e-3]
time_eps = 1e-13

# Input space
inp_space = {
  "X_O": [   0,  0.95],
  "p":   [ 1e3,   1e4],
  "Ti":  [ 1e3,   8e3],
  "T":   [ 8e3, 1.5e4]
}
read_init_sols = False

# Physics
_path_to_physics = "/home/zanardi/Codes/ML/PyCOMET/run/mp_meno/physics"
path_to_physics = {
  f"NRVE{g}": f"{_path_to_physics}/NRVE{g}/" for g in (5, 20, 60)
}
ref_phy_model = "NRVE60"
load_plato = "module load plato/dev-gcc-9.3.0-openmpi-3.1.3-sundials-5.8.0"
groups_splits = [1,5,4,3]

# SVD decomposition
# > Use only to generate data to perform SVD
save_sols_only = False
# > Path to SVD tensors
# svd = None
svd = {
  k: {
    "tp": f"./decomp/tensors/{k}/tp.txt",
    "phi": f"./decomp/tensors/{k}/phi.txt"
  } for k in ("pa",)
  # } for k in ("na", "pa", "T")
}

# Multi-threading
nb_workers = 32
nb_threads = 4
