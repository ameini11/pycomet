# Data generator inputs
# =====================================
# Training/Validation data
train_pts = 240
validation_split = 0.2

# Time space
time_lim = [1e-8, 1e-6]
time_eps = 1e-10
time_eval = [1e-8, 1e-7, 1e-6]

# Multi-threading
nb_workers = 10
nb_threads = 1

# Physics
load_plato = "module load plato/dev-gcc-11.3.0-openblas-0.3.20"
path_to_dtb = "./database/"
# > ROM
rom_config = {"z_dim": 9}
for k in ("phi", "psi", "xref", "xscale", "mask"):
  rom_config[k] = path_to_dtb + f"/rom_data/full/{k}.txt"

# Samples
path_to_samples = "/home/zanardi/Codes/ML/PyCOMET/run/plasma_cr_rom/00.datagen/tools/init_sols"
# path_to_samples = "./tools/output/sampling/samples_shock/"
