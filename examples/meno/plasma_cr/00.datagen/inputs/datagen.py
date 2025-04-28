# Data generator inputs
# =====================================
# Training/Validation data
train_pts = 240
validation_split = 0.2

# Time space
time_lim = [1e-9, 1e-5]
time_eps = 1e-10
time_eval = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

# Multi-threading
nb_workers = 64
nb_threads = 1

# Physics
load_plato = "module load plato/dev-gcc-12.4.0"
path_to_dtb = "./database/"
rom_config = "./tools/output/lowrank_state/vel1/rom.p"

# Samples
path_to_samples = "./tools/output/sampling/samples_i100/"
