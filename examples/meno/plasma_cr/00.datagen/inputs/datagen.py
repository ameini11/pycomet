# Data generator inputs
# =====================================
# Training/Validation data
train_pts = 160
validation_split = 0.2

# Time space
time_lim = [1e-9, 1e-5]
time_eps = 1e-12
time_eval = [1e-8, 1e-7, 1e-6]

# Multi-threading
nb_workers = 1
nb_threads = 2

# Physics
load_plato = "module load plato"
path_to_dtb = "./database"
rom_config = "./sampling/output/compress/rom.p"

# Samples
path_to_inp_space = "./sampling/output/samples"
