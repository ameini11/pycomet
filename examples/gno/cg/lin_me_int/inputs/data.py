# Data properties
# =====================================
data_type = "StaticGraphDataSet"

# Data source
# -------------------------------------
from_file = False

data_src = {
  "train": "./data/train.hdf5",
  "valid": "./data/valid.hdf5"
}

use_static_edges = False

# Batching
# -------------------------------------
batch_size = {
  "train": 256,
  "valid": 256
}
