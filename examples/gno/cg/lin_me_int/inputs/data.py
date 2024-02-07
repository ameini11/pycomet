# Data properties
# =====================================
data_type = "StaticGraphDataSet"

# Data source
# -------------------------------------
from_file = True

data_src = {
  "train": "./data/train.hdf5",
  "valid": "./data/valid.hdf5"
}

use_static_edges = False
use_conditioning = True

# Batching
# -------------------------------------
batch_size = {
  "train": 256,
  "valid": 256
}
