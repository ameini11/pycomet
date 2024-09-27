import numpy as np
import tensorflow as tf

from silx.io.dictdump import h5todict

# Data properties
# =====================================
data_type = "DataSet"

# Data source
# -------------------------------------
from_file = True
# Path to data
_path_to_data = "./../../00.datagen/data/"

data_src = {
  "train": _path_to_data + "/train.csv",
  "valid": _path_to_data + "/valid.csv",
  "test": _path_to_data + "/data/test/"
}

species_indices = h5todict(_path_to_data + "/physics/species_indices.hdf5")
_columns = {
  "x": [
    species_indices["lin"],       # Linear species at time t=0
    species_indices["nlin"],      # Nonlinear species at time t=0
    species_indices["inp"],       # Input species at time t=0
    np.array([20])                # Time
  ],
  "y": [
    species_indices["nlin"][0]+21 # Linear species at time t
  ]
}
_columns = tf.nest.map_structure(lambda x: x.reshape(-1).tolist(), _columns)
columns = {k: _columns for k in data_src.keys()}

header = 0

# Batching
# -------------------------------------
batch_size = {
  "train": 256*20,
  "valid": 256*20
}
