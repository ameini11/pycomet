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

_columns = {
  "x": [list(range(4)),[4],[5],[6],[11]],
  "y": [list(range(7,11))]
}
columns = {k: _columns for k in data_src.keys()}

header = 0

# Batching
# -------------------------------------
batch_size = {
  "train": 256*20,
  "valid": 256*20
}
