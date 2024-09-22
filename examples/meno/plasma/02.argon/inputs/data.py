# Data properties
# =====================================
data_type = "DataSet"

# Data source
# -------------------------------------
from_file = True

data_src = {
  "train": "./../../datagen/data/train.csv",
  "valid": "./../../datagen/data/valid.csv",
  "test": "./../../datagen/data/test/"
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
