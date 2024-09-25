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
  "x": [[ 0],list(range(1, 46)),[46],[47]],
  "y": [[48],list(range(49,94)),[94]]
}
columns = {k: _columns for k in data_src.keys()}

header = 0

# Batching
# -------------------------------------
batch_size = {
  "train": 256*10,
  "valid": 256*10
}
