import json

# Data properties
# =====================================
data_type = "DataSet"

# Data source
# -------------------------------------
from_file = True

_path_to_data = "./../../../datagen/data/"
data_src = {
  "train": _path_to_data + "/train.csv",
  "valid": _path_to_data + "/valid.csv",
  "test": _path_to_data + "/test/"
}

with open(_path_to_data + "/indices.json") as file:
  _indices = json.load(file)
_columns = {
  "x": [
    _indices["na_0"],
    _indices["nmi_0"],
    _indices["T_0"],
    _indices["t"],
    _indices["qoi_eq"]
  ],
  "y": [
    _indices["na"] + _indices["nmi"],
    _indices["T"]
  ]
}
columns = {k: _columns for k in data_src.keys()}

header = 0

# Batching
# -------------------------------------
batch_size = {
  "train": 256*10,
  "valid": 256*10
}
