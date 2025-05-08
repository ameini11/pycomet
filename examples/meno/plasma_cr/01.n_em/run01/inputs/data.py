import json

# Data properties
# =====================================
data_type = "DataSet"

# Data source
# -------------------------------------
from_file = True

_path_to_data = "./../../00.datagen.r03/data.v3/"
data_src = {
  "train": _path_to_data + "/train.csv",
  "valid": _path_to_data + "/valid.csv",
  "test": _path_to_data + "/test/"
}

with open(_path_to_data + "/indices.json") as file:
  _indices = json.load(file)
_columns = {
  "x": [
    _indices["rho_0"],
    _indices["Th_0"],
    _indices["Te_0"],
    _indices["t"]
  ],
  "y": [
    _indices["n_em"]
  ]
}
columns = {k: _columns for k in data_src.keys()}
header = 0

# Database
_path_to_dtb = "./../../00.datagen.r03/database/"
_species = {k: _path_to_dtb+f"/species/{k}.json" for k in ("em", "Ar", "Arp")}

# Batching
# -------------------------------------
batch_size = {
  "train": 256*40,
  "valid": 256*40
}
