import numpy as np
import tensorflow as tf
from inputs.general import dim

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

cols = {
  "x": np.split(
    np.arange(np.sum(dim["inp"])),
    np.cumsum(dim["inp"])
  )[:-1],
  "y": np.split(
    np.sum(dim["inp"]) + np.arange(np.sum(dim["out"])),
    np.cumsum(dim["out"])
  )[:-1]
}
cols = tf.nest.map_structure(list, cols)
columns = tf.nest.map_structure(lambda _: cols, data_src)

header = 0

# Batching
# -------------------------------------
batch_size = {
  "train": 256*4,
  "valid": 256*4
}
