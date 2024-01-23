# Data properties
# =====================================
data_type = 'GraphDataSet'

# Data source
# -------------------------------------
from_file = True

data_src = {
  'train': './data/train.hdf5',
  'valid': './data/valid.hdf5',
  'test': None
}

use_static_edges = False

# Batching
# -------------------------------------
batch_size = {
  'train': 32,
  'valid': 32
}
