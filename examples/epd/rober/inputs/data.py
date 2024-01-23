# Data properties
# =====================================
data_type = 'GraphDataSet'

# Data source
# -------------------------------------
from_file = False

data_src = {
  'train': './data/train/ics.csv',
  'valid': './data/valid/ics.csv',
  'test': './data/test/'
}

# Batching
# -------------------------------------
batch_size = {
  'train': 64,
  'valid': 64
}
