import tensorflow as tf


# Data properties
# =====================================
data_type = 'ODE'

physics_informed = True

# Data source
# -------------------------------------
from_file = False

data_src = {
  'test': './data/test/'
}

columns = {
  'x': [0],
  'y': [1,2]
}
columns = tf.nest.map_structure(lambda _: columns, data_src)

# Batching
# -------------------------------------
batch_size = {
  'train': {
    'ics': 4,
    'pts': 32,
    'res': 32
  },
  'valid': {
    'ics': 4,
    'pts': 32,
    'res': 32
  }
}
