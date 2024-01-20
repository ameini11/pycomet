# Architecture properties
# =====================================
architecture = 'FFNN'

# Layers
# -------------------------------------
common = {
  'kernel_initializer': 'GlorotNormal',
  'regularizer': {
    'name': 'l1+l2',
    'l1': 1e-4,
    'l2': 1e-4
  },
  'dropout_kwargs': {'rate': 1e-3},
  'batch_norm': None,
  'trainable': True
}

lays_propty = {

  'hidden_block': {
    'lay_id': 'FFBlock',
    'kwargs': {
      'layer_size': [1, 32, 32, 2],
      'activation': ['sigmoid']*2 + ['linear'],
      **common
    }
  }

}
