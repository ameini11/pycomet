# Architecture properties
# =====================================
architecture = 'EncodeProcessDecode'

# Layers
# -------------------------------------
latent_dim = 8

common = {
  'kernel_initializer': 'GlorotNormal',
  'regularizer': {
    'name': 'l1+l2',
    'l1': 1e-4,
    'l2': 1e-4
  },
  'dropout': {'rate': 1e-3},
  'layer_norm': True,
  'trainable': True
}

lays_propty = {

  'edge_encoder': {
    'lay_id': 'EdgeEncoder',
    'kwargs': {
      'layer_size': [2, 32, 32, latent_dim],
      'activation': ['sigmoid']*2 + ['linear'],
      "static_edge_index": "./data/edge_index.txt",
      "static_edge_attr": None,
      "name": 'edge_encoder',
      **common
    }
  },

  'node_encoder': {
    'lay_id': 'FFBlock',
    'kwargs': {
      'layer_size': [1, 32, 32, latent_dim],
      'activation': ['sigmoid']*2 + ['linear'],
      "name": 'node_encoder',
      **common
    }
  },

  'graphnet': [{
    'lay_id': 'GraphNet',
    'kwargs': {
      'layer_size': {
        "node": [latent_dim, 32, 32, latent_dim],
        "edge": [latent_dim, 32, 32, latent_dim]
      },
      'activation': {
        "node": ['tanh']*2 + ['linear'],
        "edge": ['tanh']*2 + ['linear']
      },
      "static_edge_index": "./data/edge_index.txt",
      "name": f'graphnet_{i+1}',
      **common
    }
  } for i in range(2)],

  'node_decoder': {
    'lay_id': 'FFBlock',
    'kwargs': {
      'layer_size': [latent_dim, 32, 32, 1],
      'activation': ['sigmoid']*2 + ['linear'],
      "name": 'node_decoder',
      **common
    }
  }

}