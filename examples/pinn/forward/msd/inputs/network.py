# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


# Architecture properties
# =============================================================================
architecture = 'FFNN'

# Layers ----------------------------------------------------------------------
common = {
    'kernel_initializer': 'GlorotNormal',
    'regularizer': {
        'name': 'l1+l2',
        'l1':   1.e-5,
        'l2':   1.e-5
    },
    'dropout_kwargs': {'rate': 0.},
    'batch_norm': None,
    'weight_norm': False,
    'trainable': True
}

lays_propty = {

    # 'hidden_block': {
    #     'block_type': 'FFBlock',
    #     'layer_size': [1, 16, 16, 16, 2],
    #     'activation': ['sigmoid']*3 + ['linear'],
    #     **common
    # }

    'hidden_block': {
        'lay_id': 'FFBlock',
        'kwargs': {
            'layer_size': [1, 16, 16, 16, 2],
            'activation': ['sigmoid']*3 + ['linear'],
            **common
        }
    }

}
