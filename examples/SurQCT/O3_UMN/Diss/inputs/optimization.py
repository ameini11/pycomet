# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining model characteristics and training parameters.
# --------------------------------------------------------------------------- #


# Training
# =============================================================================
epochs = 100000
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 2.e-4
decay_steps = 12*500
decay_rate = 0.98
decay = ["exponential", decay_steps, decay_rate]

# Loss & Metrics
# =============================================================================
axis = 0
reduction = 'auto'
loss = 'mape'

## Callbacks
# =============================================================================
callbacks = {

    'early_stopping': {
       'monitor':              'val_loss',
       'min_delta':            1.e-4,
       'patience':             30000,
       'restore_best_weights': True,
       'verbose':              1
    },

    'lr_tracker': {
        'verbose': 1
    },

    'model_ckpt': {
        'monitor':           'val_loss',
        'save_best_only':    True,
        'save_weights_only': True
    }, 

    'tensorboard': {
        'histogram_freq': 0,
        'write_graph':    True,
        'write_images':   True,
        'profile_batch':  0
    }
    
}
