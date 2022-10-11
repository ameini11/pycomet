# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining model characteristics and training parameters.
# --------------------------------------------------------------------------- #


# Training
# =============================================================================
epochs = 2 #10000
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 1.e-4
decay_steps = 4*4413*10
decay_rate = 0.98
decay = ["exponential", decay_steps, decay_rate]

# Loss & Metrics
# =============================================================================
axis = 0 # -1 0 None
reduction = 'auto' # 'auto' 'sum_over_batch_size' 'sum'
loss = {
    'ics': 'malpe',
    'pts': 'malpe',
    'res': 'mae'
}

## Callbacks
# =============================================================================
callbacks = {

    'early_stopping': {
       'monitor':              'val_loss',
       'min_delta':            1.e-4,
       'patience':             300,
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
