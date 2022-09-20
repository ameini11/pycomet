# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining model characteristics and training parameters.
# --------------------------------------------------------------------------- #


from inputs.postproc import figures


# Training
# =============================================================================
epochs = 100
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 5.e-3
decay_steps = 160*100
decay_rate = 0.98
decay = ["exponential", decay_steps, decay_rate]

# Loss & Metrics
# =============================================================================
axis = 0 # -1 0 None
reduction = 'auto' # 'auto' 'sum_over_batch_size' 'sum'
loss = {
    'ics': 'mse',
    'pts': 'mse',
    'res': 'mse'
}

# Callbacks
# =============================================================================
callbacks = {

    'early_stopping': {
       'monitor':              'val_loss',
       'min_delta':            1.e-8,
       'patience':             20000,
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

    'plotter': {
        'figures': figures,
        'freq': 50
    },

    'tensorboard': {
        'histogram_freq': 0,
        'write_graph':    True,
        'write_images':   True,
        'profile_batch':  0
    },

    # 'weighted_loss': {
    #     'name':         'InverseDirichletWeightsAdapter',
    #     'alpha':        0.9,
    #     'freq':         50,
    #     'max_samples':  20000
    # }

}
