# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining model characteristics and training parameters.
# --------------------------------------------------------------------------- #


from inputs.postproc import figures


# Training
# =============================================================================
epochs = 5000
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 5.e-4
decay_steps = 960*200
decay_rate = 0.98
decay = ["exponential", decay_steps, decay_rate]

# Loss & Metrics
# =============================================================================
axis = 0 # -1 0 None
reduction = 'auto' # 'auto' 'sum_over_batch_size' 'sum'
# loss = 'malpe'
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
       'min_delta':            1.e-6,
       'patience':             500,
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
    },

    'plotter': {
        'figures': figures,
        'freq':    500
    },

    # 'weighted_loss': {
    #     'name': 'ConstantWeightsAdapter',
    #     'multi_obj_loss_weights': {
    #         'ics': 64,
    #         'pts': 512,
    #         'res': 512
    #     }
    # }
    
}
