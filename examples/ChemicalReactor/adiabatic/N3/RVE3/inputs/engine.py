# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining model characteristics and training parameters.
# --------------------------------------------------------------------------- #


import inputs.postprocessing as inp_post


# Training
# =============================================================================
epochs = 100000
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 1.e-3
decay_steps = 50*1000
decay_rate = 0.98
decay = ["exponential", decay_steps, decay_rate]

# Loss & Metrics
# =============================================================================
axis = 0            # -1 0 None
reduction = 'auto'  # 'auto' 'sum_over_batch_size' 'sum'
loss = {
    'ics': 'malpe',
    'pts': 'malpe'
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
        **inp_post.plot_style,
        'freq': 1000
    },

    'tensorboard': {
        'histogram_freq': 0,
        'write_graph':    True,
        'write_images':   True,
        'profile_batch':  0
    }

}