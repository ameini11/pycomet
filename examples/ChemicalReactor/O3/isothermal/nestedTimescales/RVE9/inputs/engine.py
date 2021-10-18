# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining model characteristics and training parameters.
# --------------------------------------------------------------------------- #


import inputs.network as inp_net
import inputs.postprocessing as inp_post


# Training
# =============================================================================
epochs = 50000
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 5.e-4
decay_steps = 512*600
decay_rate = 0.98
decay = ["exponential", decay_steps, decay_rate]

# Loss & Metrics
# =============================================================================
loss = {
    'ics': {
        name: {'identifier': 'malpe', 'axis': 0} for name in inp_net.names
    },
    'pts': {
        name: {'identifier': 'malpe', 'axis': 0} for name in inp_net.names
    },
    'res': {
        name: {'identifier': 'mae', 'axis': 0} for name in inp_net.names
    }
}

# Callbacks
# =============================================================================
callbacks = {

    'early_stopping': {
       'monitor':              'val_tot_loss',
       'min_delta':            1.e-8,
       'patience':             20000,
       'restore_best_weights': True,
       'verbose':              1
    },

    'lr_tracker': {
        'verbose': 1
    },

    'model_ckpt': {
        'monitor':           'val_tot_loss',
        'save_best_only':    True,
        'save_weights_only': True
    },

    'plotter': {
        **inp_post.plot_style,
        'limit_y': inp_post.limit_y,
        'freq':  500
    },

    'tensorboard': {
        'histogram_freq': 0,
        'write_graph':    True,
        'write_images':   True,
        'profile_batch':  0
    },

    'weighted_loss': {
        'name':         'EmpiricalWeightsAdapter',
        'alpha':        0.9,
        'freq':         50,
        'max_samples':  20000
    }

    # 'weighted_loss': {
    #     'name':         'NTKWeightsAdapter',
    #     'freq':         50,
    #     'max_samples':  1000
    # }

}
