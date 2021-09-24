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


import inputs.data as inp_data
import inputs.network as inp_net
import inputs.postprocessing as inp_post


# Architecture properties
# =============================================================================
name = 'MEq_OpNN'

# Execution Flags
# =============================================================================
flags = {
    'load':  0,       # Loading model       0: no, 1: saved model, 2: ckpt
    'train': 1,       # Training model      0: no, 1: yes
    'test':  1,       # Evaluating          0: no, 1: yes
    'plot':  2,       # Plotting            0: no, 1: offline, 2: online
    'save':  1        # Saving model        0: no, 1: yes
}
verbose = 0
visualize_graph = 1

# Training
# =============================================================================
epochs = 100#50000
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 5.e-4
decay_steps = 512*300
decay_rate = 0.98
decay = ["exponential", decay_steps, decay_rate]

# Loss & Metrics
# =============================================================================
loss = {
    'ics': {'identifier': 'malpe', 'axis': 0},
    'pts': {'identifier': 'malpe', 'axis': 0},
    'res': {'identifier': 'mae',   'axis': 0}
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
        'period':  500
    },

    'tensorboard': {
        'histogram_freq': 0,
        'write_graph':    True,
        'write_images':   True,
        'profile_batch':  0
    },

    'weighted_loss': {
        'name':     'EmpiricalWeightsAdapter',
        'alpha':    0.9,
        'freq':     50,
        'method':   'mean'
    }

}
