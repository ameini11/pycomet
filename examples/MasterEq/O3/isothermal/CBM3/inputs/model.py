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


name = 'MEq_MMOpNN'

# Execution Flags
# =============================================================================
load_flg  = 0       # Loading model     0: no, 1: saved model, 2: ckpt
train_flg = 1       # Training          0: no, 1: yes
test_flg  = 1       # Evaluating        0: no, 1: yes
plot_flg  = 2       # Plotting          0: no, 1: offline, 2: online
save_flg  = 1       # Saving model      0: no, 1: yes


# Training
# =============================================================================
epochs = 100
optimizer = 'adam'

# Learning rate scheduler -----------------------------------------------------
lr = 5.e-4
decay_steps = 512*300 # round(train_pts*train_cases*(1.-valid))/batch_size*decay_epochs
decay = ["exponential", decay_steps, 0.98]


# Loss & Metrics
# =============================================================================
loss = {
    'ics': {
        inp_net.output_names[0]: {'identifier': 'malpe', 'axis': 0},
        inp_net.output_names[1]: {'identifier': 'malpe', 'axis': 0}
    },
    'pts': {
        inp_net.output_names[0]: {'identifier': 'malpe', 'axis': 0},
        inp_net.output_names[1]: {'identifier': 'malpe', 'axis': 0}
    },
    'res': {
        inp_net.output_names[0]: {'identifier': 'mae', 'axis': 0},
        inp_net.output_names[1]: {'identifier': 'mae', 'axis': 0}
    }
}
loss_weights = None
metrics = None

# Callbacks
# =============================================================================
callbacks = {

    'early_stopping': {
       'monitor':              'val_tot_loss',
       'min_delta':            1.e-8,
       'patience':             10000,
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
        'output_ops': inp_post.output_ops,
        'var_names':  inp_post.var_names,
        'labels':     inp_post.labels,
        'scales':     inp_post.scales,
        'limit_y':    inp_post.limit_y,
        'log_flg':    inp_post.log_plot,
        'period':     50
    },

    'tensorboard': {
        'histogram_freq': 0,
        'write_graph':    True,
        'write_images':   True,
        'profile_batch':  0
    },

    'weighted_loss': {
        'name':  'EmpiricalWeightsAdapter',
        'alpha': 0.9,
        'freq':  10,
        'method': 'mean'
    }

}
