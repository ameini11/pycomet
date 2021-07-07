import os
import prode as pro
import tensorflow as tf

from tensorflow.keras import layers
from prode.utilities.model import layers as pro_layers


###########################################################################
### Paths
###########################################################################
database_path = os.path.normpath(os.getcwd()+'/../../database')
utils_path = os.path.normpath(os.getcwd()+'/../utils')


###########################################################################
### Execution Flags
###########################################################################
load_flg  = 0       # Loading model     0: no, 1: saved model, 2: ckpt
train_flg = 1       # Training          0: no, 1: yes
test_flg  = 1       # Evaluating        0: no, 1: yes
plot_flg  = 2       # Plotting          0: no, 1: after training, 2: also during training
# video_flg = 0       # Video history     0: no, 1: yes
save_flg  = 1       # Saving model      0: no, 1: yes


###########################################################################
### ODE Properties
###########################################################################
name  = 'MEq_OpNN'

# Input Space Extreme Values ----------------------------------------------
inp_dim = 4                         # Input Space Dimensionality
Ttr = [5000.0e0,    10000.0e0]      # Tanslational Temperature
T0  = [1000.0e0,    10000.0e0]      # Initial TInt
P0  = [1.013250e2,  1.013250e4]     # Initial P
XO2 = [0.05e0,      1.e0]           # Initial O2 Mole Frac

# Other properties --------------------------------------------------------
grouping  = 'CBM3' #'DPM10'
MEq_mat   = {'excit': 1, 'diss': 1, 'recomb': 1}    # Processes Activator
t0, t_end = 1.e-14, 1.e-3           # Time limits for testing


###########################################################################
### NN Architecture
###########################################################################
nn_name = 'OpNN'

# Variables Dimensionality ------------------------------------------------
dim = {
    "inp": [6, 1],      # Branch-Trunk input dimensionality
    "out": 4            # Output dimensionality
}

# Structure ---------------------------------------------------------------
layer_size = {
    "branch": [dim['inp'][0], 128, 128, 64],
    "trunk":  [dim['inp'][1], 256, 256, 64]
}
activation = {
    "branch": 'elu',
    "trunk":  'sigmoid'
}
regularization = {
    'name':        'l1+l2', # 'name':       'fine_tuning',
    'l1':          1.e-4,
    'l2':          1.e-4,
    'weights_SP': './saving/MEq_OpNN_weights.h5'    # Path to for `FineTuningRegularizer`
}
batch_norm = {
    "branch": None,
    "trunk":  None
}
dropout_rate = 0
# NB: Not use `dropout_rate` and `batch_norm` together >> too much regularization
kernel_initializer = "GlorotNormal"
block = 'ResPIBlock' # FFBlock, ResPIBlock

# I/O Transformation ------------------------------------------------------
def branch_fn(x):
    T_scaled = (x[:,0:1] - Ttr[0]) / (Ttr[1] - Ttr[0])
    return tf.concat([T_scaled, x[:,1:]], 1)

input_fn = {
    "branch": layers.Lambda(branch_fn, name='branch_fn'),
    "trunk" : pro_layers.Scaling(init_scale=1.e7, name='trunk_fn', trainable=False)
}
output_fn = layers.Softmax(axis=-1, name='output_fn')


###########################################################################
### Plotting Quantities
###########################################################################
# Variables Name ----------------------------------------------------------
var_names = [ r'$O_2^{({%s})}$' % (i+1) for i in range(dim['out']-1) ]
var_names.append(r'$O$')

# Axis --------------------------------------------------------------------
labels   = [r'$t\quad[s]$', r'$Y$']         # x, y labels
scales   = ['log', 'linear']                # x, y scales
log_plot = True
limit_y  = False


###########################################################################
### Data
###########################################################################
data_name = 'ODE'

# Data source -------------------------------------------------------------
from_file = False
data_train_file = {
    'ics': './data/train/ics.csv',
    'pts': './data/train/pts.csv',
    'res': './data/train/res.csv'
}
data_valid_file = {
    'ics': './data/valid/ics.csv',
    'pts': './data/valid/pts.csv',
    'res': './data/valid/res.csv'
}
col_x = [[0,6],[6,7]]
col_y = [7,11]

# Training ----------------------------------------------------------------
train_pts = {
    'ics': 30,     # Number of training scenarios
    'pts': 1,       # Number of total anchor pts for each scenario
    'res': 160     # Number of total collocaion pts for each scenario
}
valid = 0.2                 # % of Validation pts
train_pts_dist = 'uniform'  # Training pts distribution

# Testing -----------------------------------------------------------------
test_cases     = 20              # Number of testing cases
test_training  = True           # Test training case


###########################################################################
### Training
###########################################################################
epochs       = 30000
batch_size   = {'ics': 128, 'pts': 128, 'res': 512}
valid_batch_size = {'ics': 128, 'pts': 128, 'res': 128}

optimizer    = 'adam'
lr           = 5.e-4

# Learning rate scheduler -------------------------------------------------
# decay_epochs = 1000
decay_steps  = 512*300 # round(train_pts*train_cases*(1.-valid))/batch_size*decay_epochs
decay        = ["exponential", decay_steps, 0.98]

# Losses ------------------------------------------------------------------
loss = {
    'ics': {'identifier': 'malpe', 'axis': 0}, # `axis` = reduction axis
    'pts': {'identifier': 'malpe', 'axis': 0},
    'res': {'identifier': 'mae',   'axis': 0}
}
metrics = None

# Callbacks ---------------------------------------------------------------
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
        'var_names': var_names,
        'labels':    labels,
        'scales':    scales,
        'log_flg':   log_plot,
        'limit_y':   limit_y,
        'period':    200,
    },

    'tensorboard': {
        'histogram_freq': 0,
        'write_graph':    True,
        'write_images':   True,
        'profile_batch':  0
    },

    'weighted_loss': {
        'name':  'EmpiricalMeanWeightsAdapter',
        'alpha': 0.9,
        'freq':  50
    }

}

