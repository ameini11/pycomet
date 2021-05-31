import os
import numpy as np
import tensorflow.keras.layers as lay

import prode as pro


class InputData(object):

    ###########################################################################
    ### Paths
    ###########################################################################
    utils_path = os.path.normpath(os.getcwd()+'/utils')


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
    name  = 'MSD_OpNN'

    # Input Space Extreme Values ----------------------------------------------
    pos = [4.0, 4.0]
    vel = [0.0, 0.0]

    # Other properties --------------------------------------------------------
    params = np.array([1.0, 0.5, 3.0], dtype=np.float64)
    t0, t_end = 0., 10.           # Time limits


    ###########################################################################
    ### NN Architecture
    ###########################################################################
    nn_name = 'OpNN'

    # Variables Dimensionality ------------------------------------------------
    dim = {
            "u_dim": 2,     # Branch input dimensionality
            "y_dim": 1,     # Trunk input dimensionality
            "G_dim": 2      # Data dimensionality
            }

    # Structure ---------------------------------------------------------------
    layer_size         = {
                            "branch": [dim['u_dim'], 16, 16, 8],
                            "trunk":  [dim['y_dim'], 16, 16, 8]
                            }
    activation         = {
                            "branch": 'tanh',
                            "trunk":  'tanh'
                            }
    regularization     = {
                            'name':       'l1+l2',
                            'l1':         1.e-4,
                            'l2':         1.e-4
                            }
    batch_norm         = {
                            "branch": None,
                            "trunk":  None
                            }
    dropout_rate       = 0
    # NB: Not use `dropout_rate` and `batch_norm` together >> too much regularization
    kernel_initializer = "GlorotNormal"

    # I/O Transformation ------------------------------------------------------
    input_fn  = None
    output_fn = None


    ###########################################################################
    ### Data
    ###########################################################################
    from_saved_data = False
    data_train_file = './data/train_data.csv'
    data_valid_file = './data/valid_data.csv'
    use_ohe         = True

    # Training ----------------------------------------------------------------
    train_cases    = 1            # Number of training cases
    train_pts      = 160            # Number of total training pts for each case
    valid          = 0.2            # % of Validation pts for each case
    train_pts_dist = 'uniform'      # Training pts distribution

    # Testing -----------------------------------------------------------------
    test_cases     = 1              # Number of testing cases
    test_training  = False           # Test training case
    period         = 1000            # Plotting frequency during training [epochs]


    ###########################################################################
    ### Training
    ###########################################################################
    epochs       = 100000
    batch_size   = 128

    optimizer    = 'adam'
    lr           = 5.e-3

    # Learning rate scheduler -------------------------------------------------
    decay_epochs = 2000
    decay_steps  = round(train_pts*train_cases*(1.-valid))/batch_size*decay_epochs
    decay        = ["exponential", decay_steps, 0.98]

    # Losses ------------------------------------------------------------------
    # Uppercase names are custom losses, while lowercase ones are tf.keras.losses
    loss         = {'pts': 'mse', 'res': 'MSE'} # [ 'MALPE', 'MSLE', 'MSLPE' ] 
    metrics      = ['MSE']

    # Early Stopping Callback -------------------------------------------------
    min_delta    = 1.e-6
    patience     = 100000
    monitor      = 'val_res_loss'


    ###########################################################################
    ### Plotting Quantities
    ###########################################################################
    # Variables Name ----------------------------------------------------------
    var_names = [ 'Position', 'Velocity' ]

    # Axis --------------------------------------------------------------------
    labels   = [r'$t\quad[s]$', r'$x$']         # x, y labels
    scales   = ['linear', 'linear']                # x, y scales
    log_plot = False
