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
    pos = [-5.0, 5.0]
    vel = [-5.0, 5.0]

    # Other properties --------------------------------------------------------
    params = np.array([1.0, 0.5, 3.0], dtype=np.float64)
    t0, t_end = 0., 10.         # Time limits


    ###########################################################################
    ### NN Architecture
    ###########################################################################
    nn_name = 'OpNN'

    # Variables Dimensionality ------------------------------------------------
    dim = {
            "inp": [2, 1],     # Branch-Trunk input dimensionality
            "out": 2          # Output dimensionality
            }

    # Structure ---------------------------------------------------------------
    layer_size         = {
                            "branch": [dim['inp'][0], 32, 32, 16],
                            "trunk":  [dim['inp'][1], 32, 32, 16]
                            }
    activation         = {
                            "branch": 'elu',
                            "trunk":  'tanh'
                            }
    regularization     = {
                            'name':       None,#'l1+l2',
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
    block = 'FFBlock' #'ResPIBlock'

    # I/O Transformation ------------------------------------------------------
    input_fn  = None
    output_fn = None


    ###########################################################################
    ### Data
    ###########################################################################
    data_name = 'ODE'

    # Data source -------------------------------------------------------------
    from_file       = False
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
    col_x = [[0,2],[2,3]]
    col_y = [3,5]

    # Training ----------------------------------------------------------------
    train_pts = {
                'ics': 40,   # Number of training scenarios
                'pts': 10, # Number of total anchor pts for each scenario
                'res': 160  # Number of total collocaion pts for each scenario
            }
    valid   = 0.2   # % of Validation pts
    train_pts_dist = 'uniform'      # Training pts distribution

    # Testing -----------------------------------------------------------------
    test_cases     = 1              # Number of testing cases
    test_training  = False           # Test training case
    period         = 1000            # Plotting frequency during training [epochs]


    ###########################################################################
    ### Training
    ###########################################################################
    epochs       = 100000
    # batch_size   = 128
    batch_size   = {'ics': 16, 'pts': 32, 'res': 128}
    valid_batch_size = {'ics': 16, 'pts': 8, 'res': 32}

    optimizer    = 'adam'
    lr           = 1.e-4

    # Learning rate scheduler -------------------------------------------------
    decay_epochs = 2000
    decay_steps  = 10000 #round(train_pts*num_ics*(1.-valid))/batch_size*decay_epochs
    decay        = ["exponential", decay_steps, 0.98]

    # Losses ------------------------------------------------------------------
    # Uppercase names are custom losses, while lowercase ones are tf.keras.losses
    loss         = {'ics': 'mse', 'pts': 'mse', 'res': 'mse'} # [ 'MALPE', 'MSLE', 'MSLPE' ]
    metrics      = ['mse']
    adaptive_weights = {'name': 'ConstantAdapter'}

    # Early Stopping Callback -------------------------------------------------
    min_delta    = 1.e-6
    patience     = 100000
    monitor      = 'val_tot_loss'


    ###########################################################################
    ### Plotting Quantities
    ###########################################################################
    # Variables Name ----------------------------------------------------------
    var_names = [ 'Position', 'Velocity' ]

    # Axis --------------------------------------------------------------------
    labels   = [r'$t\quad[s]$', r'$x$']         # x, y labels
    scales   = ['linear', 'linear']                # x, y scales
    log_plot = False
