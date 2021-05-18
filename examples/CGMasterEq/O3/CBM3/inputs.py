import os
import tensorflow.keras.layers as lay

import prode as pro


class InputData(object):

    ###########################################################################
    ### Paths
    ###########################################################################
    database_path = os.path.normpath(os.getcwd()+'/../database')
    utils_path = os.path.normpath(os.getcwd()+'/../utils')


    ###########################################################################
    ### Execution Flags
    ###########################################################################
    load_flg  = 0       # Loading model     0: no, 1: saved model, 2: ckpt
    train_flg = 1       # Training          0: no, 1: yes
    test_flg  = 1       # Evaluating        0: no, 1: yes
    plot_flg  = 1       # Plotting          0: no, 1: after training, 2: also during training
    # video_flg = 0       # Video history     0: no, 1: yes
    save_flg  = 1       # Saving model      0: no, 1: yes


    ###########################################################################
    ### ODE Properties
    ###########################################################################
    name  = 'MEq_OpNN'

    # Input Space Extreme Values ----------------------------------------------
    inp_dim = 4                         # Input Space Dimensionality
    Ttr = [10000.0e0,  10000.0e0]       # Tanslational Temperature
    # T_tr = [1500.0e0,2500.0e0,5000.0e0,6000.0e0,8000.0e0,10000.0e0,12000.0e0,14000.0e0,15000.0e0,20000.0e0]
    T0  = [300.0e0,    300.e0]          # Initial TInt
    P0  = [1.013250e3, 1.013250e3]      # Initial P
    XO2 = [0.95e0,     0.95e0]          # Initial O2 Mole Frac

    # Other properties --------------------------------------------------------
    grouping  = 'CBM3' #'DPM10'
    MEq_mat   = {'excit': 1, 'diss': 1, 'recomb': 1}    # Processes Activator
    t0, t_end = 1.e-12, 1.e-3           # Time limits for testing


    ###########################################################################
    ### NN Architecture
    ###########################################################################
    nn_name = 'OpNN'

    # Variables Dimensionality ------------------------------------------------
    dim = {
            "u_dim": 5,     # Branch input dimensionality
            "y_dim": 1,     # Trunk input dimensionality
            "G_dim": 4      # Data dimensionality
            }

    # Structure ---------------------------------------------------------------
    layer_size         = {
                            "branch": [dim['u_dim'], 16, 16, 8],
                            "trunk":  [dim['y_dim'], 16, 16, 8]
                            }
    activation         = {
                            "branch": 'elu',
                            "trunk":  'sigmoid'
                            }
    regularization     = {
                            # 'name':       'l1+l2',
                            'name':       'fine_tuning',
                            'l1':         1.e-4,
                            'l2':         1.e-4,
                            'weights_SP': './saving/MEq_OpNN_weights.h5'    # Path to for `FineTuningRegularizer`
                            }
    batch_norm         = {
                            "branch": None,
                            "trunk": 'before'
                            }
    dropout_rate       = 0
    kernel_initializer = "GlorotNormal"
    # NB: Not use `dropout_rate` and `batch_norm` together >> too much regularization

    # I/O Transformation ------------------------------------------------------
    input_fn = {
                "branch": lay.Lambda(lambda x: x, name='branch_fn'),
                "trunk" : pro.layers.Scaling(init_scale=1.e7, name='trunk_fn', \
                                             trainable=False)
                }
    output_fn = lay.Softmax(axis=-1, name='output_fn')


    ###########################################################################
    ### Data
    ###########################################################################
    # Training ----------------------------------------------------------------
    train_cases    = 1            # Number of training cases
    train_pts      = 160            # Number of total training pts for each case
    valid          = 0.2            # % of Validation pts for each case
    train_pts_dist = 'uniform'      # Training pts distribution

    # Testing -----------------------------------------------------------------
    test_cases     = 1              # Number of testing cases
    test_training  = True           # Test training case
    period         = 1000            # Plotting frequency during training [epochs]


    ###########################################################################
    ### Training
    ###########################################################################
    epochs       = 100
    batch_size   = 128

    optimizer    = 'adam' 
    lr           = 5.e-4

    # Learning rate scheduler -------------------------------------------------
    decay_epochs = 5000
    decay_steps  = round(train_pts*train_cases*(1.-valid))/batch_size*decay_epochs
    decay        = ["exponential", decay_steps, 0.98]

    # Losses ------------------------------------------------------------------
    # Uppercase names are custom losses, while lowercase ones are tf.keras.losses
    loss         = {'pts': 'MALPE', 'res': 'mse'} # [ 'MALPE', 'MSLE', 'MSLPE' ] 
    metrics      = ['MALPE']

    # Early Stopping Callback -------------------------------------------------
    min_delta    = 1.e-6
    patience     = 20000
    monitor      = 'val_loss'


    ###########################################################################
    ### Plotting Quantities
    ###########################################################################
    # Variables Name ----------------------------------------------------------
    var_names = [ r'$O_2^{({%s})}$' % (i+1) for i in range(dim['G_dim']-1) ]
    var_names.append(r'$O$')

    # Axis --------------------------------------------------------------------
    labels   = [r'$t\quad[s]$', r'$Y$']         # x, y labels
    scales   = ['log', 'linear']                # x, y scales
    log_plot = True
