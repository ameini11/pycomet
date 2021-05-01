import os

class InputData(object):

    num_threads = 16

    ###########################################################################
    ### Paths
    ###########################################################################
    database_path = os.path.normpath(os.getcwd()+'/../database')
    utils_path = os.path.normpath(os.getcwd()+'/../utils')
    pro_path = os.path.normpath(os.environ['WORKSPACE_PATH'] + '/PrODE/prode/')

    ###########################################################################
    ### Execution Flags
    ###########################################################################
    load_flg  = 0       # Loading model     0: no, 1: saved model, 2: ckpt
    train_flg = 1       # Training          0: no, 1: Yes
    test_flg  = 1       # Evaluating        0: no, 1: after training, 2: also during training
    # video_flg = 0       # Video history     0: no, 1: yes
    save_flg  = 1       # Saving model      0: no, 1: yes


    ###########################################################################
    ### ODE Properties
    ###########################################################################
    name  = 'MEq_OpNN'

    # Variables Dimensionality ------------------------------------------------
    G_dim = 4       # Data dimensionality
    y_dim = 1       # Trunk input dimensionality
    u_dim = 4       # Branch input dimensionality

    # Input Space Extreme Values ----------------------------------------------
    Ttr = [10000.0e0]                   # Tanslational Temperature
    # T_tr = [1500.0e0,2500.0e0,5000.0e0,6000.0e0,8000.0e0,10000.0e0,12000.0e0,14000.0e0,15000.0e0,20000.0e0]
    T0  = [300.0e0,    300.e0]         # Initial TInt
    P0  = [1.013250e3, 1.013250e3]      # Initial P
    XO2 = [0.5e0,     0.999e0]         # Initial O2 Mole Frac

    # Other properties --------------------------------------------------------
    grouping  = 'CBM3' #'DPM10'
    MEq_mat   = {'excit': 1, 'diss': 1, 'recomb': 1}    # Processes Activator
    t0, t_end = 1.e-10, 1.e-4       # Time limits for testing
    mass_frac = True               # Output data in mass fractions [Bool]
    if mass_frac: u_dim += 1


    ###########################################################################
    ### NN Architecture
    ###########################################################################
    layer_size_branch  = [u_dim, 16, 16, 8]
    layer_size_trunk   = [y_dim, 16, 16, 8]
    activation         = {"branch": 'elu', "trunk": 'sigmoid'}
    kernel_initializer = "GlorotNormal"
    regularization     = ["l1+l2", 1.e-4, 1.e-4]
    dropout_rate       = 0
    batch_norm         = None

    # Input Transformation Quantities -----------------------------------------
    scale_fac = 1.e7


    ###########################################################################
    ### Data
    ###########################################################################
    # Training ----------------------------------------------------------------
    train_cases    = 10              # Number of training cases
    train_pts      = 230             # Number of total training pts for each case
    valid          = 0.2            # % of Validation pts for each case
    train_pts_dist = 'uniform'      # Training pts distribution

    # Testing -----------------------------------------------------------------
    test_cases     = 3              # Number of testing cases
    test_training  = True           # Test training case
    period         = 100            # Plotting frequancy during training [epochs]


    ###########################################################################
    ### Training
    ###########################################################################
    epochs      = 100000
    batch_size  = 230

    optimizer   = 'adam' 
    lr          = 1.e-4
    decay       = ["exponential", (epochs*train_cases/100), 0.98]

    loss        = {'pts': 'MALPE', 'res': None} # [ 'MSE', 'MALPE', 'MSLE', 'MSLPE' ] 
    metrics     = ["mape"]

    # Early Stopping Callback -------------------------------------------------
    min_delta   = 1.e-6
    patience    = 20000


    ###########################################################################
    ### Plotting Quantities
    ###########################################################################
    # Variables Name ----------------------------------------------------------
    var_names = [ r'$O_2^{({%s})}$' % (i+1) for i in range(G_dim-1) ]
    var_names.append(r'$O$')

    # Axis --------------------------------------------------------------------
    labels   = [r'$t\quad[s]$', r'$\rho\quad[kg/m^3]$']     # x, y labels
    if mass_frac: labels[1] = r'$Y$'
    scales   = ['log', 'linear']                            # x, y scales
    log_plot = True
