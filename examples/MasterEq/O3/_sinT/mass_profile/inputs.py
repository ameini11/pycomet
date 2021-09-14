import os


class InputData(object):

    ###########################################################################
    ### Paths
    ###########################################################################
    database_path = os.path.normpath(os.getcwd()+'/../database')
    utils_path = os.path.normpath(os.getcwd()+'/../utils')


    ###########################################################################
    ### Execution Flags
    # ###########################################################################
    # load_flg  = 0       # Loading model     0: no, 1: saved model, 2: ckpt
    # train_flg = 1       # Training          0: no, 1: yes
    # test_flg  = 1       # Evaluating        0: no, 1: yes
    # plot_flg  = 2       # Plotting          0: no, 1: after training, 2: also during training
    # # video_flg = 0       # Video history     0: no, 1: yes
    # save_flg  = 1       # Saving model      0: no, 1: yes


    ###########################################################################
    ### ODE Properties
    ###########################################################################
    name  = 'MEq_OpNN'

    # Input Space Extreme Values ----------------------------------------------ß
    inp_dim = 4                         # Input Space Dimensionality
    Ttr = [10000.0e0,  10000.0e0]       # Tanslational Temperature
    # T_tr = [1500.0e0,2500.0e0,5000.0e0,6000.0e0,8000.0e0,10000.0e0,12000.0e0,14000.0e0,15000.0e0,20000.0e0]
    T0  = [300.0e0,    10000.0e0]       # Initial TInt
    P0  = [1.013250e2, 1.013250e4]      # Initial P
    XO2 = [0.05e0,      1.e0]            # Initial O2 Mole Frac

    # Other properties --------------------------------------------------------
    grouping  = 'CBM3' #'DPM10'
    dim    = 3
    MEq_mat   = {'excit': 1, 'diss': 1, 'recomb': 1}    # Processes Activator
    t0, t_end = 1.e-12, 1.e-3           # Time limits for testing

    ###########################################################################
    ### Data
    ###########################################################################
    # Training ----------------------------------------------------------------
    train_pts = {
                'ics': 5,     # Number of training scenarios
                'pts': 0,       # Number of total anchor pts for each scenario
                'res': 160      # Number of total collocaion pts for each scenario
            }
    valid = 0.2                 # % of Validation pts
    train_pts_dist = 'uniform'  # Training pts distribution

    # Testing -----------------------------------------------------------------
    test_cases     = 0              # Number of testing cases
    test_training  = False           # Test training case


    ###########################################################################
    ### Plotting Quantities
    ###########################################################################
    # Variables Name ----------------------------------------------------------
    var_names = [ r'$O_2^{({%s})}$' % (i+1) for i in range(dim) ]

    # Axis --------------------------------------------------------------------
    labels   = [r'$t\quad[s]$', r'$Y$']     # x, y labels
    scales   = ['log', 'log']               # x, y scales
