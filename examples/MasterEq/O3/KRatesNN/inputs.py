import os
import prode as pro


class InputData(object):

    ###########################################################################
    ### Paths
    ###########################################################################
    database_path = os.path.normpath('/home/zanardi/WORKSPACE/ML/PrODE/run/MasterEq/O3/database')
    utils_path = os.path.normpath('./utils')


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
    name  = 'KRatesNN'

    # Input Space Extreme Values ----------------------------------------------
    T = [1500.0e0,2500.0e0,5000.0e0,6000.0e0,8000.0e0,10000.0e0,12000.0e0,\
         14000.0e0,15000.0e0,20000.0e0]

    # Other properties --------------------------------------------------------
    grouping  = 'CBM3' #'CBM3' #'DPM10'

    ###########################################################################
    ### NN Architecture
    ###########################################################################
    nn_name = 'FFNN'

    # Variables Dimensionality ------------------------------------------------
    dim = {
            "inp": 1,
            "out": 6
            }

    # Structure ---------------------------------------------------------------
    layer_size         = [dim['inp'], 16, 16, dim['inp']]
    activation         = 'sigmoid'
    regularization     = {
                            'name': 'l1+l2',
                            'l1':   1.e-2,
                            'l2':   1.e-2,
                            }
    batch_norm         = None
    dropout_rate       = 0
    # NB: Not use `dropout_rate` and `batch_norm` together >> too much regularization
    kernel_initializer = "GlorotNormal"
    block = 'FFBlock' # 'FFBlock' ResPIBlock


    ###########################################################################
    ### Data
    ###########################################################################
    data_name = 'DataSet'

    # Testing -----------------------------------------------------------------
    period         = 1000            # Plotting frequency during training [epochs]


    ###########################################################################
    ### Training
    ###########################################################################
    epochs       = 10
    optimizer    = 'adam'
    lr           = 5.e-4

    # Learning rate scheduler -------------------------------------------------
    decay        = ["exponential", 2000, 0.98]

    # Losses ------------------------------------------------------------------
    loss    = {'identifier': 'mape', 'axis': 0} # `axis` = reduction axis
    metrics = ['mape']

    # Early Stopping Callback -------------------------------------------------
    min_delta    = 1.e-6
    patience     = 10000
    monitor      = 'val_tot_loss'


    ###########################################################################
    ### Plotting Quantities
    ###########################################################################
    # # Variables Name ----------------------------------------------------------
    # var_names = [ r'$O_2^{({%s})}$' % (i+1) for i in range(dim['out']-1) ]
    # var_names.append(r'$O$')

    # # Axis --------------------------------------------------------------------
    # labels   = [r'$t\quad[s]$', r'$Y$']         # x, y labels
    # scales   = ['log', 'linear']                # x, y scales
    # log_plot = True
    # limit_y  = False
