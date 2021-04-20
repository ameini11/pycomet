import os

class InputData(object):

        database_path = os.path.normpath(os.getcwd()+'/../database')
        utils_path = os.path.normpath(os.getcwd()+'/../utils')

        ################################################################################################################################################################
        # ### Execution Flags
        # LoadIntFlg         = 0                                             # Loading Parameters             0=>No, 1=>Yes
        # TrainIntFlg        = 1                                             # Training                       0=>No, 1=>Yes
        # TestIntFlg         = 2                                             # Evaluating                     0=>Never, 1=>After Training, 2=>Also During Training
        # PlotIntFlg         = 1                                             # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training
        # VideoFlg           = 0                                             # Video History of Data          0=>No, 1=>yes
        # WriteDataIntFlg    = 2                                             # Writing Data After Training    0=>Never, 1=>After Training, 2=>Also During Training
        # WriteParamsIntFlg  = 1                                             # Writing Parameters             0=>Never, 1=>After Training, 2=>Also During Training
        # NEpochTest         = 1000

        ################################################################################################################################################################
        ### ODE Properties
        G_dim = 4                                  # Data dimensionality
        y_dim = 1                                  # Data dimensionality
        u_dim = 4                                  # Data dimensionality

        var_names      = []
        for i in range(G_dim-1):
            var_names.append(r'$\rho_{}$'.format(i+1))
        var_names.append(r'$\rho_O$')                                         # Names for the State Variables

        # Input Space
        Ttr    = [10000.0e0]                                               # Extreme Values for Tanslational Temperature
        # T_tr   = [1500.0e0,2500.0e0,5000.0e0,6000.0e0,8000.0e0,10000.0e0,12000.0e0,14000.0e0,15000.0e0,20000.0e0]  # Extreme Values for Tanslational Temperature
        T0     = [300.0e0,    10000.e0]                                     # Extreme Values for Initial TInt
        P0     = [1.013250e3, 1.013250e5]                                  # Extreme Values for Initial P
        XO2   = [0.950e0, 0.990e0]                                     # Extreme Values for Initial O2 Mole Frac

        # Time Domain
        t0     = 1.e-10                                                    # Minimum Time for Selecting Training Collocation Points
        tEnd   = 1.e-4                                                     # Maximum Time for Selecting Training Collocation Points

        grouping       = 'CBM3' #'DPM10'
        MEq_mat    = {'excit': True, 'diss': True, 'recomb': True}                                          # Defined in 'generate_data' function


        ###############################################################################################################################################################
        # ## NN Model Structure
        # ApproxModel     = 'DeepONet'
        # BranchLayers    = [uDim, 32, 32, 32] * dim
        # TrunkLayers     = [1, 64, 64, 32]
        # ScaleActFunsFlg = False
        # NormLayerFlg    = False
        # BiasBranchFlg   = True

        ################################################################################################################################################################

        ### Data set up

        train_cases = 3                    # Number of training cases
        train_pts = 64                     # Number of total training pts for each case
        valid = 0.2 # Number of validation v for each case
        train_pts_dist = 'uniform'                 # Training pts distribution

        test_cases = 2                   # Number of testing cases
        test_training = True                 # Test training case [Bool]


        # NTrainCases = 80                               # Number of Training Cases

        # NCollPts    = 0                                # Number of Training Collocation Pts

        # NTrainPts   = 128                              # Number of Training Anchor Pts
        # random_dist = False                            # Random distribution for Anchor Pts

        # RandDataFlg = True                             # Shuffle Training Data
        
        ### Data Transformation
        # scale_fac   = 1.e7
        # var_name    = ['u','y','G_uy']
        # var_func    = ['dir_trans','inv_trans']
        # transform   = {
        #                         'u':    {
        #                                     'dir_trans': tf.math.log,
        #                                     'inv_trans': tf.math.exp
        #                                     },
        #                         'y':    {
        #                                     'dir_trans': lambda x: tf.math.log(x * scale_fac),
        #                                     'inv_trans': lambda x: tf.math.exp(x) / scale_fac
        #                                     },
        #                         'G_uy': {
        #                                     'dir_trans': tf.math.exp,
        #                                     'inv_trans': tf.math.log
        #                                     }
        #                         }

        # ################################################################################################################################################################
        # ### Training
        # NEpoch          = 800000                       # Number of Epoches
        # batch_size      = None                         # Number of Epoches

        # # Validation Split
        # data_test       = 0.25                         # Percentage of Train/Test data splitting
        
        # # Learning Rate
        # LearningRate    = 1.e-4                        # Initial Learning Rate
        # lr_scheduler    = 'performance'                # Initial Scheduler: [None, 'exponential', 'performance', 'piecewise']
        # # For 'piecewise' scheduler only
        # bound           = [int(0.5*NEpoch), int(0.75*NEpoch)]
        # lr_rel_value    = [1.0, 0.5, 0.2]

        # # Optimizer
        # Optimizer       = 'adam'                       # Optimizer
        # OptimizerParams = [0.95, 1.e-7]                # Optimizer Parameters

        # # Loss Weighting
        # loss_name       = ['anch_pts','residual','l1_reg', 'l2_reg']
        # loss_weights    = {
        #                             'anch_pts': 1.,         # Loss from Anchor Pts
        #                             'residual': 1.,         # Loss from Residual
        #                             'l1_reg':   1.e-4,      # L1 regularization
        #                             'l2_reg':   1.e-4       # L2 regularization
        #                             }

        # ################################################################################################################################################################
        # ### Evaluating
        # NTestCases  = 3                                # Number of Test Cases
        # t0Test      = 0.0                              # Initial Time for Plotting Test Solutions
        # tEndTest    = tEnd * 1.e1                 # Final Time for Plotting Test Solutions

        # ################################################################################################################################################################
        # ### Plotting Quantities
        # xLogPlotFlg = True
        # yLogPlotFlg = True
        # if xLogPlotFlg:
        #     xMinValue  = t0
