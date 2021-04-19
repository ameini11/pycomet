import os
import tensorflow as tf
import numpy as np

np.random.seed(0)

class input(object):


    def __init__(self, WORKSPACE_PATH, ProPDEFldr):

        ################################################################################################################################################################
        ### Case Name
        self.PDEName            = 'CGMasterEq'

        ################################################################################################################################################################
        ### Execution Flags
        self.LoadIntFlg         = 0                                             # Loading Parameters             0=>No, 1=>Yes
        self.TrainIntFlg        = 1                                             # Training                       0=>No, 1=>Yes
        self.TestIntFlg         = 2                                             # Evaluating                     0=>Never, 1=>After Training, 2=>Also During Training
        self.PlotIntFlg         = 1                                             # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training
        self.VideoFlg           = 0                                             # Video History of Data          0=>No, 1=>yes
        self.WriteDataIntFlg    = 2                                             # Writing Data After Training    0=>Never, 1=>After Training, 2=>Also During Training
        self.WriteParamsIntFlg  = 1                                             # Writing Parameters             0=>Never, 1=>After Training, 2=>Also During Training
        self.NEpochTest         = 1000

        ################################################################################################################################################################
        ### PDE Properties
        self.uDim        = 4                                                    # Dimensionality of Initial Condition Vectors
        self.xDim        = 4                                                    # Dimensionality of State Vectors

        self.xNames      = []
        for i in range(self.xDim-1):
            self.xNames.append(r'$\rho_{}$'.format(i+1))
        self.xNames.append(r'$\rho_O$')                                         # Names for the State Variables

        # Input Space
        self.Ttr    = [10000.0e0]                                               # Extreme Values for Tanslational Temperature
        # self.T_tr   = [1500.0e0,2500.0e0,5000.0e0,6000.0e0,8000.0e0,10000.0e0,12000.0e0,14000.0e0,15000.0e0,20000.0e0]  # Extreme Values for Tanslational Temperature
        self.T0     = [300.0e0,    1000.e0]                                     # Extreme Values for Initial TInt
        self.P0     = [1.013250e3, 1.013250e4]                                  # Extreme Values for Initial P
        self.X_O2   = [0.950e0,    0.999e0]                                     # Extreme Values for Initial O2 Mole Frac

        # Time Domain
        self.t0     = 1.e-10                                                    # Minimum Time for Selecting Training Collocation Points
        self.tEnd   = 1.e-4                                                     # Maximum Time for Selecting Training Collocation Points

        # Levels and grouping
        self.grouping       = 'CBM3' #'DPM10'
        self.PathToDatabase = ProPDEFldr + '/src/PDE/CGMasterEq/database/'
        self.LevelsFile     = self.PathToDatabase + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'                       # File Containing Molecule Levels
        self.BinMapFile     = self.PathToDatabase + '/grouping/O3_UMN/O2/LevelsMap_' + self.grouping + '.csv'    # File Containing Level-to-Group Mapping

        # Rates coefficients
        self.PathToRates    = self.PathToDatabase + 'kinetics/O3_UMN_' + self.grouping + '/'
        self.ReacFileName   = {'diss': 'Diss_Corrected.dat', 'exch': 'Exch_Type1.dat', 'inel': 'Inel.dat'}
        self.MasterEqMat    = {'excit': True, 'diss': True, 'recomb': True}
        self.ODEParams      = None                                              # Defined in 'generate_data' function

        ################################################################################################################################################################
        ### Paths
        self.WORKSPACE_PATH      = WORKSPACE_PATH                               # os.getenv('WORKSPACE_PATH')      
        self.ProPDEFldr          = ProPDEFldr                                   # $WORKSPACE_PATH/ProPDE/
        self.NNRunIdx            = '01a_mulSol_pts'                              # Training Case Identification Number 
        self.PathToRunFld        = self.ProPDEFldr + '/../' + self.PDEName + '/' + self.grouping \
                                   + '/Test_' + str(int(self.Ttr[0])) + '_' + self.NNRunIdx            # Path To Training Folder
        self.PathToFigFld        = self.PathToRunFld + '/Figures/'              # Path To Training Figures Folder 
        self.PathToDataFld       = self.PathToRunFld + '/Data/'                 # Path To Training Data Folder 
        self.PathToParamsFld     = self.PathToRunFld + '/Params/'               # Path To Training Parameters Folder 
        self.PathToHDF5Fld       = self.PathToRunFld + '/Params/'               # Path to HDF5 File For Loading Parameters

        ###############################################################################################################################################################
        ## NN Model Structure
        self.ApproxModel     = 'DeepONet'
        self.BranchLayers    = np.array([[self.uDim, 32, 32, 32]] * self.xDim)
        self.TrunkLayers     = np.array([1, 64, 64, 32])
        self.ScaleActFunsFlg = False
        self.NormLayerFlg    = False
        self.BiasBranchFlg   = True

        ################################################################################################################################################################
        ### Data set up
        self.NTrainCases = 80                               # Number of Training Cases

        self.NCollPts    = 0                                # Number of Training Collocation Pts

        self.NTrainPts   = 128                              # Number of Training Anchor Pts
        self.random_dist = False                            # Random distribution for Anchor Pts

        self.RandDataFlg = True                             # Shuffle Training Data
        
        ### Data Transformation
        self.scale_fac   = 1.e7
        self.var_name    = ['u','y','G_uy']
        self.var_func    = ['dir_trans','inv_trans']
        self.transform   = {
                                'u':    {
                                            'dir_trans': tf.math.log,
                                            'inv_trans': tf.math.exp
                                            },
                                'y':    {
                                            'dir_trans': lambda x: tf.math.log(x * self.scale_fac),
                                            'inv_trans': lambda x: tf.math.exp(x) / self.scale_fac
                                            },
                                'G_uy': {
                                            'dir_trans': tf.math.exp,
                                            'inv_trans': tf.math.log
                                            }
                                }

        ################################################################################################################################################################
        ### Training
        self.NEpoch          = 800000                       # Number of Epoches
        self.batch_size      = None                         # Number of Epoches

        # Validation Split
        self.data_test       = 0.25                         # Percentage of Train/Test data splitting
        
        # Learning Rate
        self.LearningRate    = 1.e-4                        # Initial Learning Rate
        self.lr_scheduler    = 'performance'                # Initial Scheduler: [None, 'exponential', 'performance', 'piecewise']
        # For 'piecewise' scheduler only
        self.bound           = [int(0.5*self.NEpoch), int(0.75*self.NEpoch)]
        self.lr_rel_value    = [1.0, 0.5, 0.2]

        # Optimizer
        self.Optimizer       = 'adam'                       # Optimizer
        self.OptimizerParams = [0.95, 1.e-7]                # Optimizer Parameters

        # Loss Weighting
        self.loss_name       = ['anch_pts','residual','l1_reg', 'l2_reg']
        self.loss_weights    = {
                                    'anch_pts': 1.,         # Loss from Anchor Pts
                                    'residual': 1.,         # Loss from Residual
                                    'l1_reg':   1.e-4,      # L1 regularization
                                    'l2_reg':   1.e-4       # L2 regularization
                                    }

        ################################################################################################################################################################
        ### Evaluating
        self.NTestCases  = 3                                # Number of Test Cases
        self.t0Test      = 0.0                              # Initial Time for Plotting Test Solutions
        self.tEndTest    = self.tEnd * 1.e1                 # Final Time for Plotting Test Solutions

        ################################################################################################################################################################
        ### Plotting Quantities
        self.xLogPlotFlg = True
        self.yLogPlotFlg = True
        if self.xLogPlotFlg:
            self.xMinValue  = self.t0
