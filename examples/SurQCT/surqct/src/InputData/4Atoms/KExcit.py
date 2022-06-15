# Case Name
# -----------------------------------------------------------------------------
NbAtoms       = 4
RatesType     = 'KExcit'
ExcitType     = 'KExcit'
SameMolecules = True

# Paths
# -----------------------------------------------------------------------------
WORKSPACE_PATH      = '/home/zanardi/WORKSPACE/'
SurQCTFld           = WORKSPACE_PATH + '/ML/PyCOMET/examples/SurQCT/'
PathToRunFld        = SurQCTFld + '/run.1/'     
PathToFigFld        = PathToRunFld + '/Figures/'
PathToDataFld       = PathToRunFld + '/Data/'
PathToHAHDF5File    = WORKSPACE_PATH  + '/Air_Database/HDF5_Database_HighAccuracy/O3_UMN.hdf5'
PathToHDF5File      = WORKSPACE_PATH  + '/Air_Database/HDF5_Database/O3_UMN.hdf5'
Molecules           = ['O2']*4
PathToLevelsFile    = [WORKSPACE_PATH + '/Air_Database/Run_0D/database/levels/O2_nd.csv']*4
PathToDiatFile      = [WORKSPACE_PATH + '/CoarseAIR/coarseair/dtb/Molecules/O2/UMN/FromUMN_Sorted.inp']*4
PathToGrouping      = [WORKSPACE_PATH + '/Air_Database/Run_0D/database/grouping/O3_UMN/O2/LevelsMap_DPM45.csv']*4

# Training Quanties
# -----------------------------------------------------------------------------
xVarsVec            = ['EVib','ERot','rMin','rMax','VMin','VMax','Tau','ri','ro'] #['EVib','ERot','VMin','VMax','Tau','ri','ro','vqn','jqn']
OtherVarI           = '_DeltaI'
OtherVarJ           = '_DeltaJ'
NSamplesNoise       = 0
RandDataFlg         = True                                                                      # Randomize Training Data 
TTranVecTrain       = np.array([1500.0, 5000.0, 10000.0, 15000.0, 20000.0])#np.array([1500.0, 5000.0, 8000.0, 12000.0, 15000.0, 20000.0, 30000.0, 50000.0])])


# Translational temperatures
TTranVecTrain       = [1500.0, 5000.0, 10000.0, 15000.0, 20000.0]

# Levels sampling strategy
#iLevelsIntFlg       = 1
#iLevelsVecTrain     = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]

#iLevelsIntFlg       = 2
#iLevelsSeedsVec     = [0, 4, 3, 1, 2]
#NiLevelsSampled     = 100

#iLevelsIntFlg       = 3
#NSamplesPerGroup    = 3
#iLevelsSeedsVec     = [0, 4, 3, 1, 2] # Same length of 'TTranVecTrain'

iLevelsIntFlg       = 4
PathToSampledLevels = WORKSPACE_PATH  + '/Air_Database/Run_0D/database/levels/DPM_Sampled/O2_Sampled_'

ExoEndoFlg          = False

ValidPerc           = 20.

# Testing Quantities
# -----------------------------------------------------------------------------
TTranVecTest        = [6000.0, 10000.0]
iLevelsVecTest      = [4580, 4581, 6058, 6097, 6105, 6108, 6111]
TTranVecExtra       = [300.0, 50000.0]