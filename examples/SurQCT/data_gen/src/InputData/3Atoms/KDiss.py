# Case Name
# -----------------------------------------------------------------------------
RatesType = 'KDiss'

# Paths
# -----------------------------------------------------------------------------
WORKSPACE_PATH      = '/home/zanardi/WORKSPACE/'
SurQCTFld           = WORKSPACE_PATH + '/ML/PyCOMET/examples/SurQCT/'
PathToRunFld        = SurQCTFld + '/run.1/'     
PathToFigFld        = PathToRunFld + '/Figures/'
PathToDataFld       = PathToRunFld + '/Data/'
PathToHAHDF5File    = WORKSPACE_PATH + '/Air_Database/HDF5_Database_HighAccuracy/O3_UMN.hdf5'
PathToHDF5File      = WORKSPACE_PATH + '/Air_Database/HDF5_Database/O3_UMN.hdf5'
PathToLevelsFile    = WORKSPACE_PATH + '/Air_Database/Run_0D/database/levels/O2.csv'
PathToGrouping      = WORKSPACE_PATH + '/Air_Database/Run_0D/database/grouping/O3_UMN/O2/LevelsMap_DPM45.csv'

# Training Data Quanties
# -----------------------------------------------------------------------------
HighFidelityData    = False
xDiatVars           = ['EVib','ERot','VMax','ro','rMax']
TTranVecTrain       = [1500.0, 2500.0, 5000.0, 10000.0, 15000.0, 20000.0]

# Levels sampling strategies
#iLevelsIntFlg       = 1
#iLevelsVecTrain     = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]

#iLevelsIntFlg       = 2
#iLevelsSeedsVec     = [0, 4, 3, 1, 2]
#NiLevelsSampled     = 100

#iLevelsIntFlg       = 3
#NSamplesPerGroup     = 1
#iLevelsSeedsVec     = [0, 4, 3, 1, 2] # Same length of 'TTranVecTrain'

iLevelsIntFlg       = 4
PathToSampledLevels = WORKSPACE_PATH + '/Air_Database/Run_0D/database/levels/O2_Sampled_'

ValidPerc           = 20.
RandomState         = 3 # For random train/valid splitting

MultFact            = 1.e8
MinRateValue        = 1.e-17
RatesLogScale       = True

# Testing Data Quantities
# -----------------------------------------------------------------------------
TTranVecTest        = [1500.0, 2500.0, 5000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 15000.0, 20000.0]
iLevelsVecTest      = [4580, 4581, 6058, 6097, 6105, 6108, 6111]
TTranVecExtra       = [300.0, 40000.0]
