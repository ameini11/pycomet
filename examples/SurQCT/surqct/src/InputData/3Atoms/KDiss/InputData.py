# Case Name
# -----------------------------------------------------------------------------
RatesType = 'KDiss'

# Paths
# -----------------------------------------------------------------------------
WORKSPACE_PATH      = '/home/zanardi/WORKSPACE/'
SurQCTFld           = WORKSPACE_PATH + '/ML/PyCOMET/pycomet/examples/SurQCT/'
PathToRunFld        = SurQCTFld + '/run.1/'
PathToHFHDF5File    = WORKSPACE_PATH + '/AirDatabase/HDF5/Active/O3_UMN.hdf5'
PathToHDF5File      = WORKSPACE_PATH + '/AirDatabase/HDF5/Orig/O3_UMN.hdf5'
PathToLevelsFile    = WORKSPACE_PATH + '/AirDatabase/Run0D/database/levels/modified/46DPM_Sampled/O2.csv'
PathToGrouping      = WORKSPACE_PATH + '/AirDatabase/Run0D/database/grouping/O3_UMN/O2/LevelsMap_DPM45.csv'
PathToSampledLevels = WORKSPACE_PATH + '/AirDatabase/Run0D/database/levels/samples/O2_Sampled_'

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

# Preprocess diatomic potential input features
# -----------------------------------------------------------------------------
# Inelastic processes
# InpManip = {
#     'Var':    {'ref': 'VMax', 'norm': None},
#     'MinMax': {'ref':   None, 'norm': None},
# }
InpManip = {
    'Var': {'ref': 'VMin', 'norm': 'VMax'},
    'MinMax': {'ref': min, 'norm': min},
}
# InpManip = {
#     'Var': {'ref': 'EInt', 'norm': 'VMax'},
#     'MinMax': {'ref': min, 'norm': min},
# }
# InpManip = {
#     'Var': {'ref': 'VMax', 'norm': 'VMin'},
#     'MinMax': {'ref': min, 'norm': min},
# }

# Heterogenous exchange processes
# InpManip = {
#     'Var': {'ref': 'VMin', 'norm': 'VMax'},
#     'MinMax': {'ref': None, 'norm': max},
# }
# InpManip = {
#     'Var': {'ref': 'VMax', 'norm': 'VMin'},
#     'MinMax': {'ref': min, 'norm': max},
# }