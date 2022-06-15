# Case Name
# -----------------------------------------------------------------------------
NbAtoms   = 3
RatesType = 'KExcit'
ExcitType = 'KInel' # ['KInel', 'KExch', 'KExcit']

# Paths
# -----------------------------------------------------------------------------
WORKSPACE_PATH      = '/home/zanardi/WORKSPACE/'
SurQCTFld           = WORKSPACE_PATH + '/ML/PyCOMET/pycomet/examples/SurQCT/'
PathToRunFld        = SurQCTFld + '/O3/inel/'
# Rates dtb
PathToHAHDF5File    = WORKSPACE_PATH + '/AirDatabase/HDF5/Active/O3_UMN.hdf5'
PathToHDF5File      = WORKSPACE_PATH + '/AirDatabase/HDF5/Orig/O3_UMN.hdf5'
# Molecules dtb
Molecules           = [('O2','UMN')]*2
PathToLevelsFld     = WORKSPACE_PATH + '/AirDatabase/Run0D/database/levels/modified/'
PathToLevelsFile    = [PathToLevelsFld + '/O2_UMN/VMin_VMax.csv']*2
PathToGrouping      = WORKSPACE_PATH + '/AirDatabase/Run0D/database/grouping/O3_UMN/O2/LevelsMap_DPM45.csv'
PathToSampledLevels = WORKSPACE_PATH + '/AirDatabase/Run0D/database/levels/samples/Uniform/O2_500_'

# Training Quanties
# -----------------------------------------------------------------------------
HighFidelityData    = False
# Input variables for i-th and j-th state
xDiatVars           = ['log_EVib','log_ERot','log_VMax','EVib','ERot','VMax']
xTTranVar           = {'i': False, 'j': False}
DeltaProperties     = True
NSamplesNoise       = 0    # Adding noise to data

# Translational temperatures
TTranVecTrain       = [20000.]

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

ValidPerc           = 20.

# Rates specs
NbProcesses         = 2 # 2 or 3: [Elastic, Inelastic, Exchange Type 1]
HomoExchNb          = 1

ExoEndoFlg          = False   # Only Exothermic Rates
PartFnRatioFlg      = False   # Only qj > qi
ReconstructExoth    = False   # Reconstruct Exothermic Rates

# SampleWeights       = [(1.e-13, 3.),(1.e-12, 6.),(1.e-11, 9.),(1.e-10, 12.),(1.e-9, 15.)]

MultFact            = 1.e9
MinRateValue        = 1.e-15 # 1.e-17
RatesLogScale       = True

# Testing Quantities
# -----------------------------------------------------------------------------
TTranVecTest        = [6000.0, 10000.0]
iLevelsVecTest      = [4580, 4581, 6058, 6097, 6105, 6108, 6111]
TTranVecExtra       = [300.0, 50000.0]

# Preprocess input features
# -----------------------------------------------------------------------------
# Inelastic processes
# InpManip = {
#     'Var':    {'ref': None, 'norm': None},
#     'MinMax': {'ref': None, 'norm': None},
# }
# InpManip = {
#     'Var':    {'ref': 'VMax', 'norm': None},
#     'MinMax': {'ref':   None, 'norm': None},
# }
InpManip = {
    'Var':    {'ref': 'VMin', 'norm': 'VMax'},
    'MinMax': {'ref':    min, 'norm':    min}
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