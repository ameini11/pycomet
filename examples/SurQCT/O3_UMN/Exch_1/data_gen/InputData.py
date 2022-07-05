from surqct import config as cfg


# Case
###############################################################################
SystemName          = 'O3_UMN'
RatesType           = 'Exch_1'    # ['Diss', 'Inel', 'Exch_1', 'Exch_2']

# Paths
###############################################################################
PathToHDF5File      = cfg.HDF5Dtb + 'Active/O3_UMN.hdf5' # `None` or path to high fidelity data
PathToLevelsFld     = cfg.MERuns + 'database/levels/modified/'
PathToLevelsFile    = {'O2': PathToLevelsFld + 'O3_UMN/O2/VMin_VMax.csv'}

# Training Quanties
###############################################################################
# Input variables
# -----------------------------------------------------------------------------
# i-th and j-th states
xDiatVars           = ['log_EVib','log_ERot','log_VMax','log_ro']
xTTranVar           = {'i': True, 'j': True}
DeltaProperties     = True
NSamplesNoise       = 0    # Adding noise to data

# Translational temperatures
TTranVecTrain       = [1500.0, 5000.0, 10000.0, 15000.0, 20000.0]

# Levels sampling strategy
# -----------------------------------------------------------------------------
# iLevelsIntFlg       = 1
# iLevelsVecTrain     = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]

# iLevelsIntFlg       = 2
# iLevelsSeedsVec     = [0, 4, 3, 1, 2]
# NiLevelsSampled     = 100

# iLevelsIntFlg       = 3
# NSamplesPerGroup    = 3
# iLevelsSeedsVec     = [0, 4, 3, 1, 2] # Same length of 'TTranVecTrain'
# PathToGrouping      = {'O2': cfg.MERuns + 'database/grouping/O3_UMN/O2/LevelsMap_DPM45.csv'}

iLevelsIntFlg       = 4
PathToSampledLevels = {'O2': cfg.MERuns + 'database/levels/samples/Active_Sampled_with4580/O2_T'}

# Rates manipulation
# -----------------------------------------------------------------------------
ExoEndoFlg          = True      # Only Exothermic Rates
PartFnRatioFlg      = False     # Only qi > qj
ReconstructExoth    = True      # Reconstruct Exothermic Rates

SampleWeights       = [(1.e-13, 3.),(1.e-12, 6.),(1.e-11, 9.),(1.e-10, 12.),(1.e-9, 15.)]

MultFact            = 1.e9
MinRateValue        = 1.e-15
RatesLogScale       = True

# Misc
# -----------------------------------------------------------------------------
ValidPerc           = 20.

# Testing Quantities
###############################################################################
TTranVecTest        = [1500.0]
iLevelsVecTest      = [4580, 4581, 6058, 6097, 6105, 6108, 6111]
TTranVecExtra       = [300.0, 50000.0]

# Preprocess diatomic potential input features
###############################################################################
# Inelastic processes
# InpManip = {
#     'Var':    {'ref': 'VMax', 'norm': None},
#     'MinMax': {'ref':   None, 'norm': None},
# }
InpManip = {
    'Var':      {'ref': 'VMin', 'norm': 'VMax'},
    'MinMax':   {'ref':    min, 'norm':    min}
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
