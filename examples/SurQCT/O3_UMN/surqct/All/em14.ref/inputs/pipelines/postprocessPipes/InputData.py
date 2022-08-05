# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Input file for 'postprocessPipes' pipeline script
# --------------------------------------------------------------------------- #

from surqct import config as cfg


TASKS = [
    0,   # Save predicted rates matrices to HDF5
    0,   # Write thermo and kinetics to Plato database
    0,   # Run 0D master equation simulations
    0,   # Postprocess 0D master equation simulations
    1    # Postprocess MCoarseAIR data and 0D master equation simulations
]

# Global
###############################################################################
SystemName      = 'O3_UMN'
KinSource       = 'SurQCT'
ProcBoolList    = [(2,1,1,0), (0,1,1,0)]
TTranVec        = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]

# MainSuffix      = ''
# MainPrefix      =  cfg.paths.Workspace + '/ML/PyCOMET/run/SurQCT/O3_UMN/final/'

# Rates masking
# >> TASKS: 1-2
###############################################################################
# RatesType               = ['Inel', 'Exch_1']
MinRateValue            = {
    'Diss':   1.e-14,
    'Inel':   1.e-14,
    'Exch_1': 1.e-14
}
PathToRatesMaskingHDF5  = None

# SurQCT model-based inputs
# >> TASKS: 1
###############################################################################
# PathToModelFld   = {
#     'Diss':   MainPrefix + '/Diss/' + MainSuffix,
#     'Inel':   MainPrefix + '/Inel/' + MainSuffix,
#     'Exch_1': MainPrefix + '/Exch_1/' + MainSuffix
# }
# PathToLevelsFile = {
#     'O2': cfg.paths.MERuns + 'database/levels/modified/O3_UMN/O2/VMin_VMax_ln.csv'
# }

# 0d simulation-based inputs
# >> TASKS: 2-3-4-5
###############################################################################
SuffixName = '/em14.ref/'

# Rates HDF5 database
# >> TASKS: 2-4
###############################################################################
SuffixHDF5 = '/final.ref/'
