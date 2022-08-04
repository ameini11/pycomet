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
    1,   # Write thermo and kinetics to Plato database
    1,   # Run 0D master equation simulations
    1,   # Postprocess 0D master equation simulations
    1    # Postprocess MCoarseAIR data and 0D master equation simulations
]

# Global
###############################################################################
SystemName      = 'O3_UMN'
KinSource       = 'QCT'
# ProcBoolList    = [(2,1,1,0), (2,0,0,0), (0,1,1,0)]
# TTranVec        = [1500.0, 5000.0, 10000.0, 20000.0]
ProcBoolList    = [(2,1,1,0), (0,1,1,0)]
TTranVec        = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]

# Rates masking
# >> TASKS: 1-2
###############################################################################
MinRateValue            = {
    'Diss':   1.e-14,
    'Inel':   1.e-14,
    'Exch_1': 1.e-15
}
PathToRatesMaskingHDF5  = None

# 0d simulation-based inputs
# >> TASKS: 2-3-4-5
###############################################################################
SuffixName = '/mixed/'

# Rates HDF5 database
# >> TASKS: 2-4
###############################################################################
SuffixHDF5 = '/Orig/'
