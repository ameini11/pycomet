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
    1,   # Save predicted rates matrices to HDF5
    1,   # Write thermo and kinetics to Plato database
    1,   # Run 0D master equation simulations
    1,   # Postprocess 0D master equation simulations
    1    # Postprocess MCoarseAIR data and 0D master equation simulations
]

# Global
###############################################################################
SystemName      = 'O3_UMN'
KinSource       = 'SurQCT'
ProcBoolList    = [(2,1,1,0)]
TTranVec        = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]

MainSuffix      = '/config.10/run.02/'
MainPrefix      =  cfg.paths.Workspace + '/ML/PyCOMET/run/SurQCT/O3_UMN/tests/'

# Rates masking
# >> TASKS: 1-2
###############################################################################
RatesType               = ['Diss']
MinRateValue            = {
    'Diss':   1.e-15,
    'Inel':   1.e-15,
    'Exch_1': 1.e-15
}
PathToRatesMaskingHDF5  = None

# SurQCT model-based inputs
# >> TASKS: 1
###############################################################################
PathToModelFld   = {
    'Diss':   MainPrefix + '/Diss/' + MainSuffix,
    'Inel':   MainPrefix + '/Inel/',
    'Exch_1': MainPrefix + '/Exch_1/'
}
PathToLevelsFile = {
    'O2': cfg.paths.MERuns + 'database/levels/modified/O3_UMN/O2/VMin_VMax_ln.csv'
}

# 0d simulation-based inputs
# >> TASKS: 2-3-4-5
###############################################################################
SuffixName = '/diss.qct_excit/em15.c10.r02/'

# Rates HDF5 database
# >> TASKS: 2-4
###############################################################################
SuffixHDF5 = '/diss.qct_excit/c10.r02/'
