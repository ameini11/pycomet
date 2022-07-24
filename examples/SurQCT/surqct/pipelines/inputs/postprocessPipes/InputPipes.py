# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Input file for 'postprocessPipes' pipeline script
# --------------------------------------------------------------------------- #


TASKS = [
    1,   # Save predicted rates matrices to HDF5
    0,   # Write thermo and kinetics to Plato database
    0,   # Run 0D master equation simulations
    0,   # Postprocess 0D master equation simulations
    0    # Postprocess MCoarseAir data and 0D master equation simulations
]

# Global
###############################################################################
SystemName      = 'O3_UMN'
KinSource       = 'SurQCT'
ProcBoolList    = [(2,1,1,0), (0,1,1,0)]
TTranVec        = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]

MainSuffix      = '/configs/config.03/run.03.ln/'
MainPrefix      = '/home/zanardi/Workspace/ML/PyCOMET/'

# Rates masking
# >> TASKS: 1-2
###############################################################################
RatesType               = ['Inel', 'Exch_1']
MinRateValue            = 1.e-16
PathToRatesMaskingHDF5  = None

# SurQCT model-based inputs
# >> TASKS: 1
###############################################################################
PathToModelFld  = {
    'Diss':   MainPrefix + '/run/SurQCT/O3_UMN/Diss/',
    'Inel':   MainPrefix + '/run/SurQCT/O3_UMN/Inel/' + MainSuffix,
    'Exch_1': MainPrefix + '/run/SurQCT/O3_UMN/Exch_1/' + MainSuffix
}

# 0d simulation-based inputs
# >> TASKS: 2-3-4-5
###############################################################################
SuffixName = '/em16/' + MainSuffix

# Rates HDF5 database
# >> TASKS: 2-4
###############################################################################
SuffixHDF5 = MainSuffix
