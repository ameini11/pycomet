# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Input file for 'savePredData' app
# --------------------------------------------------------------------------- #

from surqct import config as cfg


# Global
###############################################################################
SystemName          = 'O3_UMN'
RatesType           = ['Inel']

# Paths
###############################################################################
PathToModelFld      = {'Inel': ''} # For each 'RatesType'
PathToLevelsFile    = {'O2': cfg.MERuns + 'database/levels/modified/O3_UMN/O2/VMin_VMax.csv'}

# Input/Output Quantities
###############################################################################
# Input variables
# -----------------------------------------------------------------------------
# i-th and j-th states
xDiatVars           = ['log_EVib','log_ERot','log_VMax','log_rIn','log_rOut','log_rOut-rIn','rIn/rMin_min','log_rOut/rMin_min']
xTTranVar           = {'i': True, 'j': True}
DeltaProperties     = True

# Output variables
# -----------------------------------------------------------------------------
MultFact            = 1.e9
RatesLogScale       = True

# Saving/Writing Quantities
###############################################################################
TTranVec            = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]
UseFinalVersion     = True
SuffixHDF5          = '/configs/config.03/run.03'   # In HDF5 database