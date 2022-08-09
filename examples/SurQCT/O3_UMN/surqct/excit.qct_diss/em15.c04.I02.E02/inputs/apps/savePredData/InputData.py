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
RatesType           = ['Inel', 'Exch_1']

# Paths
###############################################################################
PathToModelFld      = {  # For each 'RatesType'
    'Inel': cfg.paths.Workspace + '/ML/PyCOMET/run/SurQCT/O3_UMN.temp/Inel/configs.ln/config.03/run.03',
    'Exch_1': cfg.paths.Workspace + '/ML/PyCOMET/run/SurQCT/O3_UMN.temp/Exch_1/configs.ln/config.03/run.03'
}
PathToLevelsFile    = {'O2': cfg.paths.MERuns + 'database/levels/modified/O3_UMN/O2/VMin_VMax_ln.csv'}

# Input/Output Quantities
###############################################################################
# Input variables
# -----------------------------------------------------------------------------
# i-th and j-th states
InpDiatVars         = ['log_EVib','log_ERot','log_VMax','log_rIn','log_rOut','log_rOut-rIn','log_rMax']
InpTTranVar         = {'i': True, 'j': True}
DeltaProperties     = True

# Output variables
# -----------------------------------------------------------------------------
MultFact            = 1.e9
RatesLogScale       = True

# Prediction
# -----------------------------------------------------------------------------
Dropout             = False
NEvals              = 100

# Saving/Writing Quantities
###############################################################################
TTranVec            = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]
UseFinalVersion     = True   # Use database with final values for other processes in "cfg.paths.HDF5Dtb/SurQCT/final/"
FinalVerFld         = cfg.paths.HDF5Dtb + 'QCT/Orig/Diss/'
SuffixHDF5          = '/configs.ln/config.03/dropout/'   # In HDF5 database
