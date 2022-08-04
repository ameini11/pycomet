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
RatesType           = ['Exch_1']

# Paths
###############################################################################
MainSuffix       = '/configs.ln/config.03/run.03/'
MainPrefix       =  cfg.paths.Workspace + '/ML/PyCOMET/run/SurQCT/O3_UMN.temp/'
PathToModelFld   = {
    'Diss':   MainPrefix + '/Diss/' + MainSuffix,
    'Inel':   MainPrefix + '/Inel/' + MainSuffix,
    'Exch_1': MainPrefix + '/Exch_1/' + MainSuffix
}
PathToLevelsFile = {'O2': cfg.paths.MERuns + 'database/levels/modified/O3_UMN/O2/VMin_VMax_ln.csv'}

# Input/Output Quantities
###############################################################################
# Input variables
# -----------------------------------------------------------------------------
# i-th and j-th states
InpDiatVars         = ['log_EVib','log_ERot','log_VMax','log_rIn','log_rOut','log_rOut-rIn','rIn/rMin_min','log_rOut/rMin_min']
InpTTranVar         = {'i': True, 'j': True}
DeltaProperties     = True

# Output variables
# -----------------------------------------------------------------------------
MultFact            = 1.e9
RatesLogScale       = True

# Prediction
# -----------------------------------------------------------------------------
Dropout             = True
NEvals              = 1000

# Saving/Writing Quantities
###############################################################################
TTranVec            = [10000.]
UseFinalVersion     = True   # Use database with final values for other processes in "cfg.paths.HDF5Dtb/SurQCT/final/"
SuffixHDF5          = MainSuffix + '/dropout/Exch_1/'   # In HDF5 database
