from surqct import config as cfg


# --------------------------------------------------------------------------- #
# Inputs for 'savePredData' app
# --------------------------------------------------------------------------- #

# Case
###############################################################################
SystemName          = 'O3_UMN'
RatesType           = ['Inel']

# Paths
###############################################################################
PathToRunFld        = {
    'Inel': ''
}
PathToLevelsFile    = {'O2': cfg.MERuns + 'database/levels/modified/O3_UMN/O2/VMin_VMax.csv'}

# Input/Output Quantities
###############################################################################
# Input variables
# -----------------------------------------------------------------------------
# i-th and j-th states
xDiatVars           = ['log_EVib','log_ERot','log_VMax','log_rOut']
xTTranVar           = {'i': True, 'j': True}
DeltaProperties     = True

# Output variables
# -----------------------------------------------------------------------------
MultFact            = 1.e9
RatesLogScale       = True

# Saving/Writing Quantities
###############################################################################
TTranVec            = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]
UseFinalVersion     = False