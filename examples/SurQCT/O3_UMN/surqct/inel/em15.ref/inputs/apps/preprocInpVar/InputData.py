# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Input file for 'preprocInpVar' app
# --------------------------------------------------------------------------- #

from surqct import config as cfg


# Global
###############################################################################
SystemName      = 'O3_UMN'

# Paths
###############################################################################
PathToLevelsFld = cfg.paths.MERuns + 'database/levels/modified/'

# Preprocess diatomic potential input features
###############################################################################
LogBase  = 'e'
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
