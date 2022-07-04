# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# --------------------------- CONFIGURATION FILE ---------------------------- #


# SurQCT ----------------------------------------------------------------------
HEADERS = {
    'main':    '[SurQCT]: ',
    'warning': '[SurQCT]:   WARNING! ',
    'err':     'from SurQCT\n>>> '
}

NUM_THREADS = 4

# Paths -----------------------------------------------------------------------
SurQCT    = '/home/zanardi/WORKSPACE/ML/PyCOMET/pycomet/examples/SurQCT/' # This package
CoarseAIR = '/home/zanardi/WORKSPACE/NeqPlasma/CoarseAIR/coarseair/'
MERuns    = '/home/zanardi/WORKSPACE/AirDatabase/Run0D/run/'
HDF5Dtb   = '/home/zanardi/WORKSPACE/AirDatabase/HDF5/'

# Standard output -------------------------------------------------------------
DISPLAY_LEN = 80

def compose(text='', start='', between=' '):
    end = ''.join(reversed(start))
    l = len(text)
    return start + between*int((DISPLAY_LEN-len(start)*2-l)/2) \
        + text + between*int((DISPLAY_LEN-len(start)*2-l)/2) + end

PKG_HEADER = "\n" + compose(start='# ', between='*') + \
    "\n" + compose(text='SurQCT', start='# ') + \
    "\n" + compose(start='# ', between='-') + \
    "\n" + compose(
        text='Surrogate modeling for QCT calculations ', start='# '
    ) + \
    "\n" + compose(start='# ') + \
    "\n" + compose(start='# ', between='*')

EPILOG = '\n' + '='*DISPLAY_LEN + '\n '
