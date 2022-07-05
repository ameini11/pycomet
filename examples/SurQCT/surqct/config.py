# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# --------------------------- CONFIGURATION FILE ---------------------------- #


import os
import surqct


# SurQCT ----------------------------------------------------------------------
HEADERS = {
    'main':    '[SurQCT]: ',
    'warning': '[SurQCT]:   WARNING! ',
    'err':     'from SurQCT\n>>> '
}

NUM_THREADS = 4

# TensorFlow ------------------------------------------------------------------
tf_setup = {
    'EPSILON':              1.e-15,
    'DTYPE':                'float64'
}

tf_env = {
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'TF_XLA_FLAGS':         f'--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' \
       f'{os.path.dirname(surqct.__file__)}/__main__.py'
}

# Paths -----------------------------------------------------------------------
CoarseAIR = '/home/zanardi/WORKSPACE/NeqPlasma/CoarseAIR/coarseair/'
MERuns    = '/home/zanardi/WORKSPACE/AirDatabase/Run0D/'
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
