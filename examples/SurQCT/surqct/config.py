# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# --------------------------- CONFIGURATION FILE ---------------------------- #


import os
import surqct
from argparse import Namespace


# SurQCT ----------------------------------------------------------------------
HEADERS = {
    'main':    '[SurQCT]: ',
    'warning': '[SurQCT]:   WARNING! ',
    'err':     'from SurQCT\n>>> '
}

NUM_THREADS = 4

# Paths -----------------------------------------------------------------------
paths = {
    'Workspace': '/home/zanardi/Workspace/',
    'CoarseAIR': '/home/zanardi/Codes/CoarseAIR/coarseair/',
    'MERuns':    '/home/zanardi/Workspace/AirDatabase/Run0D/',
    'HDF5Dtb':   '/home/zanardi/Workspace/AirDatabase/HDF5/',
    'SurQCT':    os.path.dirname(surqct.__file__)
}
paths = Namespace(**{
    k: os.path.abspath(v)+'/' for k, v in paths.items()
})


# TensorFlow ------------------------------------------------------------------
tf_cfg = {
    'setup': {
        'EPSILON': 1.e-15,
        'DTYPE': 'float64'
    },
    'env': {
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_XLA_FLAGS': f'--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' \
           f'{paths.SurQCT}/__main__.py'
    }
}

# Matplotlib ------------------------------------------------------------------
mpl_cfg = {
    'style': paths.Workspace + '/Research/styles/matplotlib/paper_1column_venturi.mplstyle'
}

# Standard output -------------------------------------------------------------
DISPLAY_LEN = 98

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
