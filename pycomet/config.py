# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                scientific computations and operator discovery               #
#                                                                             #
# *************************************************************************** #

# --------------------------- CONFIGURATION FILE ---------------------------- #


import os
import pycomet


# PyCOMET ---------------------------------------------------------------------
headers = {
    'main':    '[PyCOMET]: ',
    'warning': '[PyCOMET]:   WARNING! ',
    'err':     'from PyCOMET\n>>> '
}

# TensorFlow ------------------------------------------------------------------
tf_setup = {
    'EPSILON':              1.e-15,
    'DTYPE':                'float64',
    'NUM_THREADS':          4
}

tf_env = {
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'TF_XLA_FLAGS':         f'--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' \
       f'{os.path.dirname(pycomet.__file__)}/__main__.py'
}

# Database --------------------------------------------------------------------
__DTB__ = '/home/zanardi/WORKSPACE/NeqPlasma/HyperNet/database'

# Standard output -------------------------------------------------------------
DISPLAY_LEN = 98

def compose(text='', start='', between=' '):
    end = ''.join(reversed(start))
    l = len(text)
    return start + between*int((DISPLAY_LEN-len(start)*2-l)/2) \
        + text + between*int((DISPLAY_LEN-len(start)*2-l)/2) + end

PKG_HEADER = "\n" + compose(start='# ', between='*') + \
    "\n" + compose(text='PyCOMET ', start='# ') + \
    "\n" + compose(start='# ', between='-') + \
    "\n" + compose(
        text='Physics-informed machine learning for ', start='# '
    ) + \
    "\n" + compose(
        text='scientific computations and operator discovery',
        start='# '
    ) + \
    "\n" + compose(start='# ') + \
    "\n" + compose(start='# ', between='*')

EPILOG = '\n' + '='*DISPLAY_LEN + '\n '
