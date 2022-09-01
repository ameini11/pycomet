# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                 scientific computing and operator discovery                 #
#                                                                             #
# *************************************************************************** #

# --------------------------- CONFIGURATION FILE ---------------------------- #


import os


# Global ----------------------------------------------------------------------
_NUM_THREADS = 4

# PyCOMET ---------------------------------------------------------------------
_NAME = 'PyCOMET'

_HEADERS = {
    'main':    '['+_NAME+']: ',
    'warning': '['+_NAME+']:   WARNING! ',
    'err':     'from '+_NAME+'\n>>> '
}

_TAB = 2

# TensorFlow ------------------------------------------------------------------
_TF = {
    'setup': {
        'EPSILON': 1.e-15,
        'FLOATX': 'float64',
        'XLA': False
    },
    'env': {
        'TF_CPP_MIN_LOG_LEVEL': '3',
        # 'TF_XLA_FLAGS': f'--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' + \
        #     f'{os.path.dirname(pycomet.__file__)}/__main__.py'
    }
}

# Matplotlib ------------------------------------------------------------------
_MPL = {
    'style': 'default'
    # 'style': paths.Workspace + '/Research/styles/matplotlib/paper_1column_venturi.mplstyle'
}

# Standard output -------------------------------------------------------------
_DISPLAY_LEN = 98

_DESCR_LINES = (
    'Physics-informed machine learning for',
    'scientific computing and operator discovery'
)
