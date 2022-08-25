import os
import tensorflow as tf

from pycomet import config as cfg


_EPILOG = '\n' + '='*cfg._DISPLAY_LEN + '\n '
_TABS = np.arange(5)*min(cfg._TAB, 2)

# Printing --------------------------------------------------------------------
def _print(text, start='\n', lev=0, verbose=True):
    if verbose:
        print(start + cfg._HEADERS['main'] + ' '*_TABS[lev] + text)

def _input(text, start='', lev=0):
    return input(start + cfg._HEADERS['main'] + ' '*_TABS[lev] + text)

def _warning(text, start='', *args, **kwargs):
    print(start + cfg._HEADERS['warning'] + text, *args, **kwargs)

def _raise_value_err(text):
    message = ''.join([cfg._HEADERS['err'], text])
    raise ValueError(message)

def _raise_module_not_found_error(text):
    message = ''.join([cfg._HEADERS['err'], text])
    raise ModuleNotFoundError(message)

# Environment -----------------------------------------------------------------
def _environment():
    num_threads = min([
        os.cpu_count(),
        tf.config.threading.get_inter_op_parallelism_threads(),
        tf.config.threading.get_intra_op_parallelism_threads()
    ])
    print(
        cfg._HEADERS['main'] + "Environment setup" + \
        "\n" + "-" * cfg._DISPLAY_LEN + \
        "\n  TensorFlow version:     {}".format(tf.__version__) + \
        "\n  Eager execution:        {}".format(tf.executing_eagerly()) + \
        "\n  Num. of GPUs available: {}".format(
            len(tf.config.list_physical_devices('GPU'))
        ) + \
        "\n  Num. of CPUs available: {}".format(
            len(tf.config.list_physical_devices('CPU'))
        ) + \
        "\n  Num. of threads:        {}\n".format(num_threads) + \
        "-"*cfg._DISPLAY_LEN
    )

# Package header --------------------------------------------------------------
def _compose(text='', start='', between=' '):
    end = ''.join(reversed(start))
    l = len(text)
    return start + between*int((cfg._DISPLAY_LEN-len(start)*2-l)/2) + \
        text + between*int((cfg._DISPLAY_LEN-len(start)*2-l)/2) + end

def _compute_blanks(text):
    return ' '*((cfg._DISPLAY_LEN-len(text))%2)

def _pkg_header():
    info = ''.join([
        "\n" + _compose(
            text=line+_compute_blanks(line), start='# '
        ) for line in cfg._DESCR_LINES
    ])
    print(
        "\n" + _compose(start='# ', between='*') + \
        "\n" + _compose(
            text=cfg._NAME+_compute_blanks(cfg._NAME), start='# '
        ) + \
        "\n" + _compose(start='# ', between='-') + info + \
        "\n" + _compose(start='# ') + \
        "\n" + _compose(start='# ', between='*') + '\n'
    )
