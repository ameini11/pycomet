import sys
import time
import datetime

from tqdm import tqdm
from functools import wraps
from surqct import config as cfg


# Printing functions
###############################################################################
def print_main(text, start='\n', verbose=True, *args, **kwargs):
    if verbose:
        print(start + cfg.HEADERS['main'] + text, *args, **kwargs)

def print_submain(text, start='', delimiter='  ', verbose=True, *args, **kwargs):
    if verbose:
        print(start + cfg.HEADERS['main'] + delimiter + text, *args, **kwargs)

def input_main(text, start='', *args, **kwargs):
    return input(start + cfg.HEADERS['main'] + text, *args, **kwargs)

def input_submain(text, start='', delimiter='  ', *args, **kwargs):
    return input(start + cfg.HEADERS['main'] + delimiter + text, *args, **kwargs)

def warning(text, start='', *args, **kwargs):
    print(start + cfg.HEADERS['warning'] + text, *args, **kwargs)

def raise_value_err(text):
    message = ''.join([cfg.HEADERS['err'], text])
    raise ValueError(message)

# Decorators
###############################################################################
def timing(fun):
    """Decorator for measuring the execution time of methods."""
    @wraps(fun)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = fun(*args, **kwargs)
        te = time.time()
        delta = datetime.timedelta(seconds=(te - ts))
        print_submain(">> Function %r took %s" % (fun.__name__, delta))
        sys.stdout.flush()
        return result
    return wrapper

# Decorator for applications --------------------------------------------------
def main_decorator(fun):
    @wraps(fun)
    def wrapper():
        print(cfg.PKG_HEADER + '\n')
        result = fun()
        return result
    return wrapper

def app_decorator(name=None):
    def dec(fun):
        @wraps(fun)
        def wrapper(name=name, *args, **kwargs):
            ts = time.time()
            print_main(f'App `{name}` starts' + cfg.EPILOG, start='')
            result = fun(*args, **kwargs)
            te = time.time()
            delta = datetime.timedelta(seconds=(te - ts))
            if not name:
                name = fun.__name__
            print(cfg.EPILOG[:-2])
            print_main('App `{}` terminates'.format(name), start='')
            print_submain('>> It took %s\n' % delta)
            return result
        return wrapper
    return dec

# Handling libraries
###############################################################################
def import_matplotlib():
    import matplotlib as mpl
    if cfg.mpl_cfg['style'] is not None:
        mpl.style.use(cfg.mpl_cfg['style'])
    return mpl
