import sys
import time
import datetime

from functools import wraps
from pycomet import config as cfg
from pycomet.utilities import stdout as pcm_stdout


def timing(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = fun(*args, **kwargs)
        te = time.time()
        delta = datetime.timedelta(seconds=(te - ts))
        pcm_stdout.print_submain(
            ">> Function %r took %s" % (fun.__name__, delta)
        )
        sys.stdout.flush()
        return result
    return wrapper

def main_fun(fun):
    @wraps(fun)
    def wrapper():
        pcm_stdout._pkg_header()
        return fun()
    return wrapper

def app_fun(name=None):
    def dec(fun):
        @wraps(fun)
        def wrapper(name=name, *args, **kwargs):
            ts = time.time()
            pcm_stdout.print_main(
                f'App `{name}` starts' + pcm_stdout._EPILOG, start=''
            )
            result = fun(*args, **kwargs)
            te = time.time()
            delta = datetime.timedelta(seconds=(te - ts))
            if not name:
                name = fun.__name__
            print(pcm_stdout._EPILOG[:-2])
            pcm_stdout.print_main(
                'App `{}` terminates'.format(name), start=''
            )
            pcm_stdout.print_submain('>> It took %s\n' % delta)
            return result
        return wrapper
    return dec
