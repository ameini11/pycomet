import sys
import time
import datetime

from functools import wraps
from pycomet import utils


def timing(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = fun(*args, **kwargs)
        te = time.time()
        delta = datetime.timedelta(seconds=(te - ts))
        utils.print_fn(
            ">> Function %r took %s" % (fun.__name__, delta), start='', lev=1 
        )
        sys.stdout.flush()
        return result
    return wrapper

def app_dec(name=None):
    def dec(fun):
        @wraps(fun)
        def wrapper(name=name, *args, **kwargs):
            ts = time.time()
            utils.print_fn(
                f'App `{name}` starts' + utils.EPILOG, start='', lev=0
            )
            result = fun(*args, **kwargs)
            te = time.time()
            delta = datetime.timedelta(seconds=(te - ts))
            if not name:
                name = fun.__name__
            print(utils.EPILOG[:-2])
            utils.print_fn(
                'App `{}` terminates'.format(name), start='', lev=0
            )
            utils.print_fn('>> It took %s\n' % delta, start='', lev=1)
            return result
        return wrapper
    return dec

def lambda_lay_dec(fun):
    @wraps(fun)
    def wrapper(inputs, **kwargs):
        if isinstance(inputs, dict):
            result = fun(**inputs, **kwargs)
        elif isinstance(inputs, (list,tuple)):
            result = fun(*inputs, **kwargs)
        else:
            result = fun(inputs, **kwargs)
        return result
    return wrapper