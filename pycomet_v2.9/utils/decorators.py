import sys
import time
import datetime

from functools import wraps
from pycomet import utils
from pycomet import config as cfg


def timing(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = fun(*args, **kwargs)
        te = time.time()
        delta = datetime.timedelta(seconds=(te - ts))
        utils._print(
            ">> Function %r took %s" % (fun.__name__, delta), lev=1
        )
        sys.stdout.flush()
        return result
    return wrapper

def app_dec(name=None):
    def dec(fun):
        @wraps(fun)
        def wrapper(name=name, *args, **kwargs):
            ts = time.time()
            utils._print(
                f'App `{name}` starts' + utils._EPILOG, start='', lev=0
            )
            result = fun(*args, **kwargs)
            te = time.time()
            delta = datetime.timedelta(seconds=(te - ts))
            if not name:
                name = fun.__name__
            print(utils._EPILOG[:-2])
            utils._print(
                'App `{}` terminates'.format(name), start='', lev=0
            )
            utils._print('>> It took %s\n' % delta, lev=1)
            return result
        return wrapper
    return dec
