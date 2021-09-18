import os
import gc
import sys
import h5py
import time
import signal
import inspect
import datetime

import numpy as np
import tensorflow as tf
import multiprocessing as mp

from matplotlib import animation
from functools import wraps
from prode.config import headers


# Printing functions
###############################################################################
def print_main(text, start='\n', *args, **kwargs):
    print(start + headers['main'] + text, *args, **kwargs)

def print_submain(text, start='', *args, **kwargs):
    print(start + headers['main'] + '  ' + text, *args, **kwargs)

def input_main(text, start='', *args, **kwargs):
    return input(start + headers['main'] + text, *args, **kwargs)

def input_submain(text, start='', *args, **kwargs):
    return input(start + headers['main'] + '  ' + text, *args, **kwargs)

def warning(text, start='', *args, **kwargs):
    print(start + headers['warning'] + text, *args, **kwargs)

def raise_value_err(text):
    message = ''.join([headers['val_err'], text])
    raise ValueError(message)

# Decorators
###############################################################################
def timing(f):
    """Decorator for measuring the execution time of methods."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        delta = datetime.timedelta(seconds=(te - ts))
        print_submain(">> Function %r took %s" % (f.__name__, delta))
        sys.stdout.flush()
        return result
    return wrapper

# Decorator for applications --------------------------------------------------
LEN = 98

def compose(text='', start='', between=' '):
    end = ''.join(reversed(start))
    l = len(text)
    return start + between*int((LEN-len(start)*2-l)/2) \
        + text + between*int((LEN-len(start)*2-l)/2) + end

def app_decorator(name=None):
    def dec(fun):
        @wraps(fun)
        def wrapper(name=name, *args, **kwargs):
            ts = time.time()
            start = '# '
            print(
                "\n" + compose(start='# ', between='*') +
                "\n" + compose(text='PrODE ', start='# ') +
                "\n" + compose(start='# ', between='-') +
                "\n" + compose(
                    text='Machine-Learning-Based Approximators', start='# '
                ) +
                "\n" + compose(
                    text='for Ordinary Differential Equations (ODEs)',
                    start='# '
                ) +
                "\n" + compose(start='# ') +
                "\n" + compose(start='# ', between='*')
            )
            print_main('App `{}` starts\n'.format(name) + "="*LEN + "\n")
            result = fun(*args, **kwargs)
            te = time.time()
            delta = datetime.timedelta(seconds=(te - ts))
            if not name:
                name = fun.__name__
            print(app_epilog(name=name, delta=delta))
            # print("\n" + "=" * LEN)
            # print_main('App `{}` terminates'.format(name), start='')
            # print_submain(">> It took {}\n".format(delta))
            return result
        return wrapper
    return dec

def app_epilog(name=None, delta=None):
    s = '\n'+'='*LEN+'\n'+headers['main']+'App `{}` terminates\n'.format(name)
    if delta:
        return s + headers['main'] + '  ' + '>> It took {}\n'.format(delta)
    else:
        return s + ' '

# Handling data types
###############################################################################
def get_class(module, name):
    """Return a class object given its name and the module which belongs to."""
    for name_i, obj_i in inspect.getmembers(module, inspect.isclass):
        if name_i == name:
            return obj_i
    raise_value_err(
        "Class `{}` not found in module `{}`.".format(name, module.__name__)
    )

def get_class_name(cls):
    """Return the class name."""
    return cls.__class__.__name__

def make_dict(keys, values):
    """Convert two lists or two variables into a dictionary."""
    if isinstance(keys, (list, tuple)):
        if len(keys) != len(values):
            raise_value_err("Keys and values have different length.")
        return dict(zip(keys, values))
    return {keys: values}

def list_to_str(nums, precision=2):
    """Convert a list/tuple/array of numbers into a string."""
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(
        ", ".join(["{:.{}e}".format(x, precision) for x in nums])
    )

def get_num_args(func):
    """Get the number of arguments of a Python function."""
    if sys.version_info[0] == 2:
        return len(inspect.getargspec(func).args)
    sig = inspect.signature(func)
    return len(sig.parameters)

def get_args(obj):
    """Get the number of arguments of a Python function."""
    if inspect.isclass(obj):
        args = inspect.getfullargspec(obj).args
    else:
        args = list(dict(inspect.signature(obj).parameters).keys())

    removable = {'self', 'args', 'kwargs'}
    for arg in removable:
        if arg in args:
            args.remove(arg)

    return args

def get_args_values(module, args):
    """Get the number of arguments of a Python function."""
    _args = {}
    for arg in args:
        if hasattr(module, arg):
            _args.update({arg: getattr(module, arg)})
    return _args

# Others
###############################################################################
def kill_process():
    """Send terminating signal to job `pid`."""
    os.kill(os.getpid(), signal.SIGTERM)

def close_all_hdf5():
    """Close all opened hdf5 files."""
    for obj in gc.get_objects():        # Browse through all objects
        if isinstance(obj, h5py.File):  # Just HDF5 files
            try:
                obj.close()
            except:
                pass

# Environment setup -----------------------------------------------------------
def environment():
    num_threads = min(
        tf.config.threading.get_inter_op_parallelism_threads(),
        tf.config.threading.get_intra_op_parallelism_threads()
        )
    print(headers['main'] + "Environment setup\n" + "-"*98)
    print(" TensorFlow version:     {}".format(tf.__version__))
    print(" Eager execution:        {}".format(tf.executing_eagerly()))
    print(" GPUs available:         {}".format(
        len(tf.config.list_physical_devices('GPU')))
    )
    print(" CPUs available:         {}".format(
        len(tf.config.list_physical_devices('CPU')))
    )
    print(" Num. of threads:        {}\n".format(num_threads) + "-"*98)

# Array ops
###############################################################################
def get_cardinality(x):
    return get_cardinality(x[0]) if isinstance(x, (list,tuple)) else x.shape[0]

 

# def apply(func, args=None, kwds=None):
#     """Clear Tensorflow GPU memory after model execution.

#     References:

#         - https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
#     """
#     with Pool(1) as p:
#         if args is None and kwds is None:
#             r = p.apply(func)
#         elif kwds is None:
#             r = p.apply(func, args=args)
#         elif args is None:
#             r = p.apply(func, kwds=kwds)
#         else:
#             r = p.apply(func, args=args, kwds=kwds)
#     return r


# def save_animation(filename, xdata, ydata, y_reference=None, logy=False):
#     apply(
#         _save_animation,
#         args=(filename, xdata, ydata),
#         kwds={"y_reference": y_reference, "logy": logy},
#     )


# def _save_animation(filename, xdata, ydata, y_reference=None, logy=False):
#     """The animation figure window cannot be closed automatically.

#     References:

#         - https://stackoverflow.com/questions/43776528/python-animation-figure-window-cannot-be-closed-automatically
#     """
#     fig, ax = plt.subplots()
#     if y_reference is not None:
#         plt.plot(xdata, y_reference, "k-")
#     (ln,) = plt.plot([], [], "r-o")

#     def init():
#         ax.set_xlim(np.min(xdata), np.max(xdata))
#         if logy:
#             ax.set_yscale("log")
#             ax.set_ylim(max(1e-4, np.min(ydata)), np.max(ydata))
#         else:
#             ax.set_ylim(np.min(ydata), np.max(ydata))
#         return (ln,)

#     def update(frame):
#         ln.set_data(xdata, ydata[frame])
#         return (ln,)

#     ani = animation.FuncAnimation(
#         fig, update, frames=len(ydata), init_func=init, blit=True
#     )
#     ani.save(filename, writer="imagemagick", fps=30)
#     plt.close()

