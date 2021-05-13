import inspect
import sys
import time
import datetime

import numpy         as np
import tensorflow    as tf

from functools       import wraps
from multiprocessing import Pool


def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        delta = datetime.timedelta(seconds=(te - ts))
        print("[PrODE]: >> Function %r took %s" % (f.__name__, delta))
        sys.stdout.flush()
        return result

    return wrapper


def apply(func, args=None, kwds=None):
    """Clear Tensorflow GPU memory after model execution.

    References:

        - https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
    """
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r


def make_dict(keys, values):
    """Convert two lists or two variables into a dictionary."""
    if isinstance(keys, (list, tuple)):
        if len(keys) != len(values):
            raise ValueError("keys and values have different length.")
        return dict(zip(keys, values))
    return {keys: values}


def list_to_str(nums, precision=2):
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))


def get_num_args(func):
    """Get the number of arguments of a Python function.

    References:

        - https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function
    """
    if sys.version_info[0] == 2:
        return len(inspect.getargspec(func).args)
    sig = inspect.signature(func)
    return len(sig.parameters)


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
