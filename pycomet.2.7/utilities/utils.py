import os
import gc
import sys
import h5py
import time
import signal
import random
import inspect
import datetime
import numpy as np
import tensorflow as tf

# from matplotlib import animation
from functools import wraps
from pathlib import PurePath
from pycomet import config as cfg
from tensorflow.python.keras import backend


# Printing functions
###############################################################################
def print_main(text, start='\n', verbose=True, *args, **kwargs):
    if verbose:
        print(start + cfg.HEADERS['main'] + text, *args, **kwargs)

def print_submain(text, start='', verbose=True, *args, **kwargs):
    if verbose:
        print(start + cfg.HEADERS['main'] + '  ' + text, *args, **kwargs)

def input_main(text, start='', *args, **kwargs):
    return input(start + cfg.HEADERS['main'] + text, *args, **kwargs)

def input_submain(text, start='', *args, **kwargs):
    return input(start + cfg.HEADERS['main'] + '  ' + text, *args, **kwargs)

def warning(text, start='', *args, **kwargs):
    print(start + cfg.HEADERS['warning'] + text, *args, **kwargs)

def raise_value_err(text):
    message = ''.join([cfg.HEADERS['err'], text])
    raise ValueError(message)

def raise_module_not_found_error(text):
    message = ''.join([cfg.HEADERS['err'], text])
    raise ModuleNotFoundError(message)

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

# Handling data types
###############################################################################
def get_class(module, name=None, kwargs=None):
    """Return a class object given its name and the module which belongs to."""
    if name is None and 'name' in kwargs.keys():
        name = kwargs.pop('name')
    for name_i, cls_i in inspect.getmembers(module, inspect.isclass):
        if name_i == name:
            return cls_i(**kwargs) if kwargs else cls_i
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

def get_args_values(modules, args):
    """Get the number of arguments of a Python function."""
    if not isinstance(modules, list):
        modules = [modules]
    _args = {}
    for arg in args:
        for module in modules:
            if hasattr(module, arg):
                _args.update({arg: getattr(module, arg)})
                break
    return _args

def depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0

def value_to_dict(value, keys):
    return value if isinstance(value, dict) else {k: value for k in keys}

def to_list(x):
    return list(x) if isinstance(x, (list,tuple)) else [x]

def search_string(file, string):
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if string+' ' in line:
                return i, line

def check_format(data_type, string):
    if data_type == float and 'd' in string:
        return string.replace("d", "e")
    else:
        return string

def check_file_extension(file, extension):
    os.path.isfile(file)
    if PurePath(file).suffix != extension:
        raise_value_err(
            "The file '{}' has to be a '{}' one.".format(
                os.path.normpath(file), extension
            )
        )

def zip_dict_of_lists(d):
    keys = d.keys()
    return tuple(dict(zip(keys, v)) for v in zip(*(d[k] for k in keys)))


def convert_struct_to_tf(x):
    return tf.nest.map_structure(
        lambda y: tf.convert_to_tensor(y, dtype=backend.floatx()), x
    )

def get_key_from_value(d, v):
    for key, value in d.items():
        if v == value:
            return key
    raise_value_err(f"A key for value `{v}` doesn't exist in the dictionary.")

def reshape_to_2DArray(x):
    if x is not None:
        x = tf.convert_to_tensor(x, dtype=backend.floatx())
        if len(x.shape) == 0:
            return tf.reshape(x, [1,1])
        elif len(x.shape) == 1:
            return tf.reshape(x, [-1,1])
        elif len(x.shape) == 2:
            return x
        elif len(x.shape) == 3:
            return tf.squeeze(x, axis=0)


def replace(x, y, indices, axis=0):
    # Preprocess inputs
    x, y = tf.nest.map_structure(lambda x: np.array(x), [x,y])
    indices = tf.nest.flatten(indices)
    if axis not in (0,1):
        raise_value_err(f"The 'axis' input as to be 0 or 1.")
    # Replace 1D vectors
    if axis == 1:
        x, y = x.T, y.T
    for i in indices:
        x[i] = y[i]
    if axis == 1:
        x, y = x.T, y.T
    return x

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

class H5DataSet:
    def __init__(self, path, no_group=False):
        self.data = {}
        self.no_group = no_group
        self.df = h5py.File(path,'r')
        self.df.visititems(self)
        self.df.close()

    def __call__(self, name, node):
        if isinstance(node, h5py.Dataset):
            if self.no_group:
                name_ = os.path.join(*(name.split(os.path.sep)[1:]))
            else:
                name_ = name
            self.data[name_] = self.df[name][()]


# Environment setup -----------------------------------------------------------
def environment():
    num_threads = min([
        os.cpu_count(),
        tf.config.threading.get_inter_op_parallelism_threads(),
        tf.config.threading.get_intra_op_parallelism_threads()
    ])
    print(cfg.HEADERS['main'] + "Environment setup\n" + "-"*98)
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

def get_struct_shape(structure=None, axis=None):
    if structure is not None:
        if axis:
            fn = lambda x: x.shape[axis] if x is not None else None
        else:
            fn = lambda x: x.shape if x is not None else None
        return tf.nest.map_structure(fn, structure)


def unpack_x_y_sample_weight_index(data):
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None, None, None)
    elif len(data) == 1:
        return (data[0], None, None, None)
    elif len(data) == 2:
        return (data[0], data[1], None, None)
    elif len(data) == 3:
        return (data[0], data[1], data[2], None)
    elif len(data) == 4:
        return (data[0], data[1], data[2], data[3])
    else:
        error_msg = (
            "Data is expected to be in format `x`, `(x,)`, "
            "`(x, y)`, `(x, y, sample_weight)`, or "
            f"`(x, y, sample_weight, sample_index)`, found: {data}"
        )
        raise raise_value_err(error_msg)

def set_seeds(seed=None):
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

def set_num_threads(n):
    tf.config.threading.set_inter_op_parallelism_threads(n)
    tf.config.threading.set_intra_op_parallelism_threads(n)



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

