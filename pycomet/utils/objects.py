import inspect
import tensorflow as tf

from pycomet import utils


# Classes ---------------------------------------------------------------------
def get_class(modules, name=None, kwargs=None):
    """Return a class object given its name and the module which belongs to."""
    if name is None and kwargs is not None:
        if 'name' in kwargs.keys():
            name = kwargs.pop('name')
        else:
            utils.raise_value_err("Class name not provided.")
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for module in modules:
        for name_i, cls_i in inspect.getmembers(module, inspect.isclass):
            if name_i == name:
                return cls_i(**kwargs) if kwargs is not None else cls_i
    names = [module.__name__ for module in modules]
    utils.raise_value_err(f"Class `{name}` not found in modules: {names}.")

def get_class_name(cls):
    """Return the class name."""
    return cls.__class__.__name__

# Functions -------------------------------------------------------------------
def get_fun_args(obj, rm_args=('self', 'args', 'kwargs')):
    """Get the arguments of a Python function."""
    if inspect.isclass(obj):
        args = inspect.getfullargspec(obj).args
    else:
        args = list(dict(inspect.signature(obj).parameters).keys())
    for arg in rm_args:
        if arg in args:
            args.remove(arg)
    return args

def get_fun_args_values(modules, args):
    """Get arguments values of a Python function."""
    if not isinstance(modules, list):
        modules = [modules]
    _args = {}
    for arg in args:
        for module in modules:
            if hasattr(module, arg):
                _args.update({arg: getattr(module, arg)})
                break
    return _args
