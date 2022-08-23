import sys
import inspect
import importlib
import tensorflow as tf

from pycomet.utilities import stdout as pcm_stdout


# Modules ---------------------------------------------------------------------
def reload_mod(mod_names=[]):
    mod_list = []
    for name in tf.nest.flatten(mod_names):
        if name in sys.modules.keys():
            # Reload module
            importlib.reload(sys.modules[name])
            # Reload submodules
            for importer, submod, ispkg in pkgutil.iter_modules(
                sys.modules[name].__path__
            ):
                if not submod.startswith('__'):
                    importlib.reload(sys.modules[name+'.'+submod])
            mod_list.append(sys.modules[name])
        else:
            mod_list.append(importlib.import_module(name))
    return mod_list

def load_mod_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Classes ---------------------------------------------------------------------
def get_class(module, name=None, kwargs=None):
    """Return a class object given its name and the module which belongs to."""
    if name is None and kwargs is not None:
        if 'name' in kwargs.keys():
            name = kwargs.pop('name')
        else:
            pcm_stdout.raise_value_err("Class name not provided.")
    for name_i, cls_i in inspect.getmembers(module, inspect.isclass):
        if name_i == name:
            return cls_i(**kwargs) if kwargs is not None else cls_i
    pcm_stdout.raise_value_err(
        "Class `{}` not found in module `{}`.".format(name, module.__name__)
    )

def get_class_name(cls):
    """Return the class name."""
    return cls.__class__.__name__

# Functions -------------------------------------------------------------------
def get_fun_args(obj):
    """Get the arguments of a Python function."""
    if inspect.isclass(obj):
        args = inspect.getfullargspec(obj).args
    else:
        args = list(dict(inspect.signature(obj).parameters).keys())
    for arg in ('self', 'args', 'kwargs'):
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
