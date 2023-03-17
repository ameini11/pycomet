import os
import sys
import importlib
import tensorflow as tf

from pycomet import config as cfg


# Packages --------------------------------------------------------------------
def import_matplotlib():
    import matplotlib
    from matplotlib import pyplot
    pyplot.style.use(cfg._MPL['style'])
    return matplotlib, pyplot

def add_package_path(packages={}):
    for name, path in packages.items():
        if importlib.util.find_spec(name) is None:
            try:
                sys.path.append(os.path.abspath(path))
            except:
                continue

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
