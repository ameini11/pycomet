import os
import sys
import importlib

def add_package_path(packages={}):
    for name, path in packages.items():
        if importlib.util.find_spec(name) is None:
            sys.path.append(os.path.abspath(path))