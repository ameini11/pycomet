import os
import signal
import inspect
import tensorflow as tf

from pycomet import utils


# Classes
# =====================================
def get_class(
  modules,
  name=None,
  kwargs=None
):
  """
  Return a class object given its name and the module it belongs to.

  :param modules: A module or a list of modules to search for the class.
  :type modules: list or module

  :param name: The name of the class to retrieve.
  :type name: str, optional

  :param kwargs: Optional keyword arguments to pass when initializing the
                 class (it can contain the name of the class if 'name'
                 is not provided).
  :type kwargs: dict, optional

  :return: An instance of the class if found, or the class itself.
  :rtype: object or class

  This function searches for a class with the specified name within the given
  module(s). If found, it can return an instance of the class with optional
  keyword arguments provided in `kwargs`. If no class is found, an error is
  raised.
  """
  # Check class name
  if ((name is None) and (kwargs is not None)):
    if ("name" in kwargs.keys()):
      name = kwargs.pop("name")
    else:
      utils.raise_err(
        error=ValueError,
        text="Class name not provided."
      )
  # Loop over modules to find class
  modules = tf.nest.flatten(modules)
  for module in modules:
    members = inspect.getmembers(module, inspect.isclass)
    for (name_i, cls_i) in members:
      if (name_i == name):
        if (kwargs is not None):
          return cls_i(**kwargs)
        else:
          return cls_i
  # Raise error if class not found
  names = [module.__name__ for module in modules]
  utils.raise_err(
    error=ValueError,
    text=f"Class `{name}` not found in modules: {names}."
  )

def get_class_name(
  cls
):
  """
  Return the name of a Python class.

  :param cls: The Python class to get the name for.
  :type cls: class
  """
  return cls.__class__.__name__

# Functions
# =====================================
def get_fun_args(
  obj,
  rm_args=('self', 'args', 'kwargs')
):
  """
  Retrieve the arguments of a Python function.

  :param obj: The Python function or method to inspect.
  :type obj: function or method

  :param rm_args: Arguments to be removed from the result
                  (default: ('self', 'args', 'kwargs')).
  :type rm_args: tuple

  :return: A list of the function's arguments with specified arguments removed.
  :rtype: list

  This function retrieves the arguments of a Python function or method. If
  `obj` is a class, it gets the constructor's arguments. You can specify
  arguments to be removed using `rm_args`, which defaults to ('self', 'args',
  'kwargs').
  """
  if inspect.isclass(obj):
    args = inspect.getfullargspec(obj).args
  else:
    args = list(dict(inspect.signature(obj).parameters).keys())
  for arg in rm_args:
    if (arg in args):
      args.remove(arg)
  return args

def get_fun_args_values(
  modules,
  args
):
  """
  Retrieve argument values of Python functions from a list of modules.

  :param modules: A list of Python modules to search for argument values.
  :type modules: list[module]

  :param args: A list of argument names to search for within the modules.
  :type args: list[str]

  :return: A dictionary mapping argument names to their values.
  :rtype: dict

  This function iterates through the specified modules to find and collect the
  values of the given arguments. It returns a dictionary where the argument
  names are keys and their corresponding values are the associated values.

  If an argument is not found in any of the modules, it won't be included in
  the result.
  """
  if (not isinstance(modules, list)):
    modules = [modules]
  found_args = {}
  for arg in args:
    for module in modules:
      if hasattr(module, arg):
        found_args.update({arg: getattr(module, arg)})
        break
  return found_args

# Processes
# =====================================
def kill_process():
  """
  Terminate the current Python process by sending a SIGTERM signal.

  This function sends a termination signal (SIGTERM) to the current Python
  process, resulting in its graceful termination. It is particularly useful
  for stopping the script's execution when needed.

  Please note that calling this function will end the script abruptly, so
  ensure any necessary cleanup or finalization steps are handled before
  invoking it.
  """
  os.kill(os.getpid(), signal.SIGTERM)


class DefObject(object):

  def __init__(self):
    self.__name__ = get_class_name(self)