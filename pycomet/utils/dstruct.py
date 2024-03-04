import os
import gc
import h5py
import numpy as np
import tensorflow as tf

from pycomet import utils


# Dictionaries
# =====================================
def make_dict(
  keys,
  values
):
  """
  Convert two lists or two variables into a dictionary.

  This function takes two sequences, `keys` and `values`, and transforms them
  into a dictionary.

  :param keys: A list, tuple, or a single key.
  :type keys: list, tuple, or any comparable type

  :param values: A list, tuple, or a single value.
  :type values: list, tuple, or any value to be associated with `keys`

  :return: A dictionary containing the converted key-value pairs.
  :rtype: dict

  If both `keys` and `values` are sequences, they must have the same length,
  and the function returns a dictionary created by pairing corresponding
  elements. If only one key and one value are provided, the function creates a
  dictionary with the key-value pair.
  """
  if isinstance(keys, (list, tuple)):
    if (len(keys) != len(values)):
      utils.raise_err(
        error=ValueError,
        text="Keys and values have different length."
      )
    return dict(zip(keys, values))
  return {keys: values}

def compute_dict_depth(
  dictionary
):
  """
  Compute the depth of a dictionary.

  :param dictionary: The dictionary for which to determine depth.
  :type dictionary: dict

  :return: The depth of the dictionary.
  :rtype: int

  This function calculates the depth of a nested dictionary, indicating how
  many levels of nested dictionaries exist. If the input is not a dictionary,
  the depth is considered to be 0. An empty dictionary has a depth of 1.
  """
  if isinstance(dictionary, dict):
    return 1 + (
      max(map(compute_dict_depth, dictionary.values())) if dictionary else 0
    )
  return 0

def value_to_dict(
  value,
  keys
):
  """
  Create a dictionary by pairing a single value with multiple keys.

  :param value: The value to pair with each key.
  :type value: any

  :param keys: A list of keys to create dictionary entries.
  :type keys: list

  :return: A dictionary with each key mapped to the specified value.
  :rtype: dict

  This function takes a single value and a list of keys, and it pairs
  the value with each key to create a dictionary. If the input value
  is already a dictionary, it is returned as is.
  """
  return value if isinstance(value, dict) else {k: value for k in keys}

def zip_dict_of_lists(
  dictionary
):
  """
  Zip together a dictionary containing lists.

  :param dictionary: A dictionary where the values are lists of equal length.
  :type dictionary: dict

  :return: A tuple of dictionaries with zipped key-value pairs.
  :rtype: tuple[dict]

  This function takes a dictionary where each value is a list of equal
  length. It zips these lists together element-wise and returns a tuple
  of dictionaries where each dictionary contains zipped key-value pairs.
  """
  keys = list(dictionary.keys())
  return tuple(
    dict(zip(keys, v)) for v in zip(*(dictionary[k] for k in keys))
  )

def get_key_from_value(
  dictionary,
  value
):
  """
  Get the key associated with a given value in a dictionary.

  :param dictionary: The dictionary to search in.
  :type dictionary: dict

  :param value: The value to find the key for.
  :type value: any

  :return: The key corresponding to the provided value in the dictionary.
  :rtype: any

  This function iterates through the dictionary to find and return the key
  associated with the specified value. If the value is not found, it raises a
  ValueError with an error message indicating that the key for the given value
  doesn't exist in the dictionary.
  """
  for (k, v) in dictionary.items():
    if (value == v):
      return k
  utils.raise_err(
    error=ValueError,
    text=f"A key for value `{value}` doesn't exist in the dictionary."
  )

def get_keys_from_dict(
  dictionary=None
):
  """
  Recursively retrieve keys from a dictionary and its nested dictionaries.

  :param dictionary: The dictionary to extract keys from.
  :type dictionary: dict

  :return: A list of keys including keys from nested dictionaries, represented
           as a list of strings with parent keys separated by '/'.
  :rtype: list of str

  This function traverses the input dictionary and its nested dictionaries to
  extract all the keys. It returns a list of keys, where keys from nested
  dictionaries are represented as strings with parent keys separated by '/'.
  """
  result = []
  for (key, value) in dictionary.items():
    if isinstance(value, dict):
      new_keys = get_keys_from_dict(value)
      result.append(key)
      for innerkey in new_keys:
        result.append(f'{key}/{innerkey}')
    else:
      result.append(key)
  return result

def check_dicts_consistency(
  list_of_dicts=[],
  idx_ref_dict=0
):
  """
  Check consistency among multiple dictionaries.

  :param list_of_dicts: A list of dictionaries to compare for consistency.
  :type list_of_dicts: list[dict]

  :param idx_ref_dict: Index of the reference dictionary to compare against.
  :type idx_ref_dict: int

  This function checks the consistency among a list of dictionaries. It does
  so by comparing the keys of each dictionary to those of a reference
  dictionary, specified by the `idx_ref_dict` parameter.

  If any key in a non-reference dictionary does not exist in the reference
  dictionary, it raises a `ValueError` with an error message indicating the
  inconsistency.
  """
  # print(list_of_dicts)
  # list_of_keys = tf.nest.map_structure(
  #   lambda x: (x[0], get_keys_from_dict(x[1])), list_of_dicts
  # )
  # print(list_of_dicts)
  list_of_keys = []
  for (iname, idict) in list_of_dicts:
    if (idict is not None):
      list_of_keys.append((iname, get_keys_from_dict(idict)))
  for i, (name, keys) in enumerate(list_of_keys):
    if (i == idx_ref_dict):
      continue
    else:
      for k in keys[idx_ref_dict]:
        if (k not in keys[i]):
          utils.raise_err(
            error=ValueError,
            text=f"A structure discrepancy between '{name}' and " \
              f"'{list_of_dicts[idx_ref_dict][0]}' dictionaries " \
              "has been found."
          )

# HDF5 files
# =====================================
class H5DataSet(object):
  """
  Read datasets from HDF5 files.

  This class reads datasets from HDF5 files and allows for the option to
  retain the group path structure or remove it.

  :param path: The path to the HDF5 file.
  :type path: str

  :param no_group: Set to True to remove the group path structure when storing
                   datasets.
  :type no_group: bool

  Upon initialization, this class reads the specified HDF5 file, extracts the
  datasets, and stores them in a dictionary. The 'no_group' parameter can be
  used to control whether the group path structure should be retained in the
  stored dataset names or removed.

  Example usage:
  >>> h5_data = H5DataSet('data.h5', no_group=True)
  >>> dataset = h5_data.data['dataset_name']
  """
  def __init__(
    self,
    path,
    no_group=False
  ):
    self.data = {}
    self.no_group = no_group
    self.df = h5py.File(path,'r')
    self.df.visititems(self)
    self.df.close()

  def __call__(
    self,
    name,
    node
  ):
    if isinstance(node, h5py.Dataset):
      if self.no_group:
        name_ = os.path.join(*(name.split(os.path.sep)[1:]))
      else:
        name_ = name
      self.data[name_] = self.df[name][()]

def close_all_hdf5():
  """
  Close all opened HDF5 files.

  This function iterates through all objects in memory, and if it finds any
  open HDF5 files, it attempts to close them. Any exceptions that occur during
  the closing process are gracefully handled.

  It's essential to use this function when you want to ensure that all opened
  HDF5 files are properly closed to prevent resource leaks.
  """
  # Browse through all objects
  for obj in gc.get_objects():
    # Just HDF5 files
    if isinstance(obj, h5py.File):
      try:
        obj.close()
      except:
        pass

# Text files
# =====================================
def replace(
  file,
  patterns,
  replaces,
  use_startswith=True
):
  """
  Replace patterns/lines within a file with updated replacements.

  :param file: The path to the file in which replacements should be made.
  :type file: str

  :param patterns: A list of patterns or lines to search for in the file.
  :type patterns: list[str]

  :param replaces: A list of replacement lines corresponding to patterns.
  :type replaces: list[str]

  :param use_startswith: If True, match patterns at the beginning of lines;
                         otherwise, match anywhere in lines.
  :type use_startswith: bool

  This function reads a file, searches for lines that match the specified
  patterns (based on the 'use_startswith' parameter), and replaces them with
  the corresponding lines from the 'replaces' list. The modified content is
  then saved back to the same file.
  """
  file_old = open(file, 'r')
  file_new = open(file+'_new', 'w')
  for line in file_old:
    if use_startswith:
      cond = any([line.startswith(p) for p in patterns])
    else:
      cond = any([p in line for p in patterns])
    if cond:
      for (i, p) in enumerate(patterns):
        if (p in line):
          line_new = replaces[i]
          break
    else:
      line_new = line
    file_new.write(line_new)
  file_new.close()
  file_old.close()
  os.rename(file+'_new', file)

def generate_rnd_name(
  ref_name,
  rnd_num=1.e6
):
  """
  Generate a random name based on a reference name and an upper limit.

  :param ref_name: The reference name to build upon.
  :type ref_name: str

  :param rnd_num: An upper limit for generating random numbers.
  :type rnd_num: int or float

  This function generates a random name by appending an underscore 
  and a random integer, obtained by using `np.random.randint`, 
  to the provided reference name.

  The `rnd_num` parameter specifies the upper limit for generating the random
  integer, which is truncated to an integer if it's a float. The result is
  returned as a new string.
  """
  text = os.path.splitext(ref_name)
  rnd_num = str(np.random.randint(int(rnd_num)))
  return text[0] + '_' + rnd_num + text[1]


def check_file_extension(
  file,
  extension
):
  """
  Check if the file at the given path has the specified file extension.

  :param file: The path to the file to check.
  :type file: str

  :param extension: The expected file extension, including the dot
                    (e.g., '.txt').
  :type extension: str

  This function checks if the file located at the provided path has the
  specified file extension. If the file does not have the expected extension,
  it raises a ValueError with a descriptive error message.
  """
  file = os.path.abspath(file)
  if os.path.isfile(file):
    if (os.path.splitext(file)[-1] != extension):
      utils.raise_err(
        error=ValueError,
        text=f"The file '{file}' has to be a '{extension}' one."
      )

def search_string(
  file,
  string
):
  """
  Search for the given string within the contents of a file.

  :param file: The path to the file to search within.
  :type file: str

  :param string: The string to search for within the file's contents.
  :type string: str

  :return: A tuple containing the line number and the line where the
           string was found, or None if not found.
  :rtype: tuple[int, str] or None
  """
  with open(file, 'r') as f:
    for (i, line) in enumerate(f):
      if (string +' ' in line):
        return i, line

# Lists
# =====================================
def unpack_list(data, nb_elem=3):
  if isinstance(data, tuple):
    data = list(data)
  if (not isinstance(data, list)):
    return ([data] + [None]*(nb_elem-1))
  elif (len(data) <= nb_elem):
    return (data + [None]*(nb_elem-len(data)))
  else:
    utils.raise_err(
      ValueError,
      "The number of elements in the container " \
      "has to be less or equal than 'nb_elem'."
    )

def squeeze_list(iterable):
  """
  Squeeze a list by removing its outer list struct.

  :param iterable: An iterable (e.g., list, tuple) to be squeezed.
  :type iterable: list, tuple, or any iterable

  :return: The squeezed iterable if it contains only one element;
       otherwise, the original iterable is returned.
  :rtype: any

  This function takes an iterable and removes its outer list struct if it
  contains only one element. If the iterable contains multiple elements or is
  empty, it returns the original iterable.
  """
  iterable = list(iterable)
  if (len(iterable) == 1):
    return iterable[-1]
  else:
    return iterable

# Miscellanea
# =====================================
def num_to_str(
  numbers,
  precision=2
):
  """
  Convert a list/tuple/array of numbers into a string.

  :param numbers: A list, tuple, or array of numeric values.
  :type numbers: list, tuple, or numpy.ndarray

  :param precision: The number of decimal places to include (default is 2).
  :type precision: int

  :return: A string representation of the input numbers.
  :rtype: str

  This function takes a list, tuple, or numpy array of numeric values and
  converts them into a string. It allows customizing the precision with
  which the numbers are displayed. If 'numbers' is a single numeric value,
  it will be formatted with the specified precision. If 'numbers' is a
  sequence, the function creates a string representation in square brackets
  with elements separated by commas, each formatted with the specified
  precision.

  If 'numbers' is None, an empty string is returned.
  """
  if (numbers is None):
    return ""
  if not isinstance(numbers, (list, tuple, np.ndarray)):
    return "{:.{}e}".format(numbers, precision)
  else:
    return "[{:s}]".format(
      ", ".join(["{:.{}e}".format(x, precision) for x in numbers])
    )
