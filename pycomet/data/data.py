import os
import abc
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops


class Data(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    data_src={},
    validation_split=None,
    training=True,
    batch_size=None,
    from_file=False,
    header=0,
    columns=None,
    preproc_fn=None
  ):
    self.name = utils.get_class_name(self)
    # Globals
    self.dim = dim
    self.training = training
    self.built = False
    self.multi_obj_loss = False
    self.data_gen = None
    self.physical_mdl = None
    # Data
    self.identifiers = ("train", "valid", "test")
    self.data_src = data_src
    self.train = None
    self.valid = None
    self.test = None
    self.validation_split = validation_split
    # Check data sources
    self.check_data_src()
    # Batch sizes
    self.batch_size = {
      "train": 32,
      "valid": None
    }
    if (batch_size is not None):
      self.set_batch_size(batch_size)
    # Extra attributes if data extracted from file
    self.from_file = from_file
    self.header = header
    if self.from_file:
      self.columns = columns
      self.preproc_fn = preproc_fn
      if (columns is None):
        utils.raise_err(
          ValueError,
          f"'columns' passed into '{self.name}' has to be a " \
          "dictionary (structured as 'data_src') with last-level " \
          "keys ['x', 'y', 'sample_weight']."
        )
      elif any([(k in columns) for k in ("x", "y", "sample_weight")]):
        self.columns = tf.nest.map_structure(
          lambda _: columns, self.data_src
        )
      else:
        self.columns = columns
      # Check array preprocessing functions
      if (preproc_fn is not None):
        if callable(preproc_fn):
          self.preproc_fn = tf.nest.map_structure(
            lambda _: preproc_fn, self.data_src
          )
        elif isinstance(preproc_fn, dict):
          self.preproc_fn = preproc_fn
        else:
          utils.raise_err(
            ValueError,
            f"'preproc_fn' passed into '{self.name}' has to " \
            "be a dictionary or a callable function."
          )
      # Check consistency between dictionaries
      list_of_dicts = [
        ("data_src", self.data_src),
        ("columns", self.columns),
        ("preproc_fn", self.preproc_fn)
      ]
      utils.check_dicts_consistency(list_of_dicts)
    # Testing data
    if ("test" in self.data_src):
      test = self.data_src["test"]
      if (self.from_file and os.path.isdir(test)):
        self.test = {}
        for file in os.listdir(test):
          self.test[Path(file).stem] = self.get_data_from_src(
            data=os.path.join(test, file), identifier="test"
          )
      elif isinstance(test, (list,tuple,dict)):
        self.test = tf.nest.map_structure(lambda x: np.array(x), test)
        if (not isinstance(self.test, dict)):
          self.test = {
            "case_"+str(i).zfill(4): d for (i, d) in enumerate(self.test)
          }
      else:
        utils.raise_err(
          ValueError,
          f"Input 'data_src['test']' passed into '{self.name}' has " \
          "to be a 'list'/'tuple'/'dict' of arrays or a 'str' " \
          "pointing at the folder containting all the test cases " \
          "saved in '.csv' format."
        )

  # Building data
  # ===================================
  def build(self, model=None):
    self.built = True

  # Handling data
  # ===================================
  def check_data_src(self):
    c1 = (not isinstance(self.data_src, dict))
    c2 = any([(k not in self.identifiers) for k in self.data_src.keys()])
    if (c1 or c2):
      utils.raise_err(
        ValueError,
        "'data_src' has to be a 'dict' with keys ['train', 'valid', 'test']."
      )
    c3 = ("train" not in self.data_src)
    if (self.training and (c3 or (self.data_src["train"] is None))):
      utils.raise_err(
        ValueError,
        "No 'train' data passed into 'data_src'."
      )

  def get_data_from_src(self, data, identifier):
    kwargs = {}
    if self.from_file:
      for k in ("columns", "preproc_fn"):
        attr = getattr(self, k)
        if ((data is not None) and (attr is not None)):
          kwargs[k] = attr[identifier]
    data = self.extract_data(data=data, **kwargs)
    if (identifier != "test"):
      data = tf.nest.map_structure(ops.to_tensor, data)
    return data

  # Extract data from container
  def extract_data(self, data, columns=None, preproc_fn=None):
    if (data is not None):
      if self.from_file:
        return self._read_datafile(data, columns, preproc_fn)
      else:
        if isinstance(data, (list,tuple)):
          self._check_data_consistency(data)
          return tuple(data)
        else:
          utils.raise_err(
            ValueError,
            f"Data passed into '{self.name}' has to be a list " \
            "or a tuple of arrays: ('x', 'y', 'sample_weight')"
          )

  # Reading data from file
  def _read_datafile(self, file, columns, preproc_fn=None):
    utils.check_file_extension(file, ".csv")
    # Read csv file
    df = pd.read_csv(file, sep=",", header=self.header)
    if callable(preproc_fn):
      df = pd.DataFrame(
        data=preproc_fn(df.to_numpy(dtype=bkd.get_floatx())),
        columns=df.columns
      )
    # Collect data from file
    data = []
    for i in ("x", "y", "sample_weight"):
      if ((i in ("x", "y")) and (i not in columns.keys())):
        utils.raise_err(
          ValueError,
          f"'{i}' has not been provided in 'columns' dictionary."
        )
      if (i in columns.keys()):
        data.append(ops.extract_data_from_tensor(df, columns[i]))
    return tuple(data)

  @abc.abstractmethod
  def sample_data(self, identifier="train", max_samples=None):
    pass

  # Checking data - Network consistency
  # ===================================
  def _check_data_consistency(self, data):
    net_dim = list(self.dim.values())
    data_dim = ops.get_struct_shape(data, axis=1)
    for (i, io) in enumerate(data_dim):
      if ((i <= 2) and (data_dim[i])):
        if (tf.nest.flatten(data_dim[i]) != tf.nest.flatten(net_dim[i])):
          utils.raise_err(
            ValueError,
            "Data and network dimensions not " \
            "consistent: check 'dim' or data shape."
          )

  # Updating default batch sizes
  # ===================================
  def set_batch_size(self, batch_size):
    if isinstance(batch_size, dict):
      if all([k in batch_size.keys() for k in ("train", "valid")]):
        self.batch_size = batch_size
      else:
        utils.raise_err(
          ValueError,
          f"'batch_size' dictionary passed into '{self.name}' " \
          "has to have ['train', 'valid'] as keys."
        )
    elif isinstance(batch_size, int):
      for k in self.batch_size.keys():
        self.batch_size[k] = batch_size
    else:
      utils.raise_err(
        ValueError,
        f"'batch_size' passed into '{self.name}' " \
        "has to be a 'int' or a 'dict'."
      )

  # Logging
  # ===================================
  def get_nb_pts(self, identifier="training", verbose=1, print_fn=None):
    dset = self.train if (identifier == "training") else self.valid
    if ((dset is not None) and (print_fn is not None)):
      print_fn(dset, identifier, verbose)
    else:
      if verbose:
        utils.print_fn(text=f"No {identifier} pts available.", lev=1)
