import keras
import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.utils.MultiObjSequence')
class MultiObjSequence(keras.utils.Sequence):

  # Initialization
  # ===================================
  def __init__(
    self,
    data={},
    objectives=[],
    obj_opt_targets=[],
    batch_size=32,
    shuffle=True
  ):
    super(MultiObjSequence, self).__init__()
    self.name = utils.get_class_name(self)
    # Objective functions identifiers
    # > Example with 'ODE' data: ['ics','pts','res']
    self.objectives = list(objectives)
    self.obj_opt_targets = list(obj_opt_targets)
    # Checking 'batch_size'
    if (not isinstance(batch_size, (dict, int))):
      utils.raise_err(
        ValueError,
        f"'batch_size' passed into `{self.name}` has to be a `int` " \
        "or `dict`. If `dict`, the keys have to be the same as the " \
        "objective functions identifiers (see 'objectives')."
      )
    if isinstance(batch_size, int):
      self.batch_size = {obj: batch_size for obj in self.objectives}
    else:
      self.batch_size = batch_size
    # Checking 'data'
    if (not isinstance(data, dict)):
      utils.raise_err(
        ValueError,
        f"'data' passed into `{self.name}` has to be a `dict` with " \
        "objective functions identifiers (see 'objectives') as keys."
      )
    else:
      self.data, self.nb_samples = {}, {}
      for obj in list(objectives):
        if ((obj not in data) or (data[obj] is None)):
          # Deleting the corresponding `key` entry
          self.objectives.remove(obj)
          if (obj in self.batch_size):
            self.batch_size.pop(obj)
        else:
          if (not isinstance(data[obj], (list,tuple))):
            # Check data type: [x, y, sample_weight]
            utils.raise_err(
              ValueError,
              "Values passed into 'data' dictionary in " \
              f"`{self.name}` must be a `list`/`tuple` of " \
              "array: [x, y, sample_weight]."
            )
          else:
            self.data[obj] = list(utils.unpack_list(data[obj], nb_elem=3))
            self.nb_samples[obj] = ops.get_cardinality(data[obj])
    self.data = tf.nest.map_structure(ops.to_tensor, self.data)
    self.shapes = tf.nest.map_structure(
      lambda x: tf.shape(x) if (x is not None) else None, self.data
    )
    # Number of batches for each objective data
    self.steps = tf.nest.map_structure(
      lambda x, y: int(np.ceil(x/y)), self.nb_samples, self.batch_size
    )
    self.shuffle = shuffle
    self.keep_indices = False

  # Building
  # ===================================
  def build(self, model):
    self.build_targets(model)
    self.on_epoch_end()

  def build_targets(
    self,
    model
  ):
    for obj in self.objectives:
      # Get true values
      y = self.data[obj][1]
      # Get pred values
      inputs = tf.nest.map_structure(
        lambda d: tf.random.uniform((1, d), dtype=bkd.get_floatx()),
        model.input_dim
      )
      y_pred = model.multi_obj_pred(x=inputs, objective=obj, training=False)
      # Squeeze
      y, y_pred = list(map(
        lambda x: utils.squeeze_list(tuple(tf.nest.flatten(x))),
        [y, y_pred]
      ))
      # Get true and pred feature dimensions
      y_dim = ops.get_struct_shape(y, axis=1)
      y_pred_dim = ops.get_struct_shape(y_pred, axis=1)
      # Check if true and pred dimensions are the same
      if (y_dim != y_pred_dim):
        if (obj in self.obj_opt_targets):
          self.data[obj][1] = tf.nest.map_structure(
            lambda d: tf.zeros((self.nb_samples[obj], d)),
            y_pred_dim
          )
        else:
          utils.raise_err(
            ValueError,
            f"'{obj}' data passed into `{self.name}` are " \
            "inconsistent with the network output dimensions."
          )
    self.data = ops.remove_none_from_struct(self.data)

  def init_sample_weight(
    self,
    initializer=None,
    trainable=False,
    rank=2
  ):
    for obj in self.objectives:
      # Get initializers
      init_fn = initializer[obj] if initializer else tf.ones
      # Define shape
      if (rank == 1):
        fun = lambda x: init_fn(shape=(x.shape[0],1), dtype=bkd.get_floatx())
      else:
        fun = lambda x: init_fn(shape=x.shape, dtype=bkd.get_floatx())
      # Construct weights
      sample_weight = tf.nest.map_structure(fun, self.data[obj][1])
      # Make them tf.Variable if trainable
      if trainable:
        sample_weight =  tf.nest.map_structure(
          lambda w: tf.Variable(w), sample_weight
        )
      # Add sample weights to the data set
      self.data[obj][2] = sample_weight

  # Sequencing
  # ===================================
  def __len__(self):
    return max(list(self.steps.values()))

  # Handling batches during each epoch
  # -----------------------------------
  def __getitem__(self, index):
    return utils.zip_dict_of_lists(self.sequence[index])

  # Handling data during training
  # -----------------------------------
  def on_epoch_end(self):
    batches = {}
    for (obj, dset) in self.data.items():
      if self.shuffle:
        dset = self.shuffle_data(obj, dset)
      batches[obj] = self.batch_data(obj, dset)
    self.sequence = utils.zip_dict_of_lists(batches)

  def shuffle_data(self, obj, dset):
    indices = ops.random_permutation(self.nb_samples[obj])
    dset = tf.nest.map_structure(
      lambda z: tf.gather(z, indices, axis=0), dset
    )
    if self.keep_indices:
      dset.append(tf.reshape(indices, (-1,1)))
    return dset

  def batch_data(self, obj, dset):
    repeat = int(np.ceil(self.__len__()/self.steps[obj]))
    dset = tf.nest.map_structure(
      lambda z: ops.tensor_split(z, self.steps[obj]) * repeat, dset
    )
    for (i, z) in enumerate(dset):
      if (self.keep_indices and (i == len(dset)-1)):
        break
      else:
        if isinstance(self.shapes[obj][i], list):
          dset[i] = list(zip(*z))
    dset = list(zip(*dset))
    return dset
