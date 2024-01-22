import h5py
import keras
import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from tensorflow.python.util.tf_export import keras_export

from silx.io.dictdump import h5todict


@keras_export('keras.utils.GraphSequence')
class GraphSequence(keras.utils.Sequence):

  # Initialization
  # ===================================
  def __init__(
    self,
    data=None,
    use_static_edges=False,
    batch_size=32,
    shuffle=True
  ):
    super(GraphSequence, self).__init__()
    self.name = utils.get_class_name(self)
    self.use_static_edges = use_static_edges
    # Load data
    self.check_file_ext(data)
    self.data = self.read_data(data)
    # Batching
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.steps = int(np.ceil(self.nb_samples/self.batch_size))

  def check_file_ext(self, data):
    _raise_err = False
    if (not isinstance(data, str)):
      _raise_err = True
    elif (not data.lower().endswith(('.h5', '.hdf5'))):
      _raise_err = True
    if _raise_err:
      utils.raise_err(
        ValueError,
        f"The input data for '{self.name}' must be a string " \
        "pointing to an HDF5 file containing the graph data."
      )
    return data

  def read_data(self, data):
    # Load data
    data = h5todict(h5file=data)
    if self.use_static_edges:
      keys = ("node_attr", "target")
    else:
      keys = ("node_attr", "edge_attr", "target")
    if any([(k not in data) for k in keys]):
      utils.raise_err(
        ValueError,
        f"The input data for '{self.name}' must " \
        f"contain the following keys: {keys}."
      )
    # Check rank
    for k in ("node_attr", "edge_attr", "target"):
      if (k in data):
        if (data[k].ndim != 3):
          utils.raise_err(
            ValueError,
            f"The data for '{k}' should be a 3-dimensional tensor."
          )
    # Dimensions
    nshape = data["node_attr"].shape
    self.nb_samples, self.nb_nodes, self.nb_node_feat = nshape
    if ("edge_attr" in data):
      eshape = data["edge_attr"].shape
      if (nshape[0] != eshape[0]):
        utils.raise_err(
          ValueError,
          "Different number of samples between 'node_attr' and " \
          "'edge_attr'. Please check the first dimension of both tensors."
        )
        self.nb_edges, self.nb_edge_feat = eshape[1:]
    else:
      self.nb_edges, self.nb_edge_feat = None, None
    # Check dimension between input and output
    tshape = data["target"].shape
    if (tshape[:2] != nshape[:2]):
      utils.raise_err(
        ValueError,
        "This data class is designed exclusively for node prediction. " \
        "A mismatch was detected in the first two dimensions between " \
        "the 'node_attr' and the 'target' tensor. They should be the same."
      )
    self.nb_target_feat = tshape[-1]
    # Compose data as (x, y, sample_weight)
    x = []
    for k in ("node_attr", "edge_attr"):
      if (k in data):
        x.append(data[k])
    _data = [x, [data["target"]]]
    if ("sample_weight" in data):
      _data.append([data["sample_weight"]])
    return tf.nest.map_structure(ops.to_tensor, _data)

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
            f"'{obj}' data passed into '{self.name}' are " \
            "inconsistent with the network output dimensions."
          )
    self.data = ops.remove_none_from_struct(self.data)

  # Sequencing
  # ===================================
  def __len__(self):
    return self.steps

  # Handling batches during each epoch
  # -----------------------------------
  def __getitem__(self, index):
    return self.batches[index]

  # Handling data during training
  # -----------------------------------
  def on_epoch_end(self):
    batches = {}
    for (obj, dset) in self.data.items():
      if self.shuffle:
        dset = self.shuffle_data(obj, dset)
      batches[obj] = self.batch_data(obj, dset)
    self.sequence = utils.zip_dict_of_lists(batches)


    dset = self.data
    if self.shuffle:
      dset = self.shuffle_data(dset)
    self.batches = self.batch_data(dset)

  def shuffle_data(self, dset):
    indices = ops.random_permutation(self.nb_samples)
    return tf.nest.map_structure(
      lambda x: tf.gather(x, indices, axis=0), dset
    )

  def batch_data(self, dset):
    return tf.nest.map_structure(
      lambda x: ops.tensor_split(x, self.steps), dset
    )


bd = tf.nest.map_structure(lambda x: np.array_split(x, 2), data)
tf.nest.map_structure(np.shape, bd)

  keys = list(dictionary.keys())
  return list(list(zip(v)) for v in zip(bd))
