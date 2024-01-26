import tensorflow as tf

from pycomet import utils
from .dataset import DataSet
from pycomet import operations as ops
from silx.io.dictdump import h5todict
from tensorflow.python.keras.engine.data_adapter import train_validation_split


class StaticGraphDataSet(DataSet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    data_src={},
    validation_split=None,
    training=True,
    batch_size=None,
    use_static_edges=True,
    use_conditioning=False
  ):
    self.name = utils.get_class_name(self)
    # Globals
    self.dim = dim
    self.training = training
    self.built = False
    self.multi_obj_loss = False
    self.data_gen = None
    self.physical_mdl = None
    self.from_file = True
    # Data
    self.identifiers = ('train', 'valid', 'test')
    self.data_src = data_src
    self.train = None
    self.valid = None
    self.test = None
    self.validation_split = validation_split
    # Check data sources
    self.check_data_src()
    # Batch sizes
    self.batch_size = {
      'train': 32,
      'valid': None
    }
    if (batch_size is not None):
      self.set_batch_size(batch_size)
    # Training/Validation data
    self.use_static_edges = use_static_edges
    self.use_conditioning = use_conditioning
    if self.training:
      if (('valid' in self.data_src) and (self.data_src['valid'] is not None)):
        for k in ('train', 'valid'):
          setattr(self, k, self.get_data_from_src(self.data_src[k]))
      elif self.validation_split:
        self.train, self.valid = train_validation_split(
          self.get_data_from_src(self.data_src['train'], 'train'),
          self.validation_split
        )
      else:
        self.train = self.get_data_from_src(self.data_src['train'], 'train')
    if ('test' in self.data_src):
      utils.warning(
        f"Testing phase is currently not supported for '{self.name}'."
      )
      self.data_src["test"] = None

  # Handling data
  # ===================================
  def get_data_from_src(self, data):
    self._check_file(data)
    return self._read_data(data)

  def _check_file(self, data):
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

  def _read_data(self, data):
    # Load data
    data = h5todict(h5file=data)
    if self.use_static_edges:
      keys = ["node_attr", "target"]
    else:
      keys = ["node_attr", "edge_attr", "target"]
    if self.use_conditioning:
      keys.append("cond_attr")
    if any([(k not in data) for k in keys]):
      utils.raise_err(
        ValueError,
        f"The input data for '{self.name}' must " \
        f"contain the following keys: {keys}."
      )
    # Check rank
    for k in ("node_attr", "edge_attr", "cond_attr", "target"):
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
        "the 'node_attr' and the 'target' tensor:\n" \
        "- node_attr: shape = [nb_samples, nb_nodes, nb_node_feat]" \
        "- target:    shape = [nb_samples, nb_nodes, nb_target_feat]"
      )
    self.nb_target_feat = tshape[-1]
    # Compose data as (x, y, sample_weight)
    x = []
    for k in ("node_attr", "edge_attr", "cond_attr"):
      if (k in data):
        x.append(data[k])
    _data = [x, [data["target"]]]
    if ("sample_weight" in data):
      _data.append([data["sample_weight"]])
    return tf.nest.map_structure(ops.to_tensor, _data)
