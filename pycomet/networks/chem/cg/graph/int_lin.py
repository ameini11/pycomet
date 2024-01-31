import tensorflow as tf

from ....basenet import BaseNet
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet import utils as pcm_utils
from pycomet.networks import elements as nn_elem
from pycomet.networks.gno import utils as gutils
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class IntegralLinearMEGraphNet(BaseNet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    *args,
    **kwargs
  ):
    super(IntegralLinearMEGraphNet, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      masking=False,
      force_siso=False,
      *args,
      **kwargs
    )
    self.process_layers(lays_name)

  def process_layers(self, lays_name):
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + ("edge_encoder", "lin_me_gnet")
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("edge_encoder", "lin_me_gnet"),
      default=self.default_lays_propty
    )
    # Initializing layers
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier in ("input", "output")):
        lays[identifier] = self.init_inout_lay(propty)
    lays.update(self.init_kernel_layers())
    return lays

  def init_kernel_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "lin_me_gnet"):
        lay_id = "LinearMEGraphNet"
        lays[identifier] = self.init_lay(identifier, propty, lay_id)
      elif (identifier == "edge_encoder"):
        lays[identifier] = self.init_lay(propty=propty)[-1]
    return lays

  def init_lay(self, identifier=None, propty=None, lay_id=None):
    lay = []
    propty = propty if isinstance(propty, (list,tuple)) else [propty]
    for p in propty:
      self.check_lay_id(identifier, p, lay_id)
      lay.append(self.get_lay(propty=p))
    return lay

  def check_lay_id(self, identifier=None, propty=None, lay_id=None):
    if ((lay_id is not None) and (propty["lay_id"] != lay_id)):
      utils.raise_err(
        ValueError,
        f"The layer identifier '{identifier}' in '{self.name}' " \
        f"must be associated with the '{lay_id}' layer."
      )

  # Calling
  # ===================================
  def forward(self, graph, training=None):
    x = graph[0]
    # Input nodes manipulation
    graph[0] = self._manipulate_inout(
      inputs=dict(x=x, y=x), identifier="input", training=training
    )
    # Encode-process-decode
    graph = self.encode(graph, training)
    graph = self.process(graph, training)
    y = self.decode(x, graph[0], training)
    # Output nodes manipulation
    y = self._manipulate_inout(
      inputs=dict(x=x, y=y), identifier="output", training=training
    )
    return y

  def encode(self, graph, training):
    # Edge encoder: graph -> graph
    graph = self.active_lays["edge_encoder"](graph, training=training)
    # Node encoder: node_attr -> node_attr
    graph[0] = self.active_lays["node_encoder"](graph[0], training=training)
    return graph

  def process(self, graph, training):
    # Processor: graph -> graph
    for processor in self.active_lays["graphnet"]:
      graph = processor(graph, training=training)
    return graph

  def decode(self, x, y, training):
    # Node decoder: node_attr -> node_attr
    y = self.active_lays["node_decoder"](y, training=training)
    # Residual connection: node_attr
    if ("residual" in self.active_lays):
      y = self.active_lays["residual"]([x, y])
    return y


  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(IntegralLinearMEGraphNet, self).__init__(**kwargs)
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    self.edge_blk = LinearMEGraphEdgeBlock(
      static_edge_index=self.static_edge_index,
      **kwargs
    )
    self.node_blk = LinearMEGraphNodeBlock(
      static_edge_index=self.static_edge_index,
      **kwargs
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Apply edge transformation
    graph[-1] = self.edge_blk(graph, training=training)
    # Apply node transformation
    return self.node_blk(graph, training=training)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(IntegralLinearMEGraphNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMEGraphEdgeBlock(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMEGraphEdgeBlock, self).__init__(**kwargs)
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    # -----------------------
    node_attr, edge_index, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index, None
    )
    # Compute fluxes
    # -----------------------
    # > node_attr: number densities (n_i)
    # > edge_attr: rate constants (k_ij*n_atom)
    # > flux: n_i * (k_ij*n_atom)
    senders_attr = tf.gather(node_attr, edge_index[0], axis=-2)
    return senders_attr * edge_attr

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMEGraphEdgeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMEGraphNodeBlock(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMEGraphNodeBlock, self).__init__(**kwargs)
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    # -----------------------
    node_attr, edge_index, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index, None
    )
    # Aggregate fluxes
    # -----------------------
    # Check rank
    rank = len(edge_attr.shape)
    if (rank not in (2,3)):
      pcm_utils.raise_err(
        ValueError,
        "The 'edge_attr' input tensor has to be of rank 2 " \
        "(shape [nb_edges, nb_edge_features]) or 3 " \
        "(shape [nb_samples, nb_edges, nb_edge_features])."
      )
    # Move batches as last dimension
    if (rank == 3):
      edge_attr = tf.transpose(edge_attr, perm=[1,2,0])
    # Compute fluxes by aggregation
    # > incoming: gather receivers (index 1)
    # > outgoing: gather senders (index 0)
    fluxes = [tf.math.unsorted_segment_sum(
      data=edge_attr,
      segment_ids=edge_index[i],
      num_segments=node_attr.shape[-2]
    ) for i in range(2)]
    # Move batches as first dimension
    if (rank == 3):
      fluxes = [tf.transpose(f, perm=[2,0,1]) for f in fluxes]
    # Update nodes
    # -----------------------
    return fluxes[1] - fluxes[0]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMEGraphNodeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
