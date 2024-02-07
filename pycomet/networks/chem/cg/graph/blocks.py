import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# HyperNet library
from hypernet.thermophysics import mixture as mix_mod
from hypernet.thermophysics import specie_thermo as st_mod


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NodeTemperature(nn_elem.FFBlock):
  """Predicts specific node temperature"""

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=["linear"],
    kernel_initializer=None,
    regularizer=None,
    dropout=dict(rate=0),
    layer_norm=None,
    temp={
      "limits": [2e2, 1e5],
      "xform": "exp",
      "index_in": None
    },
    cond_attr_indices=None,
    static_node_attr=None,
    **kwargs
  ):
    super(NodeTemperature, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Set temperature-related inputs
    self.temp = temp
    self.temp["limits"] = tf.nest.flatten(self.temp["limits"])
    self.temp["delta"] = self.temp["limits"][1] - self.temp["limits"][0]
    if (self.temp["index_in"] is not None):
      self.temp["index_in"] = tf.nest.flatten(self.temp["index_in"])
    # Conditioning inputs for temperature prediction
    self.cond_attr_indices = cond_attr_indices
    if (self.cond_attr_indices is not None):
      self.cond_attr_indices = tf.nest.flatten(self.cond_attr_indices)
    # Static node features
    self.static_node_attr = gno.utils.read_attr(
      data=static_node_attr,
      identifier="static_node_attr"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_attr, cond_attr, edge_index = gno.utils.unpack_graph(graph)
    # Collect needed features
    features = []
    # > Extract information from conditioning inputs (e.g., time instant)
    if (self.cond_attr_indices is not None):
      cond_feat = tf.gather(
        params=cond_attr, indices=self.cond_attr_indices, axis=-1
      )
    else:
      cond_feat = cond_attr
    features.append(cond_feat)
    # > Tile static node attributes (e.g., state/group properties)
    if (self.static_node_attr is not None):
      features.append(
        tf.repeat(
          input=self.static_node_attr, repeats=tf.shape(cond_attr)[0], axis=0
        )
      )
    # Predict new temperature attributes
    features = tf.concat(features, axis=-1)
    temp_attr = super(NodeTemperature, self).call(features, training=training)
    # Enforce temperature to be positive and limited
    if (self.temp["index_in"] is not None):
      T = tf.gather(params=node_attr, indices=self.temp["index_in"], axis=-1)
      temp_attr = temp_attr * T + self.temp["limits"][0]
    elif (self.temp["xform"] == "sigmoid"):
      temp_attr = tf.math.sigmoid(temp_attr)
      temp_attr = temp_attr * self.temp["delta"] + self.temp["limits"][0]
    else:
      temp_attr = tf.math.exp(temp_attr) + self.temp["limits"][0]
    # Compose node attributes
    node_attr = tf.concat([node_attr, temp_attr], axis=-1)
    # Return graph
    return gno.utils.return_graph(node_attr, edge_attr, cond_attr, edge_index)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "temp": self.temp,
      "cond_attr_indices": self.cond_attr_indices,
      "static_node_attr": ops.to_numpy(self.static_node_attr)
    }
    base_config = super(NodeTemperature, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class NodePartitionFunctions(nn_elem.Layer):
  """Computes node partition functions given node temperatures"""

  # Initialization
  # ===================================
  def __init__(
    self,
    node_index_temp={
      "int": 0,
      "chem": 0,
      "trans": 0
    },
    specie_thermos={
      "filename": None,
      "thermo_process": 0,
      "eos_model": "PerfectGas",
      "config": None
    },
    **kwargs
  ):
    super(NodePartitionFunctions, self).__init__(**kwargs)
    # Temperature indices
    self.node_index_temp = {
      k: int(index) for (k, index) in node_index_temp.items()
    }
    # Thermophysics
    self.sp_th = st_mod.init_specie_thermos(**specie_thermos)
    st_mod.build_specie_thermos(self.sp_th)
    self.mix = mix_mod.MultiComponent(self.sp_th)

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, *_ = gno.utils.unpack_graph(graph)
    # Loop over species
    q = []
    for st in self.sp_th.values():
      # Extract species-related attributes
      attr = tf.gather(node_attr, st.specie.comp_indices, axis=-2)              # Shape: (nb_samples, nb_comp, nb_attr)
      # Compute species-related total partition function
      qs = self.compute_qs(attr, st)                                            # Shape: (nb_samples, nb_comp)
      # Store total partition function
      q.append(qs)
    # Concatenate species-related partition functions
    q = tf.concat(q, axis=-1)                                                   # Shape: (nb_samples, nb_nodes)
    q = tf.expand_dims(q, axis=-1)                                              # Shape: (nb_samples, nb_nodes, 1)
    # Compose node attributes
    node_attr = tf.concat([node_attr, q], axis=-1)                              # Shape: (nb_samples, nb_nodes, nb_attr+1)
    # Return graph
    return gno.utils.return_graph(node_attr, *_)

  def compute_qs(self, attributes, specie_thermo):
    qs = []
    for (k, index) in self.node_index_temp.items():
      # Extract temperature from species-related attributes
      Tk = tf.gather(attributes, index, axis=-1)                                # Shape: (nb_samples, nb_comp)
      # Retrieve the unique global temperature for "chemical"
      # and "translational" energy modes per each species
      if (k in ("chem", "trans")):
        Tk = tf.gather(Tk, [0], axis=-1)                                        # Shape: (nb_samples, 1)
      # Compute partition function
      pf_k = getattr(specie_thermo.pf, k)
      pf_k._update(Tk)
      # Store partition function
      qs.append(pf_k.Q)
    # Compute total partition function
    return tf_layers.Multiply()(qs)                                             # Shape: (nb_samples, nb_comp)

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = {name: st.get_config() for (name, st) in self.sp_th.items()}
    config = {
      "node_index_temp": self.node_index_temp,
      "specie_thermos": {"config": sp_th_cfg}
    }
    base_config = super(NodePartitionFunctions, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class EdgeBackward(nn_elem.Layer):
  """Calculates the backward edge rate based on the forward rate and
  partition functions of products/reactants"""

  # Initialization
  # ===================================
  def __init__(
    self,
    node_index_pf=[0],
    static_fwd_edge_index=None,
    static_fwd_edge_attr=None,
    **kwargs
  ):
    super(EdgeBackward, self).__init__(**kwargs)
    self.node_index_pf = tf.nest.flatten(node_index_pf)
    self.static_fwd_edge_index = gno.utils.read_edge_index(
      data=static_fwd_edge_index,
      identifier="static_fwd_edge_index"
    )
    self.static_fwd_edge_attr = gno.utils.read_attr(
      data=static_fwd_edge_attr,
      identifier="static_fwd_edge_attr"
    )
    if (self.static_fwd_edge_index is not None):
      self.static_bwd_edge_index = tf.reverse(
        self.static_fwd_edge_index, axis=[0]
      )
      self.edge_index = tf.concat(
        [self.static_fwd_edge_index, self.static_bwd_edge_index], axis=-1
      )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    _graph = gno.utils.unpack_graph(
      graph=graph,
      edge_attr=self.static_fwd_edge_attr,
      edge_index=self.static_fwd_edge_index
    )
    node_attr, fwd_edge_attr, cond_attr, fwd_edge_index = _graph
    # Extract partition functions
    q = tf.gather(node_attr, self.node_index_pf, axis=-1)
    q_in = tf.gather(q, fwd_edge_index[0], axis=-2)
    q_out = tf.gather(q, fwd_edge_index[1], axis=-2)
    # Compose edge attributes
    if (self.static_fwd_edge_attr is not None):
      fwd_edge_attr = tf.repeat(
        input=fwd_edge_attr,
        repeats=tf.shape(node_attr)[0],
        axis=0
      )
    bwd_edge_attr = fwd_edge_attr * q_in / q_out
    edge_attr = tf.concat([fwd_edge_attr, bwd_edge_attr], axis=-2)
    # Compose edge index
    if (self.static_fwd_edge_index is not None):
      edge_index = self.edge_index
    else:
      bwd_edge_index = tf.reverse(fwd_edge_index, axis=[0])
      edge_index = tf.concat([fwd_edge_index, bwd_edge_index], axis=-1)
    # Return graph
    return gno.utils.return_graph(node_attr, edge_attr, cond_attr, edge_index)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "node_index_pf": self.node_index_pf,
      "static_fwd_edge_index": ops.to_numpy(self.static_fwd_edge_index),
      "static_fwd_edge_attr": ops.to_numpy(self.static_fwd_edge_attr)
    }
    base_config = super(EdgeBackward, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMEGraphNet(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    node_attr_indices=None,
    edge_attr_indices=None,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMEGraphNet, self).__init__(**kwargs)
    self.node_attr_indices = node_attr_indices
    self.edge_attr_indices = edge_attr_indices
    self.static_edge_index = gno.utils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    self.message = gno.LinearMessage(
      origin="sender",
      reduction="prod",
      include_edge=True,
      node_attr_indices=self.node_attr_indices,
      edge_attr_indices=self.edge_attr_indices,
      static_edge_index=self.static_edge_index
    )
    self.aggregate = gno.Aggregate(
      fluxes="both",
      reduction="sum"
    )
    self.update = gno.LinearUpdate(
      reduction="diff"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Edges - Message passing
    graph = self.message(graph, training=training)
    # Edges - Aggregation
    graph = self.aggregate(graph, training=training)
    # Nodes - Updating
    return self.update(graph, training=training)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "node_attr_indices": self.node_attr_indices,
      "edge_attr_indices": self.edge_attr_indices,
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMEGraphNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package=_CFG.name)
# class NonLinearMEGraphNet(nn_elem.Layer):

#   # Initialization
#   # ===================================
#   def __init__(
#     self,
#     layer_size=None,
#     activation=["linear"],
#     kernel_initializer=None,
#     regularizer=None,
#     dropout=dict(rate=0),
#     layer_norm=None,
#     static_edge_index=None,
#     log_xform=None,
#     **kwargs
#   ):
#     super(NonLinearMEGraphNet, self).__init__(**kwargs)
#     # Hyperparameters
#     # -----------------------
#     self.layer_size = layer_size
#     self.activation = activation
#     self.kernel_initializer = kernel_initializer
#     self.regularizer = regularizer
#     self.dropout = dropout
#     self.layer_norm = layer_norm
#     self.log_xform = log_xform
#     self.static_edge_index = gutils.read_edge_index(
#       data=static_edge_index,
#       identifier="static_edge_index"
#     )
#     # Kernel
#     # -----------------------
#     self.message = glay.NonLinearMessagePassing(
#       layer_size=self.layer_size,
#       activation=self.activation,
#       kernel_initializer=self.kernel_initializer,
#       regularizer=self.regularizer,
#       dropout=self.dropout,
#       layer_norm=self.layer_norm,
#       static_edge_index=self.static_edge_index,
#       log_xform=self.log_xform,
#       **kwargs
#     )
#     self.aggregate = glay.LinearAggregate(
#       static_edge_index=self.static_edge_index,
#       **kwargs
#     )

#   # Calling
#   # ===================================
#   def call(self, graph, training=False):
#     graph_new = tf.nest.map_structure(tf.identity, graph)
#     # Apply edge transformation
#     graph_new[-1] = self.message(graph, training=training)
#     # Nodes - Aggregation
#     return self.aggregate(graph_new, training=training)

#   # Configuration update
#   # ===================================
#   def get_config(self):
#     config = {
#       "layer_size": self.layer_size,
#       "activation": self.activation,
#       "kernel_initializer": self.kernel_initializer,
#       "regularizer": self.regularizer,
#       "dropout": self.dropout,
#       "layer_norm": self.layer_norm,
#       "static_edge_index": ops.to_numpy(self.static_edge_index),
#       "log_xform": self.log_xform
#     }
#     base_config = super(NonLinearMEGraphNet, self).get_config()
#     return dict(list(base_config.items()) + list(config.items()))
