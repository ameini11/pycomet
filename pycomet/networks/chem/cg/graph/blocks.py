import tensorflow as tf

from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


from keras import layers as tf_layers

from hypernet import utils
from hypernet import solvers as sol_mod
from hypernet.thermophysics import mixture as mix_mod
from hypernet.thermophysics import kinetics as kin_mod
from hypernet.thermophysics import specie_thermo as st_mod




@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMEGraphNet(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMEGraphNet, self).__init__(**kwargs)
    self.static_edge_index = gno.utils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    self.message = gno.LinearMessage(
      origin="sender",
      reduction="prod",
      include_edge=True,
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
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMEGraphNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NodeEncoder(nn_elem.FFBlock):

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
    static_node_attr=None,
    **kwargs
  ):
    super(NodeEncoder, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    self.static_node_attr = gno.utils.read_attr(
      data=static_node_attr,
      identifier="static_node_attr"
    )

  # Calling
  # ===================================
  def call(self, node_attr, training=False):
    features = []
    # Extract needed information from node (e.g., time instant)
    features.append(
      tf.gather(
        params=node_attr,
        indices=self.feat_indices,
        axis=-1
      )
    )
    # Tile static node attributes (e.g., state/group properties)
    if (self.static_node_attr is not None):
      features.append(
        tf.repeat(
          input=self.static_node_attr,
          repeats=tf.shape(node_attr)[0],
          axis=0
        )
      )
    # Predict new node attributes
    features = tf.concat(features, axis=-1)
    node_attr_new = super(NodeEncoder, self).call(features, training=training)
    # Return node attributes
    return tf.concat([node_attr, node_attr_new], axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_node_attr": ops.to_numpy(self.static_node_attr)
    }
    base_config = super(NodeEncoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class NodePartitionFunctions(tf_layers.Layer):
  """Computes node partition functions given node temperatures"""

  # Initialization
  # ===================================
  def __init__(
    self,
    temp_indices={
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
    self.temp_indices = {k: int(index) for (k, index) in temp_indices.items()}
    # Thermophysics
    self.sp_th = st_mod.init_specie_thermos(**specie_thermos)
    st_mod.build_specie_thermos(self.sp_th)
    self.mix = mix_mod.MultiComponent(self.sp_th)

  # Calling
  # ===================================
  def call(self, node_attr, training=False):                                    # Shape: (nb_samples, nb_nodes, nb_attr)
    Q = []
    # Loop over species
    for st in self.sp_th.values():
      # Extract species-related attributes
      attr = tf.gather(node_attr, st.specie.comp_indices, axis=-2)              # Shape: (nb_samples, nb_comp, nb_attr)
      # Compute species-related total partition function
      Qs = self.compute_qs(attr, st)                                            # Shape: (nb_samples, nb_comp)
      # Store total partition function
      Q.append(Qs)
    # Concatenate species-related partition functions
    Q = tf.concat(Q, axis=-1)                                                   # Shape: (nb_samples, nb_nodes)
    Q = tf.expand_dims(Q, axis=-1)                                              # Shape: (nb_samples, nb_nodes, 1)
    # Concatenate all node attributes
    return tf.concat([node_attr, Q], axis=-1)                                   # Shape: (nb_samples, nb_nodes, nb_attr+1)

  def compute_qs(self, attributes, specie_thermo):
    Qs = []
    for (k, index) in self.temp_indices.items():
      # Extract temperature from species-related attributes
      Tk = tf.gather(attributes, index, axis=-1)                                # Shape: (nb_samples, nb_comp)
      # Retrieve the unique global temperature for 'chemical'
      # and 'translational' energy modes per each species
      if (k in ("chem", "trans")):
        Tk = tf.gather(Tk, [0], axis=-1)                                       # Shape: (nb_samples, 1)
      # Compute partition function
      pf_k = getattr(specie_thermo.pf, k)
      pf_k._update(Tk)
      # Store partition function
      Qs.append(pf_k.Q)
    # Compute total partition function
    return tf_layers.Multiply()(Qs)                                             # Shape: (nb_samples, nb_comp)

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = {name: st.get_config() for (name, st) in self.sp_th.items()}
    config = {
      "temp_indices": self.temp_indices,
      "specie_thermos": {"config": sp_th_cfg}
    }
    base_config = super(NodePartitionFunctions, self).get_config()
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
