import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# HyperNet library
from hypernet import const
from hypernet.thermophysics import mixture as mix_mod
from hypernet.thermophysics import specie_thermo as st_mod


# =========================================================================== #
# Node encoding
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NodeEnergy(nn_elem.FFBlock):
  """Predicts specific node energy"""

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
    cond_attr_indices=None,
    features_log_xform=None,
    **kwargs
  ):
    super(NodeEnergy, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Static node features
    self.static_node_attr = gno.utils.read_attr(
      data=static_node_attr,
      identifier="static_node_attr"
    )
    # Conditioning inputs for energy prediction
    self.cond_attr_indices = cond_attr_indices
    if (self.cond_attr_indices is not None):
      self.cond_attr_indices = tf.nest.flatten(self.cond_attr_indices)
    # Conditioning inputs log-transformation
    self.features_log_xform = features_log_xform
    if (self.features_log_xform is not None):
      self.log_lay = nn_elem.Log(**self.features_log_xform)

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
    # Predict new node energy attribute
    features = tf.concat(features, axis=-1)
    if (self.features_log_xform is not None):
      features = self.log_lay(features)
    node_energy = super(NodeEnergy, self).call(features, training=training)
    # Compose node attributes
    node_attr = tf.concat([node_attr, node_energy], axis=-1)
    # Return graph
    return gno.utils.return_graph(node_attr, edge_attr, cond_attr, edge_index)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_node_attr": ops.to_numpy(self.static_node_attr),
      "cond_attr_indices": self.cond_attr_indices,
      "features_log_xform": self.features_log_xform
    }
    base_config = super(NodeEnergy, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NodeThermoUpdate(nn_elem.Layer):
  """Computes node partition functions given node temperatures"""

  # Initialization
  # ===================================
  def __init__(
    self,
    node_attr_index_temp={
      "int": 0,
      "chem": 0,
      "trans": 0
    },
    node_attr_index_energy=None,
    node_attr_indices_nd_temp=[0,1],
    specie_thermos={
      "filename": None,
      "thermo_process": 0,
      "eos_model": "PerfectGas",
      "config": None
    },
    **kwargs
  ):
    super(NodeThermoUpdate, self).__init__(**kwargs)
    # Temperature indices
    self.node_attr_index_temp = tf.nest.map_structure(int, node_attr_index_temp)
    # Energy index
    self._update_energy = False
    self.node_attr_index_energy = node_attr_index_energy
    if (self.node_attr_index_energy is not None):
      self._update_energy = True
      self.node_attr_index_energy = int(self.node_attr_index_energy)
    # Number density and temperatures indices
    self.node_attr_indices_nd_temp = tf.nest.flatten(
      tf.nest.map_structure(int, node_attr_indices_nd_temp)
    )
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
    if self._update_energy:
      e = []
    for st in self.sp_th.values():
      # Extract species-related attributes
      sp_attr = tf.gather(node_attr, indices=st.specie.comp_indices, axis=-2)   # Shape: (nb_samples, nb_comp, nb_attr)
      # Compute species-related total partition function
      q.append(self.compute_q_sp(sp_attr, st.pf))                               # Shape: (nb_samples, nb_comp)
      # Compute species-related energy
      if self._update_energy:
        e.append(self.compute_e_sp(sp_attr, st.thermo))
    # Concatenate species-related partition functions
    q = tf.concat(q, axis=-1)                                                   # Shape: (nb_samples, nb_nodes)
    q = tf.expand_dims(q, axis=-1)                                              # Shape: (nb_samples, nb_nodes, 1)
    # Concatenate species-related energy
    if self._update_energy:
      e = tf.concat(e, axis=-1)                                                 # Shape: (nb_samples, nb_nodes)
      e = tf.expand_dims(e, axis=-1)                                            # Shape: (nb_samples, nb_nodes, 1)
    # Compose node attributes
    node_attr = [tf.gather(
      node_attr, indices=self.node_attr_indices_nd_temp, axis=-1
    )]
    if self._update_energy:
      node_attr.append(e)
    node_attr.append(q)
    node_attr = tf.concat(node_attr, axis=-1)
    # Return graph
    return gno.utils.return_graph(node_attr, *_)

  def compute_q_sp(self, attributes, pf_obj):
    qs = []
    for (identifier, index) in self.node_attr_index_temp.items():
      # Extract temperature from species-related attributes
      T = tf.gather(params=attributes, indices=index, axis=-1)                  # Shape: (nb_samples, nb_comp)
      # Retrieve the unique global temperature for "chemical"
      # and "translational" energy modes per each species
      if (identifier in ("chem", "trans")):
        T = tf.gather(T, [0], axis=-1)                                          # Shape: (nb_samples, 1)
      # Compute partition function
      pf = getattr(pf_obj, identifier)
      pf._update(T)
      # Store partition function
      qs.append(pf.Q)
    # Compute total partition function
    return tf_layers.Multiply()(qs)                                             # Shape: (nb_samples, nb_comp)

  def compute_e_sp(self, attributes, th_obj):
    # Compute internal energy
    T = tf.gather(
      params=attributes, indices=self.node_attr_index_temp["int"], axis=-1
    )
    th_obj.int._update(T)
    # Compute internal + formation energy
    e = th_obj.int.e * th_obj.specie.m + th_obj.specie.e_f
    # Scale predicted energy
    return e * tf.gather(
      params=attributes, indices=self.node_attr_index_energy, axis=-1
    )

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = {
      name: st.get_config() for (name, st) in self.sp_th.items()
    }
    config = {
      "node_attr_index_temp": self.node_attr_index_temp,
      "node_attr_index_energy": self.node_attr_index_energy,
      "node_attr_indices_nd_temp": self.node_attr_indices_nd_temp,
      "specie_thermos": {"config": sp_th_cfg}
    }
    base_config = super(NodeThermoUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# Edge encoding
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class EdgeFwdRate(nn_elem.FFBlock):
  """Predicts specific edge forward rates"""

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
    edge_attr_log_xform=None,
    edge_attr_indices_mulfac=[0,1],
    **kwargs
  ):
    super(EdgeFwdRate, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    self.edge_attr_log_xform = edge_attr_log_xform
    if (self.edge_attr_log_xform is not None):
      self.log_lay = nn_elem.Log(**self.edge_attr_log_xform)
    self.edge_attr_indices_mulfac = tf.nest.flatten(edge_attr_indices_mulfac)

  # Calling
  # ===================================
  def call(self, edge_attr, training=False):
    # Compute multiplicative factor
    factor = tf.gather(
      params=edge_attr, indices=self.edge_attr_indices_mulfac, axis=-1
    )
    if (len(self.edge_attr_indices_mulfac) > 1):
      factor = tf.math.reduce_prod(factor, axis=-1, keepdims=True)
    # Apply log-transform
    if (self.edge_attr_log_xform is not None):
      edge_attr = self.log_lay(edge_attr)
    # Predict new forward rates
    edge_attr = factor * super(EdgeFwdRate, self).call(
      edge_attr, training=training
    )
    # Return edges
    return edge_attr

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "edge_attr_log_xform": self.edge_attr_log_xform,
      "edge_attr_indices_mulfac": self.edge_attr_indices_mulfac
    }
    base_config = super(EdgeFwdRate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class EdgeFwdRateCorr(nn_elem.Layer):
  """Applys edge forward rates correction"""

  # Initialization
  # ===================================
  def __init__(
    self,
    edge_attr_index_rate,
    node_attr_index_temp,
    node_attr_index_energy,
    cond_attr_indices_scaling=None,
    static_fwd_edge_attr=None,
    static_fwd_edge_index=None,
    **kwargs
  ):
    super(EdgeFwdRateCorr, self).__init__(**kwargs)
    self.edge_attr_index_rate = int(edge_attr_index_rate)
    self.node_attr_index_temp = int(node_attr_index_temp)
    self.node_attr_index_energy = int(node_attr_index_energy)
    self.cond_attr_indices_scaling = cond_attr_indices_scaling
    if (self.cond_attr_indices_scaling is not None):
      self.cond_attr_indices_scaling = tf.nest.flatten(
        cond_attr_indices_scaling
      )
    self.static_fwd_edge_attr = gno.utils.read_attr(
      data=static_fwd_edge_attr,
      identifier="static_fwd_edge_attr"
    )
    self.static_fwd_edge_index = gno.utils.read_edge_index(
      data=static_fwd_edge_index,
      identifier="static_fwd_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    # -----------------------
    _graph = gno.utils.unpack_graph(
      graph=graph,
      edge_attr=self.static_fwd_edge_attr,
      edge_index=self.static_fwd_edge_index
    )
    node_attr, fwd_edge_attr, cond_attr, fwd_edge_index = _graph
    # Compute rate correction
    # -----------------------
    # > Assume the Boltzmann distribution function
    # > kij_corr = exp(-(ei-ej)/(kT))
    #            = exp(ej/(kT)-ei/(kT))
    #            = exp(zj-zi)
    # Extract temperature and energy
    T = tf.gather(node_attr, indices=[self.node_attr_index_temp], axis=-1)
    e = tf.gather(node_attr, indices=[self.node_attr_index_energy], axis=-1)
    # Compute/Extract exponent
    z = e / (const.UKB * T)
    zi = tf.gather(z, indices=fwd_edge_index[0], axis=-2)
    zj = tf.gather(z, indices=fwd_edge_index[1], axis=-2)
    # Compute correction
    kcorr = tf.math.exp(zj-zi)
    # Apply rate correction
    # -----------------------
    # Extract rate
    k = tf.gather(fwd_edge_attr, indices=[self.edge_attr_index_rate], axis=-1)
    # Compute/Apply scaling factor
    if (self.cond_attr_indices_scaling is not None):
      scale = tf.gather(
        params=cond_attr, indices=self.cond_attr_indices_scaling, axis=-1
      )
      if (len(self.cond_attr_indices_scaling) > 1):
        scale = tf.math.reduce_prod(scale, axis=-1, keepdims=True)
      k = k * scale
    # Apply correction
    fwd_edge_attr = k * kcorr
    # Return graph
    # -----------------------
    return gno.utils.return_graph(
      node_attr, fwd_edge_attr, cond_attr, fwd_edge_index
    )

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "edge_attr_index_rate": self.edge_attr_index_rate,
      "node_attr_index_temp": self.node_attr_index_temp,
      "node_attr_index_energy": self.node_attr_index_energy,
      "cond_attr_indices_scaling": self.cond_attr_indices_scaling,
      "static_fwd_edge_attr": ops.to_numpy(self.static_fwd_edge_attr),
      "static_fwd_edge_index": ops.to_numpy(self.static_fwd_edge_index)
    }
    base_config = super(EdgeFwdRateCorr, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class EdgeBackward(nn_elem.Layer):
  """Calculates the backward edge rate based on the forward rate and
  partition functions of products/reactants"""

  # Initialization
  # ===================================
  def __init__(
    self,
    node_attr_index_pf=-1,
    static_fwd_edge_attr=None,
    static_fwd_edge_index=None,
    **kwargs
  ):
    super(EdgeBackward, self).__init__(**kwargs)
    self.node_attr_index_pf = int(node_attr_index_pf)
    self.static_fwd_edge_attr = gno.utils.read_attr(
      data=static_fwd_edge_attr,
      identifier="static_fwd_edge_attr"
    )
    self.static_fwd_edge_index = gno.utils.read_edge_index(
      data=static_fwd_edge_index,
      identifier="static_fwd_edge_index"
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
    q = tf.gather(params=node_attr, indices=[self.node_attr_index_pf], axis=-1)
    qi = tf.gather(params=q, indices=fwd_edge_index[0], axis=-2)
    qj = tf.gather(params=q, indices=fwd_edge_index[1], axis=-2)
    # Compose edge attributes
    if (self.static_fwd_edge_attr is not None):
      fwd_edge_attr = tf.repeat(
        input=fwd_edge_attr,
        repeats=tf.shape(node_attr)[0],
        axis=0
      )
    bwd_edge_attr = fwd_edge_attr * qi / qj
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
      "node_attr_index_pf": self.node_attr_index_pf,
      "static_fwd_edge_attr": ops.to_numpy(self.static_fwd_edge_attr),
      "static_fwd_edge_index": ops.to_numpy(self.static_fwd_edge_index)
    }
    base_config = super(EdgeBackward, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# Linear graph net
# =========================================================================== #
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
