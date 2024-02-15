from pycomet.networks import gno
from pycomet.networks.basenet import BaseNet


class MEIntEdgeFwdGNet(BaseNet):

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
    super(MEIntEdgeFwdGNet, self).__init__(
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
    if (lays_name is None):
      self.lays_name = self.lays_name + (
        "fwd_edge_encoder", "node_pf", "edge_bwd", "lin_me_gnet", "residual"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=(
        "fwd_edge_encoder", "node_pf", "edge_bwd", "lin_me_gnet", "residual"
      ),
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
    lay_ids = {
      "fwd_edge_encoder": "EdgeFwdRate",
      "node_pf": "NodePartitionFunction",
      "edge_bwd": "EdgeBackward",
      "lin_me_gnet": "LinearMEGraphNet",
      "residual": "GatheredAdd"
    }
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      lay_id = lay_ids.get(identifier, None)
      if (lay_id is not None):
        lays[identifier] = self.init_lay(
          identifier, propty, lay_id=lay_id, default_lay_id=lay_id
        )
    return lays

  # Calling
  # ===================================
  def forward(self, graph, training=None):
    # Unpack graph
    x, edge_attr, cond_attr, edge_index = gno.utils.unpack_graph(graph)
    # Input nodes manipulation
    node_attr = self._manipulate_inout(
      inputs=dict(x=x, y=x), identifier="input", training=training
    )
    # Forward edge encoder
    edge_attr = self.active_lays["fwd_edge_encoder"](
      edge_attr, training=training
    )
    # Compose graph
    graph = gno.utils.return_graph(
      node_attr, edge_attr, cond_attr, edge_index
    )
    # Node partition function
    graph = self.active_lays["node_pf"](graph, training=training)
    # Edge backward
    graph = self.active_lays["edge_bwd"](graph, training=training)
    # Graph processor
    y = self.active_lays["lin_me_gnet"](graph, training=training)
    # Residual connection
    y = self.active_lays["residual"]([x, y], training=training)
    # Output nodes manipulation
    y = self._manipulate_inout(
      inputs=dict(x=x, y=y), identifier="output", training=training
    )
    return y
