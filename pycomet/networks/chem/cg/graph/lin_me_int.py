from pycomet import utils
from pycomet.networks import gno
from pycomet.networks.basenet import BaseNet
import tensorflow as tf

class LinMEIntGraphNet(BaseNet):

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
    super(LinMEIntGraphNet, self).__init__(
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
      self.lays_name = self.lays_name + (
        "node_temp", "node_pf", "edge_bwd", "lin_me_gnet", "residual"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("node_temp", "node_pf", "edge_bwd", "lin_me_gnet", "residual"),
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
      "node_temp": "NodeTemperature",
      "node_pf": "NodePartitionFunctions",
      "edge_bwd": "EdgeBackward",
      "lin_me_gnet": "LinearMEGraphNet",
      "residual": "GatheredAdd"
    }
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      lay_id = lay_ids.get(identifier, None)
      if (lay_id is not None):
        lays[identifier] = self.init_lay(identifier, propty, lay_id)
    return lays

  def init_lay(self, identifier=None, propty=None, lay_id=None):
    self.check_lay_id(identifier, propty, lay_id)
    return self.get_lay(propty=propty)

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
    # Unpack graph
    x, edge_attr, cond_attr, edge_index = gno.utils.unpack_graph(graph)
    # Input nodes manipulation
    node_attr = self._manipulate_inout(
      inputs=dict(x=x, y=x), identifier="input", training=training
    )
    graph = gno.utils.return_graph(node_attr, edge_attr, cond_attr, edge_index)
    # Node temperature
    graph = self.active_lays["node_temp"](graph, training=training)
    # Node partition functions
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
