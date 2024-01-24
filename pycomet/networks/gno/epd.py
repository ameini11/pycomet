from pycomet import utils
from ..basenet import BaseNet

import tensorflow as tf

class EncodeProcessDecode(BaseNet):

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
    super(EncodeProcessDecode, self).__init__(
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
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + (
        'graphnet', 'edge_encoder', 'node_encoder', 'node_decoder'
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=('graphnet', 'edge_encoder', 'node_encoder', 'node_decoder'),
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
      if (identifier in ('input', 'output')):
        lays[identifier] = self.init_inout_lay(propty)
      elif (identifier == 'graphnet'):
        lay_id = 'GraphNet'
        lays[identifier] = self.init_block_lay(identifier, propty, lay_id)
      elif (identifier == 'edge_encoder'):
        lay_id = 'EdgeEncoder'
        lays[identifier] = self.init_block_lay(identifier, propty, lay_id)[-1]
      elif (identifier == 'node_encoder'):
        lays[identifier] = self.init_block_lay(propty=propty)[-1]
      elif (identifier == 'node_decoder'):
        lays[identifier] = self.init_block_lay(propty=propty)[-1]
    return lays

  def init_block_lay(self, identifier=None, propty=None, lay_id=None):
    lay = []
    propty = propty if isinstance(propty, (list,tuple)) else [propty]
    for p in propty:
      if ((lay_id is not None) and (p['lay_id'] != lay_id)):
        utils.raise_err(
          ValueError,
          f"The layer identifier '{identifier}' in {self.name} " \
          f"must be associated with the '{lay_id}' layer."
        )
      lay.append(self.get_lay(propty=p))
    return lay

  # Calling
  # ===================================
  def forward(self, graph, training=None):
    x = graph[0]
    # Input nodes manipulation
    graph[0] = self._manipulate_inout(
      inputs=dict(x=x, y=x), identifier='input', training=training
    )
    # Edge encoder: graph -> graph
    graph = self.active_lays["edge_encoder"](graph, training=training)
    # Node encoder: node_attr -> node_attr
    graph[0] = self.active_lays["node_encoder"](graph[0], training=training)
    # Processor: graph -> graph
    for processor in self.active_lays["graphnet"]:
      graph = processor(graph)
    # Node decoder: node_attr -> node_attr
    graph[0] = self.active_lays["node_decoder"](graph[0], training=training)
    # Output nodes manipulation
    y = graph[0] + x
    y = self._manipulate_inout(
      inputs=dict(x=x, y=y), identifier='output', training=training
    )
    return y
