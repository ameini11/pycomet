from pycomet.networks import don
from .int_lin_v1 import IntegralLinearMEGraphNetV1


class IntegralLinearMEGraphNetV3(IntegralLinearMEGraphNetV1):

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
    super(IntegralLinearMEGraphNetV3, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      *args,
      **kwargs
    )
    self.process_layers(lays_name)

  def process_layers(self, lays_name):
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + ("nonlin_me_gnet",)
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("nonlin_me_gnet",),
      default=self.default_lays_propty
    )
    # Initializing layers
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # -----------------------------------
  def init_kernel_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "nonlin_me_gnet"):
        lay_id = "NonLinearMEGraphNet"
        lays[identifier] = self.init_lay(identifier, propty, lay_id)
    return lays

  # Calling
  # ===================================
  def forward(self, graph, training=None):
    x = graph[0]
    # Input nodes manipulation
    graph[0] = self._manipulate_inout(
      inputs=dict(x=x, y=x), identifier="input", training=training
    )
    # Graph processor
    y = self.active_lays["nonlin_me_gnet"](graph, training=training)
    # Output nodes manipulation
    y = self._manipulate_inout(
      inputs=dict(x=x, y=y), identifier="output", training=training
    )
    return y



class GraphOpNet(don.MIONet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_propty=None,
    *args,
    **kwargs
  ):
    super(GraphOpNet, self).__init__(
      dim=dim,
      lays_propty=lays_propty,
      *args,
      **kwargs
    )

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == 'message'):
        lays[identifier] = super(GraphOpNet, self).init_layers()
      if (identifier == "aggregate"):
        lay_id = "LinearAggregate"
        lays[identifier] = self.init_lay(identifier, propty, lay_id)
      elif (identifier in ('branch', 'trunk')):
        lays[identifier] = self.init_block_lay(identifier, propty)
      elif (identifier == 'prenet'):
        lays[identifier] = {
          'from': propty['from'],
          'to': propty['to'],
          'block': self.get_lay(
            propty={
              'lay_id': 'PreNetBlock',
              'kwargs': propty['kwargs']
            }
          )
        }
      elif (identifier == 'end_scale'):
        lays[identifier] = self.init_end_scale_lay(identifier, propty)
      elif (identifier == 'output'):
        lays[identifier] = self.init_inout_lay(propty)
    return lays
  
  def init_block_lay(self, identifier, propty):
    lay = []
    propty = propty if isinstance(propty, (list,tuple)) else [propty]
    propty = tf.nest.map_structure(copy.deepcopy, propty)
    for p in propty:
      if identifier.startswith('branch'):
        p['kwargs']['layer_size'][-1] += 2
      lay.append(
        self.get_lay(propty=p, default_lay_id='FFBlock')
      )
    return lay
    


  # Calling
  # ===================================
  def forward(self, x, training=None):
    y = x
    # Inputs manipulation
    y = don.opnn._manipulate_input(self, y)
    # Kernel - OpNN
    G_u, G_y = self._eval_kernel(y, training)
    # Outputs manipulation
    y = self._tensor_product(G_u, G_y)
    y = self._manipulate_output(x, y, training)
    return y
  
