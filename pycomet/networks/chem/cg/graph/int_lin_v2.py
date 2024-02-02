from pycomet.networks import gno
from .int_lin_v1 import IntegralLinearMEGraphNetV1


class IntegralLinearMEGraphNetV2(IntegralLinearMEGraphNetV1):

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
    super(IntegralLinearMEGraphNetV2, self).__init__(
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
      self.lays_name = self.lays_name + ("message",)
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("message",),
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
      if (identifier == "message"):
        lay_id = "NonLinearMessage"
        lays[identifier] = self.init_lay(identifier, propty, lay_id)
    lays["aggregate"] = gno.Aggregate(
      fluxes="both",
      reduction="sum"
    )
    lays["update"] = gno.LinearUpdate(
      reduction="diff"
    )
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
    graph = self.active_lays["message"](graph, training=training)
    graph = self.active_lays["aggregate"](graph, training=training)
    y = self.active_lays["update"](graph, training=training)
    # Output nodes manipulation
    y = self._manipulate_inout(
      inputs=dict(x=x, y=y), identifier="output", training=training
    )
    return y
