from pycomet import utils
from keras import layers as tf_layers
from .epd import EncodeProcessDecode
from ..don.flexopnn import _eval_prenet
from ..don.opnn import _init_input_lay, _manipulate_input


class EncodeProcessConditionDecode(EncodeProcessDecode):

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
    super(EncodeProcessConditionDecode, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      *args,
      **kwargs
    )

  def process_layers(self, lays_name):
    self.input_keys = ("node", "edge", "cond")
    # Default layers
    self.default_lays_propty.update({
      "input": {
        k: {
          "index": i,
          "xform": None
        } for (i, k) in enumerate(self.input_keys)
      },
      "output": {
        "xform": None
      }
    })
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + (
        "prenet", "cond_encoder", "edge_encoder", "node_encoder", \
        "graphnet", "node_decoder", "residual"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=(
        "cond_encoder", "graphnet", "edge_encoder", \
        "node_encoder", "node_decoder"
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
      if (identifier == "input"):
        lays[identifier] = _init_input_lay(self, identifier, propty)
        self.check_input_lay(lays[identifier])
      elif (identifier == "cond_encoder"):
        lays[identifier] = self.init_lay(propty=propty)[-1]
      elif (identifier == "prenet"):
        lays[identifier] = {
          "from": propty["from"],
          "to": propty["to"],
          "block": self.get_lay(
            propty={
              "lay_id": "PreNetBlock",
              "kwargs": propty["kwargs"]
            }
          )
        }
      elif (identifier == "output"):
        lays[identifier] = self.init_inout_lay(propty)
    lays.update(self.init_kernel_layers())
    return lays

  def check_input_lay(self, propty=None):
    if any([(k not in propty) for k in self.input_keys]):
      utils.raise_err(
        ValueError,
        "The specified input identifiers should be 'node', 'edge' " \
        "and 'cond'. 'cond' represents the conditioning input, " \
        "while 'node' and 'edge' are used to denote the input graph. " \
        "Check the 'input' layer entry."
      )

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Inputs manipulation
    inputs = _manipulate_input(self, inputs)
    # Condition
    ycond = self.condition(inputs, training)
    # Encode-process
    graph = [inputs[k] for k in ("node", "edge")]
    graph = self.encode(graph, training)
    graph = self.process(graph, training)
    # Decode
    x = inputs["node"]
    y = tf_layers.Concatenate(axis=-1)([graph[0], ycond])
    y = self.decode(x, y, training)
    # Output nodes manipulation
    y = self._manipulate_inout(
      inputs=dict(x=x, y=y), identifier="output", training=training
    )
    return y

  def condition(self, inputs, training):
    y = inputs
    if ("prenet" in self.active_lays):
      y = _eval_prenet(self, y, training)
    y = self.active_lays["cond_encoder"](y["cond"], training=training)
    return y
