from pycomet import utils
from pycomet import networks
from ..basenet import BaseNet
from .flexopnn import _eval_prenet
from .opnn import _init_input_lay, _manipulate_input


class MMOpNN(BaseNet):
  """Multi-Physics Multi-Scale Deep Operator neural network."""

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    masking=False,
    force_siso=False,
    *args,
    **kwargs
  ):
    super(MMOpNN, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      masking=masking,
      force_siso=force_siso,
      *args,
      **kwargs
    )
    # Layers
    # -----------------------
    # Default layers
    self.default_lays_propty.update({
      'input': {
        'branch': {
          'index': 0,
          'xform': None
        },
        'trunk': {
          'index': 1,
          'xform': None
        }
      },
      'output': {
        'xform': None
      }
    })
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + ('prenet', 'dons')
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=('dons',),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == 'input'):
        lays[identifier] = _init_input_lay(self, identifier, propty)
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
      elif (identifier == 'dons'):
        lays[identifier] = self.init_dons_lay(propty)
      elif (identifier == 'output'):
        lays[identifier] = self.init_inout_lay(propty)
    return lays

  def init_dons_lay(self, propty):
    if (not isinstance(propty, dict)):
      utils.raise_err(
        ValueError,
        "The properties for 'dons' layer has to be provided " \
        "as a dictionary of (don_name, don_args) pairs."
      )
    return {name: self.get_don(don_lays_propty) \
      for name, don_lays_propty in propty.items()}

  def get_don(self, propty={}):
    don = 'OpNN' if ('lay_id' not in propty) else propty['lay_id']
    don = don if callable(don) else utils.get_class(networks, don)
    return don(dim=propty['dim'], lays_propty=propty['lays_propty'])

  # Calling
  # ===================================
  def forward(self, x, training=None):
    # Inputs manipulation
    y = _manipulate_input(self, x)
    # Global preNet
    y = _eval_prenet(self, y, training)
    # Kernel
    yk = []
    for don in self.active_lays['dons'].values():
      yk.append(don(y, training=training))
    # Outputs manipulation
    y = self._manipulate_inout(
      inputs=dict(x=x, y=yk), identifier='output', training=training
    )
    return y
