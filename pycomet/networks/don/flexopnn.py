import copy
import tensorflow as tf

from .. import elements as nn_elem
from .opnn import OpNN, _init_input_lay


class FlexOpNN(OpNN):
  """Flex Deep Operator neural network.
  See: https://arxiv.org/pdf/2204.12670.pdf
  """

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
    return_latent_space=None,
    *args,
    **kwargs
  ):
    if (lays_name is None):
      lays_name = (
        'preinput', 'input', 'branch', 'trunk', 'prenet', 'end_scale', 'output'
      )
    super(FlexOpNN, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      masking=masking,
      force_siso=force_siso,
      return_latent_space=return_latent_space,
      *args,
      **kwargs
    )

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == 'input'):
        lays[identifier] = _init_input_lay(self, identifier, propty)
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
        lays[identifier] = self.init_inout_lay(identifier, propty)
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
  def _eval_kernel(self, x, training):
    y = _eval_prenet(self, x, training)
    return super(FlexOpNN, self)._eval_kernel(y, training)

  def _tensor_product(self, G_u, G_y):
    G_u = [nn_elem.Split([G_u_i.shape[1]-2,1,1], 1)(G_u_i) for G_u_i in G_u]
    G_u, d, c = list(zip(*G_u))
    y = super(FlexOpNN, self)._tensor_product(G_u, G_y)
    return [c[i] + d[i]*y[i] for i in range(len(c))]

# Shared functions
# =====================================
def _eval_prenet(nn_cls, x, training=None):
  if ('prenet' in nn_cls.active_lays.keys()):
    lay = nn_cls.active_lays['prenet']
    if isinstance(lay, dict):
      x1, x2 = x[lay['from']], x[lay['to']]
      x2 = lay['block'](x=[x1, x2], training=training)
      return {lay['from']: x1, lay['to']: x2}
    else:
      x1, x2 = x
      x2 = lay(x=[x1, x2], training=training)
    return [x1, x2]
  else:
    return x
