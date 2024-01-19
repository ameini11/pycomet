import numpy as np
import tensorflow as tf

from pycomet import utils
from .flexopnn import FlexOpNN
from .opnn import _init_input_lay
from .. import elements as nn_elem
from keras import layers as tf_layers


class MIONet(FlexOpNN):

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
    super(MIONet, self).__init__(
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
    self.nb_branches = {
      k: len(br) for (k, br) in self.active_lays['branch'].items()
    }
    self.nb_trunks = {
      k: len(tr) for (k, tr) in self.active_lays['trunk'].items()
    }
    # Check number of blocks available
    nb_blocks = np.array(
      tf.nest.flatten({**self.nb_branches, **self.nb_trunks})
    )
    c1 = np.all((nb_blocks == self.output_dim) + (nb_blocks == 1))
    c2 = (self.output_dim == np.amax(nb_blocks))
    if ((not c1) or (not c2)):
      utils.raise_err(
        ValueError,
        "The number of blocks for each available kernel layer has to " \
        "be equal to 1 or the number of outputs. Check ('branch', " \
        "'trunk', 'prenet') and the model outputs dimensions."
      )

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == 'input'):
        lays[identifier] = _init_input_lay(self, identifier, propty)
      elif (identifier in ('branch', 'trunk')):
        lays[identifier] = {
          k: super(MIONet, self).init_block_lay(
            identifier='_'.join((identifier, k)), propty=p
          ) for (k, p) in propty.items()
        }
      elif (identifier == 'prenet'):
        lays[identifier] = {
          k: {
            'from': tf.nest.flatten(p['from']),
            'to': p['to'],
            'block': self.get_lay(
              propty={
                'lay_id': 'PreNetBlock',
                'kwargs': p['kwargs']
              }
            )
          } for (k, p) in propty.items()
        }
      elif (identifier == 'end_scale'):
        lays[identifier] = self.init_end_scale_lay(identifier, propty)
      elif (identifier == 'output'):
        lays[identifier] = self.init_inout_lay(propty)
    return lays

  # Calling
  # ===================================
  def _eval_kernel(self, x, training):
    # PreNet
    if ('prenet' in self.active_lays.keys()):
      for p in self.active_lays['prenet'].values():
        x1 = [x[name] for name in p['from']]
        x1 = tf_layers.Concatenate()(x1) if (len(x1) > 1) else x1[-1]
        x[p['to']] = p['block'](x=[x1, x[p['to']]], training=training)
    # Branch/Trunk
    G_u = {
      k: [br(x[k], training=training) for br in branches] \
        for (k, branches) in self.active_lays['branch'].items()
    }
    G_y = {
      k: [tr(x[k], training=training) for tr in trunks] \
        for (k, trunks) in self.active_lays['trunk'].items()
    }
    return G_u, G_y

  def _tensor_product(self, G_u, G_y):
    y = []
    for i in range(self.output_dim[-1]):
      G_u_i, G_y_i = [
        [g[i] if (len(g) > 1) else g[-1] for g in G.values()] \
          for G in (G_u, G_y)
      ]
      y.append(self._single_tensor_product(G_u_i, G_y_i))
    return y

  def _single_tensor_product(self, G_u, G_y):
    G_u, G_y = [
      tf_layers.Multiply()(x) if (len(x) > 1) else x[-1] for x in (G_u, G_y)
    ]
    G_u, d, c = nn_elem.Split([G_u.shape[1]-2,1,1], axis=1)(G_u)
    y = c + d * tf_layers.Dot(axes=1)([G_u, G_y])
    return y
