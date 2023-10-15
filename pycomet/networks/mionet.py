import numpy as np
import tensorflow as tf

from keras import layers as tf_layers
from pycomet import utils
from pycomet.networks.flexopnn import FlexOpNN
from pycomet.networks import elements as nn_elem
from pycomet.networks.opnn import _init_input_lay



class MIONet(FlexOpNN):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        force_siso=False,
        noisy_io=None,
        *args,
        **kwargs
    ):
        super(MIONet, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            force_siso=force_siso,
            noisy_io=noisy_io,
            *args,
            **kwargs
        )
        self.n_branches = {
            k: len(br) for k, br in self.active_lays['branch'].items()
        }
        self.n_trunks = {
            k: len(tr) for k, tr in self.active_lays['trunk'].items()
        }
        # Check number of blocks available
        n_blocks = np.array(
            tf.nest.flatten({**self.n_branches, **self.n_trunks})
        )
        conditions = {
            1: np.all((n_blocks == self.output_dim) + (n_blocks == 1)),
            2: self.output_dim == np.amax(n_blocks)
        }
        if not conditions[1] or not conditions[2]:
            utils.raise_value_err(
                "The number of blocks for each available kernel layer has to "
                "be equal to 1 or the number of outputs. Check ('branch', "
                "'trunk', 'prenet') and the model outputs dimensions."
            )

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                lays[identifier] = _init_input_lay(self, identifier, propty)
            elif identifier in ('branch', 'trunk'):
                lays[identifier] = {
                    k: super(MIONet, self).init_block_lay(
                        identifier='_'.join((identifier, k)), propty=p
                    ) for k, p in propty.items()
                }
            elif identifier == 'prenet':
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
                    } for k, p in propty.items()
                }
            elif identifier == 'end_scale':
                lays[identifier] = self.init_end_scale_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_inout_lay(propty)
        return lays

    # Calling
    ###########################################################################
    def _eval_kernel(self, x, training):
        # PreNet
        if 'prenet' in self.active_lays.keys():
            for k, p in self.active_lays['prenet'].items():
                x1 = [x[name] for name in p['from']]
                x1 = tf_layers.Concatenate()(x1) if len(x1) > 1 else x1[-1]
                x[p['to']] = p['block'](x=[x1, x[p['to']]], training=training)
        # Branch/Trunk
        G_u = {
            k: [br(x[k], training=training) for br in branches] \
                for k, branches in self.active_lays['branch'].items()
        }
        G_y = {
            k: [tr(x[k], training=training) for tr in trunks] \
                for k, trunks in self.active_lays['trunk'].items()
        }
        return G_u, G_y

    def _tensor_product(self, G_u, G_y):
        y = []
        for i in range(self.output_dim[-1]):
            G_u_i, G_y_i = [
                [g[i] if len(g) > 1 else g[-1] for k, g in G.items()] \
                    for G in [G_u, G_y]
            ]
            y.append(self._single_tensor_product(G_u_i, G_y_i))
        return y

    def _single_tensor_product(self, G_u, G_y):
        G_u, G_y = [
            tf_layers.Multiply()(x) if len(x) > 1 else x[-1] \
                for x in [G_u, G_y]
        ]
        G_u, d, c = nn_elem.Split([G_u.shape[1]-2, 1, 1], axis=1)(G_u)
        y = tf_layers.Dot(axes=1)([G_u, G_y])
        y = tf_layers.Add()([c, tf_layers.Multiply()([d, y])])
        return y
