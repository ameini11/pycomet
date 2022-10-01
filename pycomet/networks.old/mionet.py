import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from pycomet.networks.elements import layers as pcm_layers
from pycomet.networks.flexopnn import FlexOpNN
from pycomet import config as cfg
from pycomet import utils


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MIONet')
class MIONet(FlexOpNN):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        *args,
        **kwargs
    ):
        super(MIONet, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )
        self.n_branches = {
            k: len(br) for k, br in self.active_lays['branch'].items()
        }
        self.n_trunks = {
            k: len(tf) for k, tf in self.active_lays['trunk'].items()
        }
        # Check number of blocks available
        n_blocks = np.array(
            tf.nest.flatten({**self.n_branches, **self.n_trunks})
        )
        conds = {
            1: np.all((n_blocks == self.outputs_dim) + (n_blocks == 1)),
            2: self.outputs_dim == np.amax(n_blocks)
        }
        if not conds[1] or not conds[2]:
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
                lays[identifier] = self.init_input_lay(identifier, propty)
            elif identifier in ('branch', 'trunk'):
                lays[identifier] = {
                    k: super(MIONet, self).init_block_lay(
                        identifier=identifier+'_'+k, propty=p
                    ) for k, p in propty.items()
                }
            elif identifier == 'prenet':
                lays[identifier] = {
                    k: {
                        'from': tf.nest.flatten(p['from']),
                        'to': p['to'],
                        'block': self.get_block(
                            propty={
                                'lay_id': 'PreNetBlock',
                                'kwargs': p['kwargs']
                            },
                            name=identifier+'_'+k
                        )
                    } for k, p in propty.items()
                }
            elif identifier == 'end_scale':
                lays[identifier] = self.init_end_scale_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_output_lay(identifier, propty)
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
        for i in range(self.outputs_dim):
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
        G_u, d, c = tf.split(G_u, [G_u.shape[1]-2, 1, 1], axis=1)
        y = tf_layers.Dot(axes=1)([G_u, G_y])
        y = tf_layers.Add()([c, tf_layers.Multiply()([d, y])])
        return y
