import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from pycomet.networks.elements import layers as pcm_layers
from pycomet.networks.flexopnn import FlexOpNN
from pycomet import config as cfg
from pycomet import utils


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MIONet')
class MIONet(FlexOpNN):
    """Flex Deep Operator neural network.
    See: https://arxiv.org/pdf/2204.12670.pdf
    """

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

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                lays[identifier] = self.init_input_lay(identifier, propty)
            elif identifier in ('branch', 'trunk'):
                lays[identifier] = self.init_block_lay(identifier, propty)
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

    def init_block_lay(self, identifier, propty):
        lay = {}
        for k, p in propty.items():
            if identifier == 'branch':
                p['kwargs']['layer_size'][-1] += 2
            lay[k] = self.get_block(propty=p, name=identifier+'_'+k)
        return lay

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
        G_u = [
            br(x[k], training=training) \
                for k, br in self.active_lays['branch'].items()
        ]
        G_y = [
            tr(x[k], training=training) \
                for k, tr in self.active_lays['trunk'].items()
        ]
        return G_u, G_y

    def _dot_product(self, G_u, G_y):
        G_u, G_y = [
            tf_layers.Multiply()(x) if len(x) > 1 else x[-1] \
                for x in [G_u, G_y]
        ]
        G_u, d, c = tf.split(G_u, [G_u.shape[1]-2, 1, 1], axis=1)
        y = tf_layers.Dot(axes=1)([G_u, G_y])
        y = tf_layers.Add()([c, tf_layers.Multiply()([d, y])])
        return y
