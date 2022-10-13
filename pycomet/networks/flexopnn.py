import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks import elements as nn_elem
from pycomet.networks.opnn import OpNN


class FlexOpNN(OpNN):
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
        if lays_name is None:
            lays_name = (
                'input', 'branch', 'trunk', 'prenet', 'end_scale', 'output'
            )
        super(FlexOpNN, self).__init__(
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
                    'from': propty['from'],
                    'to': propty['to'],
                    'block': self.get_lay(
                        propty={
                            'lay_id': 'PreNetBlock',
                            'kwargs': propty['kwargs']
                        }
                    )
                }
            elif identifier == 'end_scale':
                lays[identifier] = self.init_end_scale_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_inout_lay(identifier, propty)
        return lays
    
    def init_block_lay(self, identifier, propty):
        lay = []
        propty = propty if isinstance(propty, (list,tuple)) else [propty]
        for i, p in enumerate(propty):
            if identifier.startswith('branch'):
                p['kwargs']['layer_size'][-1] += 2
            lay.append(
                self.get_lay(propty=p, default_lay_id='FFBlock')
            )
        return lay

    # Calling
    ###########################################################################
    def _eval_kernel(self, x, training):
        # PreNet
        if 'prenet' in self.active_lays.keys():
            lay = self.active_lays['prenet']
            x[lay['to']] = lay['block'](
                x=[x[lay['from']], x[lay['to']]], training=training
            )
        return super(FlexOpNN, self)._eval_kernel(x, training)

    def _tensor_product(self, G_u, G_y):
        G_u, d, c = list(zip(*[
            tf.split(G_u_i, [G_u_i.shape[1]-2, 1, 1], axis=1) for G_u_i in G_u
        ]))
        y = super(FlexOpNN, self)._tensor_product(G_u, G_y)
        return [
            tf_layers.Add()([c[i], tf_layers.Multiply()([d[i], y[i]])]) \
                for i in range(self.max_num_blocks)
        ]