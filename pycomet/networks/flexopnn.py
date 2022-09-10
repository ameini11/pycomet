import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from pycomet.networks.elements import layers as pcm_layers
from pycomet.networks.opnn import OpNN
from pycomet import config as cfg
from pycomet import utils


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='FlexOpNN')
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
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                self.init_input_lay(identifier, propty)
            elif identifier in ('prenet', 'branch', 'trunk'):
                self.init_block_lay(identifier, propty)
            elif identifier == 'end_scale':
                self.init_end_scale_lay(identifier, propty)
            elif identifier == 'output':
                self.init_output_lay(identifier, propty)

    def init_block_lay(self, identifier, propty):
        propty = propty if isinstance(propty, (list,tuple)) else [propty]
        self.active_lays[identifier] = []
        for i, p in enumerate(propty):
            name = f'{identifier}' if i == 0 else f'{identifier}_{i}'
            if identifier == 'branch':
                p['kwargs']['layer_size'][-1] += 2
            self.active_lays[identifier].append(
                self.get_block(propty=p, name=name)
            )

    # Calling
    ###########################################################################
    def _eval_kernel(self, x, training):
        # PreNet
        if 'prenet' in self.active_lays.keys():
            x["trunk"] = self.active_lays['prenet'][-1](
                x=[x["branch"], x["trunk"]], training=training
            )
        return super(FlexOpNN, self)._eval_kernel(x, training)

    def _dot_product(self, G_u, G_y):
        G_u, d, c = list(zip(*[
            tf.split(G_u_i, [G_u_i.shape[1]-2, 1, 1], axis=1) for G_u_i in G_u
        ]))
        y = super(FlexOpNN, self)._dot_product(G_u, G_y)
        return [
            tf_layers.Add()([c[i], tf_layers.Multiply()([d[i], y[i]])]) \
                for i in range(self.outputs_dim)
        ]