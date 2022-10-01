import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from pycomet.networks.elements import layers as pcm_layers
from pycomet.networks.elements import blocks
from pycomet.networks.opnn import OpNN
from pycomet import config as cfg
from pycomet import utils


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='AugOpNN')
class AugOpNN(OpNN):
    """Augmented Deep Operator neural network.
    See: https://arxiv.org/pdf/2110.01654.pdf
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
        super(AugOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )
        self.check_layer_size()
        self.block_keys = ('use_bias', 'kernel_initializer', 'regularizer',
            'weight_norm', 'batch_norm', 'dropout_kwargs')
        for (identifier, block) in (('U','branch'), ('V','trunk')):
            propty = self.lays_propty[block]
            if isinstance(propty, (list, tuple)):
                propty = propty[-1]
            self.active_lays[identifier] = self.init_uv(identifier, propty)

    # Layers initialization
    ###########################################################################
    def init_block_lay(self, identifier, propty):
        lay = []
        propty = propty if isinstance(propty, (list,tuple)) else [propty]
        for i, p in enumerate(propty):
            name = f'{identifier}' if i == 0 else f'{identifier}_{i}'
            p['lay_id'] = 'CrossEncoderBlock'
            lay.append(self.get_block(propty=p, name=name))
        return lay

    def check_layer_size(self):
        sizes = []
        for identifier in ('branch', 'trunk'):
            propty = self.lays_propty[identifier]
            if not isinstance(propty, (list, tuple)):
                propty = [propty]
            for p in propty:
                sizes.append(p['kwargs']['layer_size'][1])
        if not all(np.array(sizes) == sizes[0]):
            utils.raise_value_err(
                "The number of neurons between 'branch' "
                "and 'trunk' have to be equal."
            )

    def init_uv(self, identifier, propty):
        kwargs = {
            k: v for k, v in propty['kwargs'].items() if k in self.block_keys
        }
        kwargs['units'] = propty['kwargs']['layer_size'][1]
        kwargs['activation'] = propty['kwargs']['activation'][0]
        kwargs['name'] = identifier
        return blocks.MultiLayerBlock(**kwargs)

    # Calling
    ###########################################################################
    def _eval_kernel(self, x, training):
        # Augmented cross-encoding
        u = self.active_lays['U'](x["branch"], training=training)
        v = self.active_lays['V'](x["trunk"], training=training)
        # Branch/Trunk
        G_u = [
            br(x["branch"], training=training, u=u, v=v) \
                for br in self.active_lays['branch']
        ]
        G_y = [
            tr(x["trunk"], training=training, u=u, v=v) \
                for tr in self.active_lays['trunk']
        ]
        return G_u, G_y
