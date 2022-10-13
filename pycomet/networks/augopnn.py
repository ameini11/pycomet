import numpy as np

from pycomet.networks import elements as nn_elem
from pycomet.networks.opnn import OpNN
from pycomet import utils


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
            p['lay_id'] = 'CrossEncoderBlock'
            lay.append(self.get_lay(propty=p))
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
        return nn_elem.MultiLayerBlock(**kwargs)

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
