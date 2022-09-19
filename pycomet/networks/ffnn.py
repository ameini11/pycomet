import tensorflow as tf

from pycomet.networks.net import Net
from pycomet import config as cfg


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='FFNN')
class FFNN(Net):
    """Feed-Forward neural network."""

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
        super(FFNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )
        # Default layers
        self.default_lays_propty = {
            'input': {
                'xform': None
            },
            'output': {
                'xform': None
            },
        }
        # Layers names
        if self.lays_name is None:
            self.lays_name = ('input', 'hidden_block', 'output')
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=('hidden_block',),
            default=self.default_lays_propty
        )
        # Initializing layers
        self.active_lays = self.init_layers()

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier in ('input', 'output'):
                lays[identifier] = self.init_inout_lay(identifier, propty)
            elif identifier == 'hidden_block':
                lays[identifier] = self.get_block(propty=propty)
        return lays

    def init_inout_lay(self, identifier, propty):
        _propty = {}
        for k, v in propty.items():
            if k in ('xform',):
                _propty[k] = v
        return _propty

    # Calling
    ###########################################################################
    def call(self, x, training=False):
        # Inputs manipulation
        x = self._manipulate_inout(x, identifier='input')
        # Kernel
        y = self.active_lays['hidden_block'](x, training=training)
        # Outputs manipulation
        y = self._manipulate_inout(y, identifier='output')
        return y

    def _manipulate_inout(self, y, identifier):
        if self.active_lays[identifier]['xform'] is not None:
            y = self.active_lays[identifier]['xform'](y)
        return y
