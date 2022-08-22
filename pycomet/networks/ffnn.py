import tensorflow as tf

from pycomet.utilities.model import blocks
from pycomet.networks.net import Net
from pycomet.utilities import utils


@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='FFNN')
class FFNN(Net):
    """Feed-Forward neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_propty,
        *args,
        **kwargs
    ):
        super(FFNN, self).__init__(
            dim=dim,
            *args,
            **kwargs
        )
        # Layers
        self.lays_name = ('input', 'hidden_block', 'output')
        self.lays_propty = self.check_layers(
            layers=lays_propty, not_opt=['hidden_block']
        )
        # Initializing
        self.init_layers()

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        for identifier in self.lays_name:
            if identifier in ('input', 'output'):
                self.init_inout_lay(identifier)
            elif identifier == 'hidden_block':
                self.init_block_lay(identifier)

    def init_inout_lay(self, identifier):
        # Initializing default properties
        _propty = {
            'xform': None
        }
        if self.lays_propty[identifier] is not None:
            # Initializing layer
            propty = self.lays_propty[identifier]
            for k, v in propty.items():
                if k in ('xform',):
                    _propty[k] = v
        self.active_lays[identifier] = _propty

    def init_block_lay(self, identifier):
        # Initializing properties
        propty = self.lays_propty[identifier]
        # Initializing block
        blk = 'FFBlock' if 'block_type' not in propty else propty['block_type']
        blk = blk if callable(blk) else utils.get_class(blocks, blk)
        if 'block_type' in propty:
            propty.pop('block_type')
        self.active_lays[identifier] = blk(**propty)

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'lays_name': self.lays_name,
            'lays_propty': self.lays_propty
        }
        base_config = super(FFNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
