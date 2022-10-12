import types
import tensorflow as tf

from pycomet.networks.basenet import BaseNet
from pycomet import config as cfg


class FFNN(BaseNet):
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
            'input': None,
            'output': None
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
                lays[identifier] = self.get_lay(
                    propty=propty, default_lay_id='Lambda'
                ) if propty is not None else None
            elif identifier == 'hidden_block':
                lays[identifier] = self.get_lay(
                    propty=propty, default_lay_id='FFBlock'
                )
        return lays

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        # Inputs manipulation
        y = self._manipulate_inout(
            x=x, identifier='input', training=training
        )
        # Kernel
        y = self.active_lays['hidden_block'](y, training=training)
        # Outputs manipulation
        y = self._manipulate_inout(
            x=x, y=y, identifier='output', training=training
        )
        return y

    def _manipulate_inout(self, x, y=None, identifier=None, training=False):
        inputs = x if identifier == 'input' else y
        if self.active_lays[identifier] is not None:
            if 'iolambda' in self.active_lays[identifier].name.lower():
                return self.active_lays[identifier](
                    x=x, y=y, training=training
                )
            else:
                return self.active_lays[identifier](
                    inputs, training=training
                )
        else:
            return inputs
