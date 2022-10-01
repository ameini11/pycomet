import types
import tensorflow as tf

from pycomet.networks.basenet import BaseNet
from pycomet import config as cfg


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='FFNN')
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
        for k in ('xform',):
            if k in propty and propty[k] is not None:
                if k == 'xform' :
                    if not isinstance(propty[k], tf_layers.Lambda):
                        args = utils.get_fun_args(propty[k])
                        fn = tf_layers.Lambda(
                            function=utils.lambda_lay_dec(propty[k])
                        )
                    else:
                        args = utils.get_fun_args(propty[k].function)
                        propty[k].function = utils.lambda_lay_dec(
                            propty[k].function
                        )
                        fn = propty[k]
                    _propty[k] = {'fn': fn, 'args': args}
                    if any([k not in ('x', 'y') for k in _propty[k]['args']]):
                        utils.raise_value_err(
                            "The outputs transformation function, 'xform', "
                            "accept only ('x', 'y') as function signature, "
                            "where 'x' <- inputs and 'y' <- model ouputs."
                        )
                else:
                    _propty[k] = propty[k]
            else:
                _propty[k] = None
        return _propty

    # Calling
    ###########################################################################
    def call(self, x, training=None, mask=None):
        # Inputs manipulation
        y = self._manipulate_inout(x=x, identifier='input', training=training)
        # Kernel
        y = self.active_lays['hidden_block'](y, training=training)
        # Outputs manipulation
        y = self._manipulate_inout(
            x=x, y=y, identifier='output', training=training
        )
        return y

    def _manipulate_inout(self, x, y=None, identifier=None, training=False):
        if self.active_lays[identifier]['xform'] is not None:
            lay = self.active_lays[identifier]['xform']
            inputs = {'x': x, 'y': y}
            y = lay['fn'](
                inputs={k: inputs[k] for k in lay['args']},
                training=training
            )
        elif identifier == 'input':
            y = x
        return y