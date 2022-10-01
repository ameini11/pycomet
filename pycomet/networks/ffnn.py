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
        for k in ('xform',):
            if k in propty and propty[k] is not None:
                if k == 'xform' :
                    _propty[k] = {}
                    _propty[k]['fn'] = propty[k]
                    _propty[k]['args'] = utils.get_fun_args(propty[k])
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
    def call(self, x, training=False):
        # Inputs manipulation
        y = self._manipulate_inout(x=x, identifier='input')
        # Kernel
        y = self.active_lays['hidden_block'](y, training=training)
        # Outputs manipulation
        y = self._manipulate_inout(x=x, y=y, identifier='output')
        return y

    def _manipulate_inout(self, x, y=None, identifier=None):
        if self.active_lays[identifier]['xform'] is not None:
            lay = self.active_lays[identifier]['xform']
            y = lay['fn'](**{k: eval(k) for k in lay['args']})
        elif identifier == 'input':
            y = x
        return y
