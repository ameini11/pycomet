import types
import numpy as np
import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks.basenet import BaseNet
from pycomet import config as cfg
from pycomet import networks
from pycomet import utils


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MMOpNN')
class MMOpNN(BaseNet):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        default_lays_propty=None,
        not_opt_lays=None,
        *args,
        **kwargs
    ):
        super(MMOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )
        # Layers --------------------------------------------------------------
        # Default layers
        self.default_lays_propty = default_lays_propty
        if self.default_lays_propty is None:
            self.default_lays_propty = {
                'input': {
                    'branch': {
                        'index': 0,
                        'xform': None
                    },
                    'trunk': {
                        'index': 1,
                        'xform': None
                    }
                },
                'output': {
                    'xform': None
                }
            }
        # Layers names
        if self.lays_name is None:
            self.lays_name = ('input', 'opnns', 'output')
        self.not_opt_lays = not_opt_lays
        if self.not_opt_lays is None:
            self.not_opt_lays = ('opnns',)
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=self.not_opt_lays,
            default=self.default_lays_propty
        )
        # Initializing
        self.active_lays = self.init_layers()

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                lays[identifier] = self.init_input_lay(identifier, propty)
            elif identifier == 'opnns':
                lays[identifier] = self.init_opnns_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_output_lay(identifier, propty)
        return lays

    def init_input_lay(self, identifier, propty):
        # Checking properties format
        if not isinstance(propty, dict):
            utils.raise_value_err(
                f"Layer '{identifier}' in 'lays_propty' input has "
                f"to be a `dict` with names of the inputs as keys."
            )
        # Initializing layer properties
        _propty = {}
        for i, (k_subnet, v_subnet) in enumerate(propty.items()):
            _propty[k_subnet] = {}
            for k in ('index', 'xform'):
                if k in v_subnet:
                    _propty[k_subnet][k] = v_subnet[k]
                elif k == 'index':
                    _propty[k_subnet][k] = i
                elif k == 'xform':
                    _propty[k_subnet][k] = None
        indices = [_propty[k]['index'] for k in _propty.keys()]
        if len(np.unique(indices)) != len(indices):
            utils.raise_value_err(
                f"Key 'index' in layer '{identifier}' has "
                "to be different between each subnet."
            )
        return _propty

    def init_opnns_lay(self, identifier, propty):
        if not isinstance(propty, dict):
            utils.raise_value_err(
                "The properties for 'opnns' layer has to be provided "
                "as a dictionary of (opnn_name, opnn_args) pairs."
            )
        return {name: self.get_opnn(opnn_lays_propty, name) \
            for name, opnn_lays_propty in propty.items()}

    def get_opnn(self, propty={}, name=None):
        nn = 'OpNN' if 'lay_id' not in propty else propty['lay_id']
        nn = nn if callable(nn) else utils.get_class(networks, nn)
        name = nn.__name__.lower() + '_' + name
        lays_propty = propty if 'lays_propty' not in propty \
            else propty['lays_propty']
        return nn(
            dim=propty['dim'],
            lays_propty=propty['lays_propty'],
            name=name
        )

    def init_output_lay(self, identifier, propty):
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
        y = self._manipulate_input(x)
        # Kernel - OpNNs
        y = [opnn(y, training=training) \
             for opnn in self.active_lays['opnns'].values()]
        # Outputs manipulation
        y = self._manipulate_output(x, y, training)
        return y

    def _manipulate_input(self, x):
        if not isinstance(x, (list, tuple)):
            utils.raise_value_err(
                f"The inputs of the `{self.name}` net "
                "have to be a a `list` or a `tuple`."
            )
        x = {k: x[v['index']] for k, v in self.active_lays['input'].items()}
        for k, v in self.active_lays['input'].items():
            if v['xform'] is not None:
                x[k] = v['xform'](x[k])
        return x

    def _manipulate_output(self, x, y, training):
        if self.active_lays['output']['xform'] is not None:
            lay = self.active_lays['output']['xform']
            inputs = {'x': x, 'y': y}
            y = lay['fn'](
                inputs={k: inputs[k] for k in lay['args']},
                training=training
            )
        return y
