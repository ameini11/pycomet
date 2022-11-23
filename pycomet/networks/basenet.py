import numpy as np
import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks import chemreacnn
from pycomet.networks import elements as nn_elem
from pycomet import utils


class BaseNet(object):
    """Base Network Layer base class."""

    # Initialization
    ###########################################################################
    def __init__( 
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        name=None,
        force_siso=False,
        *args,
        **kwargs
    ):
        super(BaseNet, self).__init__(
            *args,
            **kwargs
        )
        # Name
        self.name = self.__class__.__name__.lower()
        if name is not None:
            self.name = '_'.join((self.name, name))

        # Dimensions
        self.dim = dim
        if not isinstance(self.dim, dict):
            utils.raise_value_err(
                f"`dim` passed into `{self.name}` network has "
                "to be a `dict` of length 2:\n"
                "  - 'inp': `int` (or `list` of `int`);\n"
                "  - 'out': `int` (or `list` of `int`).\n"
                "It represents the dimensions of the I/O feature space."
            )
        self.input_dim = tf.nest.flatten(self.dim["inp"])
        self.output_dim = tf.nest.flatten(self.dim["out"])
        
        # Force single-input single-output
        self.force_siso = force_siso
        if self.force_siso:
            self.call = self.make_siso_call(self.call, self.input_dim)
            self.input_dim = [np.sum(self.input_dim)]
            self.output_dim = [np.sum(self.output_dim)]
            self.name = self.name + '_siso'

        # Layers
        self.lays_name = lays_name
        if self.lays_name == None:
            self.lays_name = ('preinput', 'input', 'output')
        self.lays_propty = lays_propty
        self.active_lays = {}

        # Default layers
        self.default_lays_propty = {
            'preinput': {
                'xform': None
            },
            'input': {
                'xform': None
            },
            'output': {
                'xform': None
            }
        }

    # Layers initialization
    ###########################################################################
    def init_inout_lay(self, propty):
        _propty = {}
        for k in ('xform', 'with_inputs'):
            if k in propty and propty[k] is not None:
                if k == 'xform':
                    if callable(propty[k]):
                        _propty[k] = propty[k]
                    else:
                        _propty[k] = self.get_lay(
                            propty=propty[k],
                            default_lay_id='Lambda'
                        )
                else:
                    _propty[k] = propty[k]
            else:
                _propty[k] = None
        return _propty
    
    # def init_inout_lay(self, identifier, propty):
    #     k = 'xform'
    #     _propty = {}
    #     if k in propty and propty[k] is not None:
    #         if callable(propty[k]):
    #             _propty[k] = propty[k]
    #         else:
    #             _propty[k] = self.get_lay(
    #                 propty=propty[k],
    #                 default_lay_id='Lambda'
    #             )
    #         _propty['signature'] = utils.get_fun_args(_propty[k].call)
    #         if not any([k in ('x', 'y') for k in _propty['signature']
    #         ]):
    #             utils.raise_value_err(
    #                 f"The function signature passed into {self.name} accepts "
    #                 "at least 'x', 'y', or both as valid options, where 'x' "
    #                 "<- model inputs and 'y' <- intermediate layer/model ouputs."
    #             )
    #     else:
    #         _propty[k] = None
    #     return _propty

    def init_preinput_lay(self):
        self.active_lays['preinput'] = self.init_inout_lay(
            self.lays_propty['preinput']
        )

    # Calling
    ###########################################################################
    def __call__(self, x, training=None):
        return self.call(x, training)
    
    def call(self, x, training=None):
        y = x
        # Pre-inputs manipulation
        y = self._manipulate_inout(
            inputs=dict(x=x, y=y), identifier='preinput', training=training
        )
        return self.call_net(y, training)

    def call_net(self, x, training=None):
        return x

    def _manipulate_inout(self, inputs, identifier=None, training=False):
        lay = self.active_lays[identifier]
        if lay['xform'] is not None:
            if lay['with_inputs'] is None or not lay['with_inputs']:
                _inputs = inputs['y']
            else:
                _inputs = [inputs[k] for k in ('x', 'y')]
            return lay['xform'](_inputs, training=training)
        else:
            return inputs['x'] if identifier in ('preinput', 'input') else inputs['y']
        
    # def _manipulate_inout(
    #     self, x=None, y=None, identifier=None, training=False
    # ):
    #     xform = self.active_lays[identifier]['xform']
    #     if xform is not None:
    #         if 'x' in self.active_lays[identifier]['signature']:
    #             y = xform(x=x, y=y, training=training)
    #         else:
    #             y = xform(y, training=training)
    #     return y

    def make_siso_call(self, call_fun, input_dim):
        def call(x, training=None):
            x = nn_elem.Split(input_dim, axis=1)(x)
            y = call_fun(x, training)
            y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
            return y
        return call

    # Handling layers
    ###########################################################################
    def check_layers(
        self, layers={}, not_opt=[], default={}, check_valid=True
    ):
        # Check input layers format
        if not isinstance(layers, dict):
            utils.raise_value_err(
                "`lays_propty` input passed into "
                f"`{self.name}` has to be a `dict`."
            )
        # Check valid layers
        if check_valid:
            for k in layers.keys():
                if k not in self.lays_name:
                    utils.raise_value_err(
                        f"Layer '{k}' in `lays_propty` input not recognized. "
                        f"Layers available: {self.lays_name}"
                    )
        # Check not optional layers
        for k in not_opt:
            if k not in layers.keys():
                utils.raise_value_err(
                    f"'{k}' layer have to be provided in `lays_propty` input."
                )
        # Add missing optional layers with default values
        for k in self.lays_name:
            if k not in layers.keys() and k in default.keys():
                layers[k] = default[k]
        return layers

    def get_nn_elem(self, elem):
        return utils.get_class(modules=(nn_elem, chemreacnn, tf_layers), name=elem)

    def get_lay(self, propty={}, default_lay_id='Lambda', name=None):
        lay = default_lay_id if 'lay_id' not in propty else propty['lay_id']
        lay = lay if callable(lay) else self.get_nn_elem(lay)
        if name is not None:
            propty['kwargs'].update(
                {'name': '_'.join((name, lay.__name__.lower()))}
            )
        return lay(**propty['kwargs'])