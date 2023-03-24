import copy
import numpy as np
import tensorflow as tf

from keras import layers as tf_layers
from tensorflow.python.keras import backend
from pycomet import utils
from pycomet.networks import chemreacnn
from pycomet.networks import elements as nn_elem


class BaseNet(object):
    """Base Network Layer base class."""

    # Initialization
    ###########################################################################
    def __init__( 
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        force_siso=False,
        inference=None,
        name=None,
        *args,
        **kwargs
    ):
        super(BaseNet, self).__init__(
            *args,
            **kwargs
        )
        # Name
        # -----------------------------
        self.name = self.__class__.__name__.lower()
        if name is not None:
            self.name = '_'.join((self.name, name))

        # Inference phase
        # -----------------------------
        self.inference = inference
        if self.inference is not None:
            if not self.inference['active']:
                self.inference = None

        # Dimensions
        # -----------------------------
        self.dim = dim
        if self.inference is not None:
            self.dim = self.inference['dim']
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
        # Extra inputs/outputs due to inference phase
        if self.inference is not None:
            self.extra_inout = {}
            for k in ("inp", "out"):
                l1 = len(tf.nest.flatten(self.inference['dim'][k]))
                l2 = len(tf.nest.flatten(dim[k]))
                self.extra_inout[k] = l1-l2
            self.name = self.name + '_infer'

        # Call
        # -----------------------------
        self.call = self._call if self.inference is None else self._call_infer
        # Force single-input single-output
        self.force_siso = force_siso
        if self.force_siso:
            self.call = self.make_siso_call(self.call, self.input_dim)
            self.input_dim = [np.sum(self.input_dim)]
            self.output_dim = [np.sum(self.output_dim)]
            self.name = self.name + '_siso'

        # Layers
        # -----------------------------
        self.lays_name = lays_name
        if self.lays_name is None:
            self.lays_name = ('preinput', 'input', 'output')
        self.lays_propty = copy.deepcopy(lays_propty)
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

        # Inputs
        # -----------------------------
        if self.inference is not None and self.inference['size'] is not None:
            self.inputs = []
            s = self.inference['size']
            for i in self.input_dim:
                self.inputs.append(tf.keras.Input(
                    tensor=tf.zeros((s, i), dtype=backend.floatx())
                ))
        else:
            self.inputs = [tf.keras.Input((i,)) for i in self.input_dim]
        self.inputs = utils.squeeze_list(tf.nest.flatten(self.inputs))
        
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

    def init_preinput_lay(self):
        k = 'preinput'
        self.active_lays[k] = self.init_inout_lay(self.lays_propty[k])

    # Calling
    ###########################################################################
    def __call__(self, x, training=None, *args, **kwargs):
        return self.call(x, training, *args, **kwargs)
    
    # Training phase
    # ---------------------------------
    def _call(self, x, training=None, *args, **kwargs):
        # Pre-inputs manipulation
        y = self._manipulate_inout(
            inputs=dict(y=x), identifier='preinput', training=training
        )
        return self.call_net(y, training, *args, **kwargs)

    def call_net(self, x, training=None, *args, **kwargs):
        return x

    # Inference phase
    # ---------------------------------
    def _call_infer(self, x, training=None, *args, **kwargs):
        n = self.extra_inout["inp"]
        if n == 0:
            return self._call(x, training, *args, **kwargs)
        else:
            # Extract inputs
            x, x_extra = x[:-n], x[-n:]
            # Pre-inputs manipulation
            x = self._manipulate_inout(
                inputs=dict(x=x, y=x), identifier='preinput'
            )
            return self.call_net_infer(
                x, x_extra, training=False, *args, **kwargs
            )

    def call_net_infer(self, x, training=None, *args, **kwargs):
        return self.call_net(x, training=None, *args, **kwargs)

    # Single-Input-Single-Output call
    # ---------------------------------
    def make_siso_call(self, call_fun, input_dim):
        def siso_call(x, training=None):
            x = nn_elem.Split(input_dim, axis=1)(x)
            y = call_fun(x, training)
            y = tf.nest.flatten(y)
            y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
            return y
        return siso_call

    # Sub-operations
    ###########################################################################
    def _manipulate_inout(self, inputs, lay=None, identifier=None, training=None):
        if lay is None:
            lay = self.active_lays[identifier]
        if lay['xform'] is not None:
            if lay['with_inputs'] is None or not lay['with_inputs']:
                y = inputs['y']
            else:
                y = [inputs[k] for k in ('x', 'y')]
            y = lay['xform'](y, training=training)
        else:
            y = inputs['y']
        return y

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
        if 'kwargs' not in propty:
            propty['kwargs'] = {}
        return lay(**propty['kwargs'])
