import numpy as np

from keras import layers as tf_layers
from pycomet.networks import elements as nn_elem
from pycomet.networks.basenet import BaseNet
from pycomet import utils


class OpNN(BaseNet):
    """Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        noisy_io=None,
        inference=None,
        masking=False,
        force_siso=False,
        *args,
        **kwargs
    ):
        super(OpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            noisy_io=noisy_io,
            inference=inference,
            masking=masking,
            force_siso=force_siso,
            *args,
            **kwargs
        )
        # Input dimensionality ------------------------------------------------
        if not isinstance(self.input_dim, (list, tuple)):
            utils.raise_value_err(
                f"`dim['inp']` passed into `{self.name}` has to be a "
                "`list` or a `tuple`of length 2: [u_dim, y_dim]"
            )
        # Layers --------------------------------------------------------------
        # Default layers
        self.default_lays_propty.update({
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
            },
            'end_scale': {
                'lay_id': None,
                'kwargs': {}
            }
        })
        # Layers names
        if lays_name == None:
            self.lays_name = self.lays_name + ('branch', 'trunk', 'end_scale')
        else:
            self.lays_name = lays_name
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=('branch', 'trunk'),
            default=self.default_lays_propty
        )
        # Initializing
        self.active_lays = self.init_layers()
        self.init_preinput_lay()
        self.nb_branches = len(self.active_lays['branch'])
        self.nb_trunks = len(self.active_lays['trunk'])
        max_nb_blocks = max(self.nb_branches, self.nb_trunks)
        # if (max_nb_blocks != 1) \
        #     and (max_nb_blocks != self.output_dim[-1]):
        #     utils.raise_value_err(
        #         "The number of trunks/branches do "
        #         "not match the number of outputs"
        #     )

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                lays[identifier] = _init_input_lay(self, identifier, propty)
            elif identifier in ('branch', 'trunk'):
                lays[identifier] = self.init_block_lay(identifier, propty)
            elif identifier == 'end_scale':
                lays[identifier] = self.init_end_scale_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_inout_lay(propty)
        return lays

    def init_block_lay(self, identifier, propty):
        lay = []
        propty = propty if isinstance(propty, (list,tuple)) else [propty]
        for i, p in enumerate(propty):
            lay.append(
                self.get_lay(propty=p, default_lay_id='FFBlock')
            )
        return lay

    def init_end_scale_lay(self, identifier, propty):
        if propty['lay_id'] is not None:
            if propty['lay_id'] not in ('Bias', 'Dense'):
                utils.raise_value_err(
                    f"Value for key 'lay_id' in layer '{identifier}' not "
                    f"recognized. Values available: ('Bias', 'Dense')"
                )
            if 'kwargs' not in propty:
                propty['kwargs'] = {}
                if propty['lay_id']  == 'Dense':
                    propty['kwargs'] = {
                        'units': self.output_dim[-1],
                        'kernel_initializer': 'GlorotNormal'
                    }
            if 'kernel_initializer' in propty['kwargs']:
                init = propty['kwargs']['kernel_initializer']
                init = nn_elem.initializers.get(init)
                propty['kwargs']['kernel_initializer'] = init
            return self.get_lay(propty)

    # Calling
    ###########################################################################
    def call_net(self, x, training=None):
        y = x
        # Inputs manipulation
        y = _manipulate_input(self, y)
        # Kernel - OpNN
        G_u, G_y = self._eval_kernel(y, training)
        # Outputs manipulation
        y = self._tensor_product(G_u, G_y)
        y = self._manipulate_output(x, y, training)
        return y

    def _eval_kernel(self, x, training):
        # Branch/Trunk
        G_u = [
            br(x["branch"], training=training) \
                for br in self.active_lays['branch']
        ]
        G_y = [
            tr(x["trunk"], training=training) \
                for tr in self.active_lays['trunk']
        ]
        return G_u, G_y

    def _tensor_product(self, G_u, G_y):
        if self.output_dim[-1] > 1:
            if self.nb_branches > self.nb_trunks:
                G_y = [G_y[-1]] * self.nb_branches
            elif self.nb_branches < self.nb_trunks:
                G_u = [G_u[-1]] * self.nb_branches
            elif self.nb_branches == self.nb_trunks == 1:
                G_u = nn_elem.Split(self.output_dim[-1], axis=1)(G_u[-1])
                G_y = nn_elem.Split(self.output_dim[-1], axis=1)(G_y[-1])
        return [tf_layers.Dot(axes=1)([G_u[i], G_y[i]]) \
            for i in range(self.output_dim[-1])]

    def _manipulate_output(self, x, y, training):
        # Concatenation
        if isinstance(y, (list,tuple)):
            y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        # End scaling
        if self.active_lays['end_scale'] is not None:
            y = self.active_lays['end_scale'](y)
        # Transformation
        return self._manipulate_inout(
            inputs=dict(x=x, y=y), identifier='output', training=training
        )
            
# Public util functions
# -----------------------------------------------------------------------------
def _init_input_lay(opnn_cls, identifier, propty):
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
                if k == 'index':
                    _propty[k_subnet][k] = v_subnet[k]
                else:
                    if callable(v_subnet[k]):
                        _propty[k_subnet][k] = v_subnet[k]
                    elif v_subnet[k] is not None:
                        _propty[k_subnet][k] = opnn_cls.get_lay(
                            propty=v_subnet[k],
                            default_lay_id='Lambda'
                        )
                    else:
                        _propty[k_subnet][k] = None
            else:
                _propty[k_subnet][k] = i if k == 'index' else None
    indices = [_propty[k]['index'] for k in _propty.keys()]
    if len(np.unique(indices)) != len(indices):
        utils.raise_value_err(
            f"Key 'index' in layer '{identifier}' has "
            "to be different between each subnet."
        )
    return _propty

def _manipulate_input(opnn_cls, x):
    if not isinstance(x, (list, tuple, dict)):
        utils.raise_value_err(
            f"The inputs of `{opnn_cls.name}` network have to "
            "be one of the following: `list`,`tuple`,`dict`."
        )
    if isinstance(x, (list, tuple)):
        x = {
            k: x[v['index']] for k, v in opnn_cls.active_lays['input'].items()
        }
    # Inputs transformation
    _x = {}
    for k, v in opnn_cls.active_lays['input'].items():
        if v['xform'] is not None:
            _x[k] = v['xform'](x[k])
        else:
            _x[k] = x[k]
    return _x