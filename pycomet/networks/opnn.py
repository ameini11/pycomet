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
        *args,
        **kwargs
    ):
        super(OpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
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
            },
            'end_scale': {
                'lay_id': None,
                'kwargs': {}
            }
        }
        # Layers names
        if self.lays_name is None:
            self.lays_name = (
                'input', 'branch', 'trunk', 'end_scale', 'output'
            )
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=('branch', 'trunk'),
            default=self.default_lays_propty
        )
        # Initializing
        self.active_lays = self.init_layers()
        self.n_branches = len(self.active_lays['branch'])
        self.n_trunks = len(self.active_lays['trunk'])
        self.max_num_blocks = max(self.n_branches, self.n_trunks)

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                lays[identifier] = self.init_input_lay(identifier, propty)
            elif identifier in ('branch', 'trunk'):
                lays[identifier] = self.init_block_lay(identifier, propty)
            elif identifier == 'end_scale':
                lays[identifier] = self.init_end_scale_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_inout_lay(identifier, propty)
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
                    if k == 'index':
                        _propty[k_subnet][k] = v_subnet[k]
                    else:
                        if callable(v_subnet[k]):
                            _propty[k_subnet][k] = v_subnet[k]
                        elif v_subnet[k] is not None:
                            _propty[k_subnet][k] = self.get_lay(
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
            return self.get_lay(propty)

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        y = x
        # Inputs manipulation
        y = self._manipulate_input(y)
        # Kernel - OpNN
        G_u, G_y = self._eval_kernel(y, training)
        # Outputs manipulation
        y = self._tensor_product(G_u, G_y)
        y = self._manipulate_output(x, y, training)
        return y

    def _manipulate_input(self, x):
        if not isinstance(x, (list, tuple, dict)):
            utils.raise_value_err(
                f"The inputs of `{self.name}` network have to "
                "be one of the following: `list`,`tuple`,`dict`."
            )
        elif isinstance(x, (list, tuple)):
            x = {
                k: x[v['index']] for k, v in self.active_lays['input'].items()
            }
        for k, v in self.active_lays['input'].items():
            if v['xform'] is not None:
                x[k] = v['xform'](x[k])
        return x

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
        if self.max_num_blocks > 1:
            if self.n_branches > self.n_trunks:
                G_y = [G_y[-1]] * self.n_branches
            elif self.n_branches < self.n_trunks:
                G_u = [G_u[-1]] * self.n_branches
            elif self.n_branches == self.n_trunks == 1:
                G_u = nn_elem.Split(self.max_num_blocks, axis=1)(G_u[-1])
                G_y = nn_elem.Split(self.max_num_blocks, axis=1)(G_y[-1])
        return [
            tf_layers.Dot(axes=1)([G_u[i], G_y[i]]) \
                for i in range(self.max_num_blocks)
        ]

    def _manipulate_output(self, x, y, training):
        # Concatenation
        if isinstance(y, (list,tuple)):
            y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        # End scaling
        if self.active_lays['end_scale'] is not None:
            y = self.active_lays['end_scale'](y)
        # Transformation
        return super(OpNN, self)._manipulate_inout(
            x=x, y=y, identifier='output', training=training
        )
            
