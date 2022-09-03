import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from pycomet.networks.elements import layers as pcm_layers
from pycomet.networks.net import Net
from pycomet import config as cfg
from pycomet import utils


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='OpNN')
class OpNN(Net):
    """Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_propty,
        *args,
        **kwargs
    ):
        super(OpNN, self).__init__(
            dim=dim,
            *args,
            **kwargs
        )
        # Input dimensionality ------------------------------------------------
        if not isinstance(self.inputs_dim, (list, tuple)):
            utils.raise_value_err(
                f"`dim['inp']` passed into `{self.name}` has to be a "
                "`list` or a `tuple`of length 2: [u_dim, y_dim]"
            )

        # Layers --------------------------------------------------------------
        self.lays_name = (
            'input', 'prenet', 'branch', 'trunk', 'end_scale', 'output'
        )
        self.lays_propty = self.check_layers(
            layers=lays_propty, not_opt=['branch', 'trunk']
        )
        # Initializing
        self.init_layers()
        self.n_branches = len(self.active_lays['branch'])
        self.n_trunks = len(self.active_lays['trunk'])

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        for identifier in self.lays_name:
            if identifier == 'input':
                self.init_input_lay(identifier)
            elif identifier in ('prenet', 'branch', 'trunk'):
                self.init_block_lay(identifier)
            elif identifier == 'end_scale':
                self.init_end_scale_lay(identifier)
            elif identifier == 'output':
                self.init_output_lay(identifier)

    def init_input_lay(self, identifier):
        # Initializing default properties
        _propty = {
            'branch': {
                'index': 0,
                'xform': None
            },
            'trunk': {
                'index': 1,
                'xform': None
            }
        }
        if identifier in self.lays_propty.keys():
            # Checking properties format
            propty = self.lays_propty[identifier]
            if not isinstance(propty, dict) \
                or any(k not in ('branch', 'trunk') for k in propty.keys()):
                utils.raise_value_err(
                    f"Layer '{identifier}' in 'lays_propty' input has to be a "
                    f"`dict` with keys: ('branch', 'trunk')"
                )
            # Initializing layer
            for k_subnet, v_subnet in propty.items():
                for k, v in v_subnet.items():
                    if k in ('index', 'xform'):
                        if k == 'index' and v > 1:
                            utils.raise_value_err(
                                f"Key 'index' in layer '{identifier}' "
                                "has to be 0 or 1."
                            )
                        _propty[k_subnet][k] = v
            if _propty['branch']['index'] == _propty['trunk']['index']:
                utils.raise_value_err(
                    f"Key 'index' in layer '{identifier}' has to be "
                    "different between the subnets."
                )
        self.active_lays[identifier] = _propty

    def init_block_lay(self, identifier):
        if identifier in self.lays_propty.keys():
            # Initializing properties
            propty = self.lays_propty[identifier]
            propty = propty if isinstance(propty, (list,tuple)) else [propty]
            # Initializing layer
            self.active_lays[identifier] = []
            for i, p in enumerate(propty):
                name = f'{identifier}' if i == 0 else f'{identifier}_{i}'
                self.active_lays[identifier].append(
                    self.get_block(propty=p, name=name)
                )

    def init_end_scale_lay(self, identifier):
        if identifier in self.lays_propty.keys():
            # Initializing default properties
            _propty = {
                'type': 'bias'
            }
            # Initializing layer
            propty = self.lays_propty[identifier]
            if 'type' in propty.keys():
                typ = propty.pop('type')
                if typ == 'bias':
                    self.active_lays[identifier] = pcm_layers.Bias(
                        **propty,
                        name=identifier
                    )
                elif typ == 'dense':
                    self.active_lays[identifier] = tf_layers.Dense(
                        **propty,
                        units=self.outputs_dim,
                        name=identifier
                    )
                else:
                    utils.raise_value_err(
                        f"Value for key 'type' in layer '{identifier}' not "
                        f"recognized. Values available: ('bias', 'dense')"
                    )

    def init_output_lay(self, identifier):
        # Initializing default properties
        _propty = {
            'xform': None
        }
        if identifier in self.lays_propty.keys():
            # Initializing layer
            propty = self.lays_propty[identifier]
            for k, v in propty.items():
                if k in ('xform',):
                    _propty[k] = v
        self.active_lays[identifier] = _propty

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'lays_propty': self.lays_propty,
            'lays_name': self.lays_name
        }
        base_config = super(OpNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # Calling
    ###########################################################################
    def call(self, x, training=False):
        # Inputs manipulation
        x = self._manipulate_input(x)
        # Kernel - OpNN
        G_u, G_y = self._eval_kernel(x, training)
        # Outputs manipulation
        y = self._dot_product(G_u, G_y)
        y = self._manipulate_output(y)
        return y

    def _manipulate_input(self, x):
        if not isinstance(x, (list, tuple)):
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))
        x = {k: x[v['index']] for k, v in self.active_lays['input'].items()}
        for k, v in self.active_lays['input'].items():
            if v['xform'] is not None:
                x[k] = v['xform'](x[k])
        return x

    def _eval_kernel(self, x, training):
        # PreNet
        if 'prenet' in self.active_lays.keys():
            x["trunk"] = self.active_lays['prenet'][-1](
                x=[x["branch"], x["trunk"]], training=training
            )
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

    def _dot_product(self, G_u, G_y):
        if self.outputs_dim > 1:
            if self.n_branches > self.n_trunks:
                G_y = [G_y[-1]] * self.n_branches
            elif self.n_branches < self.n_trunks:
                G_u = [G_u[-1]] * self.n_trunks
            elif self.n_branches == self.n_trunks == 1:
                G_u = tf.split(G_u[-1], self.outputs_dim, axis=1)
                G_y = tf.split(G_y[-1], self.outputs_dim, axis=1)
        return [
            tf_layers.Dot(axes=1)([G_u[i], G_y[i]]) \
                for i in range(self.outputs_dim)
        ]

    def _manipulate_output(self, y):
        # Concatenation
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        # End scaling
        if 'end_scale' in self.active_lays.keys():
            y = self.active_lays['end_scale'](y)
        # Transformation
        if self.active_lays['output']['xform'] is not None:
            y = self.active_lays['output']['xform'](y)
        return y
