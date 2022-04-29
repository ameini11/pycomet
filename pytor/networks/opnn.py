import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.utilities.model import layers
from prode.utilities.model import blocks
from prode.networks.net import Net
from prode.utilities import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='OpNN')
class OpNN(Net):
    """Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        end_scale=None,
        stacked_blocks={"branch": True, "trunk": False},
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        block_linear_out=False,
        shifter=None,
        augmented=False,
        name='opnn',
        *args,
        **kwargs
    ):
        super(OpNN, self).__init__(
            dim=dim,
            block=block,
            block_linear_out=block_linear_out,
            name=name,
            *args,
            **kwargs
        )

        self.sub_nets = ("branch", "trunk")

        ### Hyperparameters Initialization ------------------------------------
        if not isinstance(self.inp_dim, (list, tuple)):
            utils.raise_value_err(
                "`dim['inp']` passed into `{}` has to be a `list` or "
                "a `tuple`of length 2: [u_dim, y_dim]".format(self.name))

        # Layers size
        if not isinstance(layer_size, dict):
            self.layer_size = {k: layer_size for k in self.sub_nets}
        else:
            self.layer_size = layer_size

        # Activation functions
        activation = tf.nest.map_structure(activations.get, activation)
        if not isinstance(activation, dict):
            self.activation = {k: activation for k in self.sub_nets}
        else:
            self.activation = activation

        # Regularization
        if regularization is not None:
            self.regularizer = regularizers.get(**regularization)
        else:
            self.regularizer = regularization

        # Batch Normalization
        if not isinstance(batch_norm, dict):
            self.batch_norm = {k: batch_norm for k in self.sub_nets}
        else:
            self.batch_norm = batch_norm

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.dropout_rate = dropout_rate
        self.weight_norm = weight_norm
        self.stacked_blocks = stacked_blocks

        # Augmentation
        self.augmented = augmented
        if self.augmented and self.block.name != 'cross_encoded_block':
            self.block = utils.get_class(blocks, 'CrossEncodedBlock')

        # Shifter
        self.shifter = shifter

        ### Final scaling ----------------------------------------------------
        if end_scale not in {None, "bias", "dense"}:
            utils.raise_value_err(
                f"Invalid `end_scale` argument passed into `{self.name}`. "
                f"Available choices: {{None|'bias'|'dense'}}."
            )
        self.end_scale = end_scale

        ### Blocks Initialization ---------------------------------------------
        # Branch/Trunk
        self.init_blocks()
        # Shifter
        if self.shifter is not None:
            self.init_shifter()
        # Augmentation
        if self.augmented:
            self.init_uv()
        # Final scaling
        if self.end_scale == 'bias':
            self.end_scale_lay = layers.Bias(name='end_scale')
        elif self.end_scale == 'dense':
            self.end_scale_lay = tf_layers.Dense(
                units=self.out_dim,
                kernel_initializer=self.kernel_initializer,
                name='end_scale'
            )

    # Blocks Initialization
    ###########################################################################
    def init_blocks(self):
        shared_args = dict(
            kernel_initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            weight_norm=self.weight_norm,
            linear_out=self.block_linear_out
        )
        args = {
            "branch": {
                **shared_args,
                **dict(
                    activation=self.activation["branch"],
                    batch_norm=self.batch_norm["branch"]
                )
            },
            "trunk": {
                **shared_args,
                **dict(
                    activation=self.activation["trunk"],
                    batch_norm=self.batch_norm["trunk"]
                )
            }
        }
        # Branches
        self.n_branches = self.out_dim if self.stacked_blocks["branch"] else 1
        self._branches = [ self.block(
            **args["branch"],
            layer_size=self.layer_size["branch"],
            name=f'branch_{i}' if i != 0 else 'branch'
        ) for i in range(self.n_branches) ]
        # Trunks
        self.n_trunks = self.out_dim if self.stacked_blocks["trunk"] else 1
        self._trunks = [ self.block(
            **args["trunk"],
            layer_size=self.layer_size["trunk"],
            name=f'trunk_{i}' if i != 0 else 'trunk'
        ) for i in range(self.n_trunks) ]

    def init_shifter(self):
        # Check 'shifter' network
        if isinstance(self.shifter, dict):
            for k in ('layer_size', 'activation', 'stacked'):
                if k not in self.shifter:
                    utils.raise_value_err(
                        f"'{k}' missing in `shifter` dictionary"
                    )
            if any(k not in ('layer_size', 'activation', 'stacked') \
                for k in self.shifter.keys()):
                utils.raise_value_err(
                    f"Invalid `shifter` dictionary keys passed into "
                    f"`{self.name}`. Available choices: "
                    "{{'layer_size'|'activation'|'stacked'}}."
                )
        else:
            utils.raise_value_err(
                f"Input `shifter` has to be a dictionary with "
                f"keys: {{'layer_size'|'activation'|'stacked'}}."
            )
        # Shifter block
        n = 2 if self.shifter['stacked'] else 1
        self._shifter = [utils.get_class(blocks, 'FFBlock')(
            kernel_initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            weight_norm=self.weight_norm,
            linear_out=True,
            activation=self.shifter["activation"],
            layer_size=self.shifter["layer_size"],
            name=f'shifter_{i}'# if i != 0 else 'shifter'
        ) for i in range(n)]

    def init_uv(self):
        # Augmented cross-encoding layers
        self.U = blocks.MultiLayerBlock(
            units=self.layer_size["branch"][1],
            activation=self.activation["branch"],
            kernel_initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            weight_norm=self.weight_norm,
            batch_norm=self.batch_norm["branch"],
            dropout_rate=self.dropout_rate,
            name='U'
        )
        self.V = blocks.MultiLayerBlock(
            units=self.layer_size["trunk"][1],
            activation=self.activation["trunk"],
            kernel_initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            weight_norm=self.weight_norm,
            batch_norm=self.batch_norm["trunk"],
            dropout_rate=self.dropout_rate,
            name='V'
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'activation': self.activation,
            'sub_nets': self.sub_nets,
            'layer_size': self.layer_size,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'end_scale': self.end_scale,
            'shifter': self.shifter,
            'augmented': self.augmented
        }
        base_config = super(OpNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
        if isinstance(x, (list, tuple)):
            x = {k: x[i] for i, k in enumerate(self.sub_nets)}
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))
        if self.inp_trans is not None:
            for k in self.sub_nets:
                if isinstance(self.inp_trans, dict):
                    x[k] = self.inp_trans[k](x[k])
                else:
                    x[k] = self.inp_trans(x[k])
        return x

    def _eval_kernel(self, x, training):
        # Shifter
        if self.shifter is not None:
            if self.shifter['stacked']:
                a = self._shifter[0](x["branch"], training=training)
                b = self._shifter[1](x["branch"], training=training)
            else:
                s = self._shifter[0](x["branch"], training=training)
                a, b = tf.split(s, 2, axis=1)
            x["trunk"] = tf_layers.Add()(
                [tf_layers.Multiply()([a, x["trunk"]]), b]
            )
        # Augmentation
        args = dict(training=training)
        if self.augmented:
            u = self.U(x["branch"], training=training)
            v = self.V(x["trunk"], training=training)
            args.update(dict(u=u, v=v))
        # Kernel - OpNN
        G_u = [br(x["branch"], **args) for br in self._branches]
        G_y = [tr(x["trunk"], **args) for tr in self._trunks]
        return G_u, G_y

    def _dot_product(self, G_u, G_y):
        if self.stacked_blocks["branch"] or self.stacked_blocks["trunk"]:
            n = max(self.n_branches, self.n_trunks)
            if self.n_branches == self.n_trunks:
                G_uy = [
                    tf_layers.Dot(axes=1)([G_u[i], G_y[i]]) for i in range(n)
                ]
            elif self.n_branches > self.n_trunks:
                G_uy = [
                    tf_layers.Dot(axes=1)([G_u[i], G_y[-1]]) for i in range(n)
                ]
            elif self.n_branches < self.n_trunks:
                G_uy = [
                    tf_layers.Dot(axes=1)([G_u[-1], G_y[i]]) for i in range(n)
                ]
        else:
            G_uy = tf.split(
                tf_layers.Dot(axes=1)([G_u[-1], G_y[-1]]), self.out_dim, axis=1
            )
            G_uy = [tf.math.reduce_sum(g, axis=1, keepdims=True) for g in G_uy]
        return G_uy

    def _manipulate_output(self, y):
        # Concatenation
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        # End scaling
        if self.end_scale is not None:
            y = self.end_scale_lay(y)
        # Transformation
        if self.out_trans is not None:
            y = self.out_trans(y)
        return y
