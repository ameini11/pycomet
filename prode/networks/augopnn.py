import copy
import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.utilities.model import layers
from prode.utilities.model import blocks
from prode.networks.net import Net
from prode.utilities import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='AugOpNN')
class AugOpNN(Net):
    """Augmented Deep Operator neural network."""

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
        stacked=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block_linear_out=True,
        name='opnn',
        *args,
        **kwargs
    ):
        super(AugOpNN, self).__init__(
            dim=dim,
            block='CrossEncodedBlock',
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
        act = tf.nest.map_structure(activations.get, activation)
        if not isinstance(act, dict):
            self.act = {k: act for k in self.sub_nets}
        else:
            self.act = act

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
        self.stacked = stacked

        ### Final scaling ----------------------------------------------------
        if end_scale not in {None, "bias", "dense"}:
            utils.raise_value_err(
                f"Invalid `end_scale` argument passed into `{self.name}`. "
                f"Available choices: {{None|'bias'|'dense'}}."
            )
        self.end_scale = end_scale

        ### Blocks Initialization ---------------------------------------------
        # Arguments
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
                    activation=self.act["branch"],
                    batch_norm=self.batch_norm["branch"],
                )
            },
            "trunk": {
                **shared_args,
                **dict(
                    activation=self.act["trunk"],
                    batch_norm=self.batch_norm["trunk"],
                )
            }
        }

        # Branches
        num_branches = self.out_dim if self.stacked else 1
        self._branches = [ self.block(
            **args["branch"],
            layer_size=self.layer_size["branch"],
            name='branch_' + str(i) if i != 0 else 'branch'
        ) for i in range(num_branches) ]

        # Trunk
        self._trunk = self.block(
            **args["trunk"],
            layer_size=self.layer_size["trunk"],
            name='trunk'
        )

        # Cross-encoding layers
        self.U = blocks.MultiLayerBlock(
            units=self.layer_size["branch"][1],
            activation=self.act["branch"],
            kernel_initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            weight_norm=self.weight_norm,
            batch_norm=self.batch_norm["branch"],
            dropout_rate=self.dropout_rate,
            name='U'
        )

        self.V = blocks.MultiLayerBlock(
            units=self.layer_size["trunk"][1],
            activation=self.act["trunk"],
            kernel_initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            weight_norm=self.weight_norm,
            batch_norm=self.batch_norm["trunk"],
            dropout_rate=self.dropout_rate,
            name='V'
        )

        # Final scaling
        if self.end_scale == 'bias':
            self._end_scale = layers.Bias(name='end_scale')
        elif self.end_scale == 'dense':
            self._end_scale = tf_layers.Dense(
                units=self.out_dim,
                kernel_initializer=self.kernel_initializer,
                name='end_scale'
            )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'sub_nets': self.sub_nets,
            'layer_size': self.layer_size,
            'act': self.act,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'end_scale': self.end_scale
        }
        base_config = super(AugOpNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False, mask=None):

        # Inputs
        if isinstance(x, (list, tuple)):
            x = {k: x[i] for i, k in enumerate(self.sub_nets)}
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))

        # Input transformation
        if self.inp_trans is not None:
            for k in self.sub_nets:
                if isinstance(self.inp_trans, dict):
                    x[k] = self.inp_trans[k](x[k])
                else:
                    x[k] = self.inp_trans(x[k])

        # Cross-encoding layers
        u = self.U(x["branch"], training=training)
        v = self.V(x["trunk"], training=training)

        # Branch/-es
        G_u = [
            branch(
                x["branch"], u, v, training=training
            ) for branch in self._branches
        ]

        # Trunk
        G_y = self._trunk(x["trunk"], u, v, training=training)

        # Output layer - Dot product
        if self.stacked:
            G_uy = [ tf_layers.Dot(axes=1)([G_u_i, G_y]) for G_u_i in G_u ]
        else:
            G_uy = tf.split(
                tf_layers.Dot(axes=1)([*G_u, G_y]), self.out_dim, axis=1
            )
            G_uy = [tf.math.reduce_sum(g, axis=1, keepdims=True) for g in G_uy]

        # Output layer - Concatenation
        G_uy = tf_layers.Concatenate()(G_uy) if len(G_uy) > 1 else G_uy[-1]
        if self.end_scale != None:
            G_uy = self._end_scale(G_uy)

        # Output transformation
        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)
        
        return G_uy
