import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.utilities.model import layers
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
        stacked=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        cross_encoder=False,
        cross_encoder_layer_size=None,
        name='opnn',
        *args,
        **kwargs
    ):
        super(OpNN, self).__init__(
            dim=dim,
            block=block,
            name=name,
            *args,
            **kwargs
        )

        ### Hyperparameters Initialization ------------------------------------
        if not isinstance(self.inp_dim, (list, tuple)):
            utils.raise_value_err(
                "`dim['inp']` passed into `{}` has to be a `list` or "
                "a `tuple`of length 2: [u_dim, y_dim]".format(self.name))

        # Layers size
        if not isinstance(layer_size, dict):
            self.layer_size = {
                "branch": layer_size,
                "trunk": layer_size
            }

        # Activation functions
        act = tf.nest.map_structure(activations.get, activation)
        if not isinstance(act, dict):
            self.act = {
                "branch": act,
                "trunk": act
            }

        # Regularization
        if regularization is not None:
            self.regularizer = regularizers.get(**regularization)
        else:
            self.regularizer = regularization

        # Batch Normalization
        if not isinstance(batch_norm, dict):
            self.batch_norm = {
                "branch": batch_norm,
                "trunk": batch_norm
            }

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.dropout_rate = dropout_rate
        self.weight_norm = weight_norm
        self.stacked = stacked

        # Cross-encoding between branch and trunk
        if isinstance(cross_encoder, dict):
            if any(k not in {"branch", "trunk"} for k in cross_encoder.keys()):
                utils.raise_value_err(
                    f"Invalid `cross_encoder` dictionary "
                    f"keys passed into into `{self.name}`. "
                    f"Available choices: {{'branch'|'trunk'}}."
                )
            else:
                self.cross_encoder = cross_encoder
        elif isinstance(cross_encoder, bool):
            self.cross_encoder = {
                "branch": cross_encoder,
                "trunk": cross_encoder
            }
        else:
            utils.raise_value_err(
                f"Invalid `cross_encoder` argument passed into `{self.name}`. "
                f"It has to be a `bool` or a `dict` of `bool`."
            )

        self.cross_encoder_layer_size = {}
        if isinstance(cross_encoder_layer_size, dict):
            if any(k not in {"branch", "trunk"} for k in cross_encoder.keys()):
                utils.raise_value_err(
                    f"Invalid `cross_encoder_layer_size` dictionary "
                    f"keys passed into into `{self.name}`. "
                    f"Available choices: {{'branch'|'trunk'}}."
                )
            else:
                self.cross_encoder_layer_size = cross_encoder_layer_size
        elif isinstance(cross_encoder_layer_size, (list, tuple)):
            for k, v in self.cross_encoder.items():
                if v:
                    self.cross_encoder_layer_size[k] = cross_encoder_layer_size
        else:
            k_rev = list(self.cross_encoder.keys()).reverse()
            for i, (k, v) in enumerate(self.cross_encoder.items()):
                if v:
                    l = self.layer_size[k]
                    l[-1] = self.layer_size[k_rev[i]][0]*2
                    self.cross_encoder_layer_size[k] = l

        # Final scaling
        if end_scale not in {None, "bias", "dense"}:
            utils.raise_value_err(
                f"Invalid `end_scale` argument passed into `{self.name}`. "
                f"Available choices: {{None|'bias'|'dense'}}."
            )
        self.end_scale = end_scale

        ### Blocks Initialization ---------------------------------------------
        # Branches
        num_branches = self.out_dim if self.stacked else 1
        self._branches = [ self.block(
            self.layer_size["branch"],
            self.act["branch"],
            self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm["branch"],
            weight_norm=self.weight_norm,
            name='branch_' + str(i) if i != 0 else 'branch'
        ) for i in range(num_branches) ]

        # Trunk
        self._trunk = self.block(
            self.layer_size["trunk"],
            self.act["trunk"],
            self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm["trunk"],
            weight_norm=self.weight_norm,
            name='trunk'
        )

        # Cross-encoding blocks
        self._cross_encoder = {}
        # Branch encoder
        if self.cross_encoder["branch"]:
            self._cross_encoder["branch"] = self.block(
                self.cross_encoder_layer_size["branch"],
                self.act["branch"],
                self.kernel_initializer,
                regularizer=self.regularizer,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm["branch"],
                weight_norm=self.weight_norm,
                name='cross_encoder_branch'
            )
        # Trunk encoder
        if self.cross_encoder["trunk"]:
            self._cross_encoder["trunk"] = self.block(
                self.cross_encoder_layer_size["trunk"],
                self.act["trunk"],
                self.kernel_initializer,
                regularizer=self.regularizer,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm["trunk"],
                weight_norm=self.weight_norm,
                name='cross_encoder_trunk'
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
            'layer_size': self.layer_size,
            'act': self.act,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'end_scale': self.end_scale,
            'cross_encoder': self.cross_encoder,
            'cross_encoder_layer_size': self.cross_encoder_layer_size
        }
        base_config = super(OpNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False, mask=None):

        # Inputs
        if isinstance(x, (list, tuple)):
            x_u, x_y = x[0], x[1]
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))

        # Input transformation
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                x_u = self.inp_trans["branch"](x_u)
                x_y = self.inp_trans["trunk"](x_y)
            else:
                x_u = self.inp_trans(x_u)
                x_y = self.inp_trans(x_y)

        # Cross-encoding
        if self.cross_encoder["branch"]:
            br_encoder = tf.split(
                self._cross_encoder["branch"](x_u), 2, axis=1
            )
            x_y = tf_layers.Multiply()([br_encoder[0], x_y]) + br_encoder[1]
        if self.cross_encoder["trunk"]:
            tr_encoder = tf.split(
                self._cross_encoder["trunk"](x_y), 2, axis=1
            )
            x_u = tf_layers.Multiply()([tr_encoder[0], x_u]) + tr_encoder[1]

        # Branch/-es
        G_u = [ branch(x_u, training=training) for branch in self._branches ]

        # Trunk
        G_y = self._trunk(x_y, training=training)

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
