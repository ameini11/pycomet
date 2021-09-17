import tensorflow as tf
import tensorflow.keras.layers as tf_layers

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

        if isinstance(layer_size, dict):
            self.layer_size_branch = layer_size["branch"]
            self.layer_size_trunk = layer_size["trunk"]
        else:
            self.layer_size_branch = self.layer_size_trunk = layer_size

        if isinstance(activation, dict):
            self.act_branch = activations.get(activation["branch"])
            self.act_trunk = activations.get(activation["trunk"])
        else:
            self.act_branch = self.act_trunk = activations.get(activation)

        if regularization is not None:
            self.regularizer = regularizers.get(**regularization)
        else:
            self.regularizer = regularization

        if isinstance(batch_norm, dict):
            self.batch_norm_branch = batch_norm["branch"]
            self.batch_norm_trunk  = batch_norm["trunk"]
        else:
            self.batch_norm_branch = self.batch_norm_trunk = batch_norm

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.dropout_rate = dropout_rate
        self.weight_norm = weight_norm
        self.stacked = stacked

        if end_scale not in {None, "bias", "dense"}:
            utils.raise_value_err("Invalid `end_scale` argument "
                "passed into `{}`. Available choices: {{None|'bias'|"
                "'dense'}}.".format(self.name))
        self.end_scale = end_scale

        ### Blocks Initialization ---------------------------------------------
        num_branches = self.out_dim if self.stacked else 1
        self._branches = [ self.block(
            self.layer_size_branch,
            self.act_branch,
            self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm_branch,
            weight_norm=self.weight_norm,
            name='branch_' + str(i) if i != 0 else 'branch'
        ) for i in range(num_branches) ]

        self._trunk = self.block(
            self.layer_size_trunk,
            self.act_trunk,
            self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm_trunk,
            weight_norm=self.weight_norm,
            name='trunk'
        )

        if self.end_scale == 'bias':
            self._end_scale = layers.BiasLay(name='end_scale')
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
            'layer_size_branch': self.layer_size_branch,
            'layer_size_trunk': self.layer_size_trunk,
            'act_branch': self.act_branch,
            'act_trunk': self.act_trunk,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'batch_norm_branch': self.batch_norm_branch,
            'batch_norm_trunk': self.batch_norm_trunk,
            'weight_norm': self.weight_norm,
            'end_scale': self.end_scale
        }
        base_config = super(OpNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x, training=None):

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

        # Branch/-es
        G_u = [ branch(x_u, training=training) for branch in self._branches ]

        # Trunk
        G_y = self._trunk(x_y, training=training)

        # Output layer - Dot product
        if self.stacked:
            G_uy = [ tf_layers.Dot(axes=1)([G_u_i, G_y]) for G_u_i in G_u ]
        else:
            G_uy = tf.split(tf_layers.Multiply()([*G_u, G_y]), self.out_dim, axis=1)
            # G_uy = tf.split(tf.math.multiply(*G_u, G_y), self.out_dim, axis=1)
            G_uy = [tf.math.reduce_sum(g, axis=1, keepdims=True) for g in G_uy]

        # Output layer - Concatenation
        G_uy = tf_layers.Concatenate()(G_uy) if len(G_uy) > 1 else G_uy[-1]
        if self.end_scale != None:
            G_uy = self._end_scale(G_uy)

        # Output transformation
        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)
        
        return G_uy


    # Graph
    ###########################################################################
    def get_graph(self):
        x = [ tf.keras.Input(shape=(self.inp_dim[0],), name="branch_input"),
              tf.keras.Input(shape=(self.inp_dim[1],), name="trunk_input") ]
        y = self.call(x)
        return tf.keras.Model(inputs=x, outputs=y, name=self.name)
