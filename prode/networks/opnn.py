import copy
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

        ### Cross-encoding between branch and trunk ---------------------------
        # Cross-encoding block
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
            self.cross_encoder = {k: cross_encoder for k in self.sub_nets}
        else:
            utils.raise_value_err(
                f"Invalid `cross_encoder` argument passed into `{self.name}`. "
                f"It has to be a `bool` or a `dict` of `bool`."
            )

        # Cross-encoding block layer sizes
        self.ce_layer_size = {}
        if isinstance(cross_encoder_layer_size, dict):
            if any(k not in {"branch", "trunk"} for k in cross_encoder.keys()):
                utils.raise_value_err(
                    f"Invalid `cross_encoder_layer_size` dictionary "
                    f"keys passed into into `{self.name}`. "
                    f"Available choices: {{'branch'|'trunk'}}."
                )
            else:
                self.ce_layer_size = cross_encoder_layer_size
        elif isinstance(cross_encoder_layer_size, (list, tuple)):
            for k, v in self.cross_encoder.items():
                if v:
                    self.ce_layer_size[k] = cross_encoder_layer_size
        else:
            k_rev = list(self.cross_encoder.keys())[::-1]
            for i, (k, v) in enumerate(self.cross_encoder.items()):
                if v:
                    d = self.inp_dim[i-1]
                    self.ce_layer_size[k] = copy.deepcopy(self.layer_size[k])
                    self.ce_layer_size[k][-1] = d * (d + 1)

        # Array manipulation layers
        self.reshape_lay, self.split_dim = {}, {}
        for i, (k, v) in enumerate(self.cross_encoder.items()):
            if v:
                self.reshape_lay[k] = tf_layers.Reshape([self.inp_dim[i-1]]*2)
                self.split_dim[k] = self.inp_dim[i-1]
        self.einsum_lay = tf_layers.Lambda(
            lambda y: tf.einsum('nij,nj->ni', y[0], y[1])
        )

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
            weight_norm=self.weight_norm
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

        # Cross-encoding blocks
        self._cross_encoder = {}
        # Branch encoder
        if self.cross_encoder["branch"]:
            self._cross_encoder["branch"] = self.block(
                **args["branch"],
                layer_size=self.ce_layer_size["branch"],
                name='cross_encoder_branch'
            )
        # Trunk encoder
        if self.cross_encoder["trunk"]:
            self._cross_encoder["trunk"] = self.block(
                **args["trunk"],
                layer_size=self.ce_layer_size["trunk"],
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
            'sub_nets': self.sub_nets,
            'layer_size': self.layer_size,
            'act': self.act,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'end_scale': self.end_scale,
            'cross_encoder': self.cross_encoder,
            'ce_layer_size': self.ce_layer_size
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

        # Cross-encoding
        A, b = {}, {}
        for k in self.sub_nets:
            if self.cross_encoder[k]:
                A[k], b[k] = tf.split(
                    self._cross_encoder[k](x[k]),
                    [self.split_dim[k]**2, self.split_dim[k]],
                    axis=1
                )
                A[k] = self.reshape_lay[k](A[k])
        for k in self.sub_nets:
            if self.cross_encoder[k]:
                j = "branch" if k == "trunk" else "trunk"
                x[j] = tf.keras.layers.Add()(
                    [self.einsum_lay([A[k], x[j]]), b[k]]
                )

        # Branch/-es
        G_u = [
            branch(x["branch"], training=training) for branch in self._branches
        ]

        # Trunk
        G_y = self._trunk(x["trunk"], training=training)

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
