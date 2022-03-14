import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities.model import layers
from prode.utilities import utils
from prode.networks.opnn import OpNN
from prode.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PrODE', name='MMOpNN')
class MMOpNN(Net):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        names,
        opnn_dim,
        layer_size,
        activation,
        kernel_initializer,
        end_scale=None,
        opnn_inp_trans=None,
        opnn_out_trans=None,
        opnn_end_scale='bias',
        stacked=True,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        block_linear_out=True,
        cross_encoder=False,
        cross_encoder_layer_size=None,
        augmented=False,
        name='mmopnn',
        *args,
        **kwargs
    ):
        super(MMOpNN, self).__init__(
            dim=dim,
            block=block,
            block_linear_out=block_linear_out,
            name=name,
            *args,
            **kwargs
        )

        self.sub_nets = ("branch", "trunk")

        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### OpNNs Initialization ----------------------------------------------
        if len(names) <= 1:
            utils.raise_value_err(
                "The number of `names` passed into `{}` has to be "
                "grater than 1 (multiple DeepONets).".format(self.name))
        else:
            self.names = names
        self.opnn_dim = opnn_dim
        self.opnn_inp_trans = opnn_inp_trans
        self.opnn_out_trans = opnn_out_trans
        self.opnn_end_scale = opnn_end_scale
        self.stacked = stacked
        self.cross_encoder = cross_encoder
        self.cross_encoder_layer_size = cross_encoder_layer_size
        self.augmented = augmented

        self.opnns = {
            name: OpNN(
                self.opnn_dim[name],
                self.layer_size[name],
                self.activation[name],
                self.kernel_initializer,
                regularization=self.regularization,
                end_scale=self.opnn_end_scale,
                stacked=self.stacked,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                block=self.block,
                block_linear_out=self.block_linear_out,
                cross_encoder=self.cross_encoder,
                cross_encoder_layer_size=self.cross_encoder_layer_size,
                augmented=self.augmented,
                name='opnn_'+name
            ) for name in self.names
        }

        # Apply I/O Transformation
        if self.opnn_inp_trans:
            if not isinstance(self.opnn_inp_trans, dict):
                utils.raise_value_err(
                    "`opnn_inp_trans` passed into `{}` has to be a `dict` in "
                    "which a specific input transform function is defined "
                    "for each DeepONet: {'name': fun}".format(self.name)
                )
            for name, fun in self.opnn_inp_trans.items():
                self.opnns[name].inp_trans = fun
        if self.opnn_out_trans:
            if not isinstance(self.opnn_out_trans, dict):
                utils.raise_value_err(
                    "`opnn_out_trans` passed into `{}` has to be a `dict` in "
                    "which a specific output transform function is defined "
                    "for each DeepONet: {'name': fun}".format(self.name)
                )
            for name, fun in self.opnn_out_trans.items():
                self.opnns[name].out_trans = fun

        ### Final Scaling -----------------------------------------------------
        if end_scale not in {None, "bias", "dense"}:
            utils.raise_value_err(
                "Invalid `end_scale` argument passed into `{}`. Available "
                "choices: {{None|'bias'|'dense'}}.".format(self.name)
            )
        self.end_scale = end_scale

        if self.end_scale == 'bias':
            self.end_scale_lay = layers.Bias(name='end_scale')
        elif self.end_scale == 'dense':
            self.end_scale_lay = tf_layers.Dense(
                units=self.out_dim,
                kernel_initializer=self.kernel_initializer,
                name='end_scale'
            )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'sub_nets': self.sub_nets,
            'names': self.names,
            'opnn_dim': self.opnn_dim,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'opnn_inp_trans': self.opnn_inp_trans,
            'opnn_out_trans': self.opnn_out_trans,
            'opnn_end_scale': self.opnn_end_scale,
            'regularization': self.regularization,
            'end_scale': self.end_scale,
            'stacked': self.stacked,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'cross_encoder': self.cross_encoder,
            'cross_encoder_layer_size': self.cross_encoder_layer_size,
            'augmented': self.augmented
        }
        base_config = super(MMOpNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(
        self,
        x,
        training=False
    ):
        # Inputs manipulation -------------------------------------------------
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

        # Input flattening
        x = tf.nest.flatten(x)

        # Kernel - OpNNs ------------------------------------------------------
        G_uy = [self.opnns[name](x, training=training) for name in self.names]

        # Outputs manipulation ------------------------------------------------
        # Concatenate if single output or if final scaling is applied
        if not isinstance(self.out_dim, (list,tuple)) \
            or self.end_scale is not None:
            G_uy = tf_layers.Concatenate()(G_uy)

        # Final scaling
        if self.end_scale is not None:
            G_uy = self.end_scale_lay(G_uy)

        # Split if multiple output
        if isinstance(self.out_dim, (list,tuple)) and tf.is_tensor(G_uy):
            G_uy = tf.split(G_uy, self.out_dim, axis=1)

        # Output transformation
        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)

        return G_uy
