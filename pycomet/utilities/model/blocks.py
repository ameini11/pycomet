import tensorflow as tf
import tensorflow_addons.layers as tfa_layers
import tensorflow.python.keras.layers as tf_layers

from pycomet.utilities.model import activations
from pycomet.utilities.model import initializers
from pycomet.utilities.model import regularizers
from pycomet.utilities import utils



# =========================================================================== #
# MultiLayerBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package='PyCOMET', name='MultiLayerBlock'
)
class MultiLayerBlock(tf_layers.Layer):
    """Multi Layer Block."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='GlorotNormal',
        regularizer=None,
        weight_norm=False,
        batch_norm=None,
        dropout_rate=0,
        name='multi_layer_block',
        **kwargs
    ):
        super(MultiLayerBlock, self).__init__(name=name, **kwargs)
        
        ### Hyperparameters Initialization ------------------------------------
        self.units = units
        if batch_norm not in {None, "before", "after"}:
            utils.raise_value_err(
                "Invalid `batch_norm` argument passed into "
                "`{}`. Available choices: {{None|'before'|"
                "'after'}}.".format(self.name)
            )
        self.batch_norm = batch_norm
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(**regularizer)
        self.weight_norm = weight_norm
        self.dropout_rate = dropout_rate

        ### Layers Initialization ---------------------------------------------
        self.lays = []
        dense = tf_layers.Dense(
            units=self.units,
            activation=None if self.batch_norm == "before" else self.activation,
            use_bias=False if self.batch_norm == "before" else self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            name=self.name+'_dense'
        )
        if self.weight_norm:
            dense = tfa_layers.WeightNormalization(dense)
        self.lays.append(dense)

        if self.batch_norm is not None:
            self.lays.append(
                tf_layers.BatchNormalization(name=self.name+'_batchNorm')
            )

        if self.dropout_rate > 0:
            self.lays.append(
                tf_layers.Dropout(self.dropout_rate, name=self.name+'_dropout')
            )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'units': self.units,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'weight_norm': self.weight_norm,
            'dropout_rate': self.dropout_rate,
            'lays': self.lays
        }
        base_config = super(MultiLayerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        y = self.lays[0](x)
        if self.batch_norm == "before":
            y = self.activation(self.lays[1](y, training=training))
        elif self.batch_norm == "after":
            y = self.lays[1](y, training=training)
        if self.dropout_rate > 0:
            y = self.lays[-1](y, training=training)
        return y


# # =========================================================================== #
# # MultiLayerBlock Layer
# # =========================================================================== #
# @tf.keras.utils.register_keras_serializable(
#     package='PyCOMET', name='MultiLayerBlock'
# )
# class MultiLayerBlock(tf_layers.Layer):
#     """Multi Layer Block."""

#     # Initialization
#     ###########################################################################
#     def __init__(
#         self,
#         units,
#         activation=None,
#         use_bias=True,
#         kernel_initializer='glorot_uniform',
#         regularizer=None,
#         weight_norm=False,
#         batch_norm=None,
#         dropout_rate=0,
#         name='multi_layer_block',
#         **kwargs
#     ):
#         super(MultiLayerBlock, self).__init__(name=name, **kwargs)
        
#         ### Hyperparameters Initialization ------------------------------------
#         self.units = units
#         if batch_norm not in {None, "before", "after"}:
#             utils.raise_value_err(
#                 "Invalid `batch_norm` argument passed into "
#                 "`{}`. Available choices: {{None|'before'|"
#                 "'after'}}.".format(self.name)
#             )
#         self.batch_norm = batch_norm
#         self.activation = activation
#         self.use_bias = use_bias
#         self.kernel_initializer = kernel_initializer
#         self.regularizer = regularizer
#         self.weight_norm = weight_norm
#         self.dropout_rate = dropout_rate

#         ### Layers Initialization ---------------------------------------------
#         self.layers = []
#         dense = tf_layers.Dense(
#             units=self.units,
#             activation=None if self.batch_norm == "before" else self.activation,
#             use_bias=False if self.batch_norm == "before" else self.use_bias,
#             kernel_initializer=self.kernel_initializer,
#             kernel_regularizer=self.regularizer,
#             bias_regularizer=self.regularizer,
#             name=self.name+'_dense'
#         )
#         if self.weight_norm:
#             dense = tfa_layers.WeightNormalization(dense)
#         self.layers.append(dense)

#         if self.batch_norm is not None:
#             self.layers.append(
#                 tf_layers.BatchNormalization(name=self.name+'_batchNorm')
#             )

#         if self.dropout_rate > 0:
#             self.layers.append(
#                 tf_layers.Dropout(self.dropout_rate, name=self.name+'_dropout')
#             )

#     # Configuration update
#     ###########################################################################
#     def get_config(self):
#         config = {
#             'units': self.units,
#             'batch_norm': self.batch_norm,
#             'activation': self.activation,
#             'use_bias': self.use_bias,
#             'kernel_initializer': self.kernel_initializer,
#             'regularizer': self.regularizer,
#             'weight_norm': self.weight_norm,
#             'dropout_rate': self.dropout_rate,
#             'layers': self.layers
#         }
#         base_config = super(MultiLayerBlock, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

#     # Calling
#     ###########################################################################
#     def call(self, x, training=None):
#         y = self.layers[0](x)
#         if self.batch_norm == "before":
#             y = self.activation(self.layers[1](y, training=training))
#         elif self.batch_norm == "after":
#             y = self.layers[1](y, training=training)
#         if self.dropout_rate > 0:
#             y = self.layers[-1](y, training=training)
#         return y


# =========================================================================== #
# FFBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='FFBlock')
class FFBlock(tf_layers.Layer):
    """Simple Feed-forward block."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        layer_size=None,
        activation=['linear'],
        kernel_initializer=None,
        regularizer=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='ff_block',
        **kwargs
    ):
        super(FFBlock, self).__init__(name=name, **kwargs)
        
        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm
        # Checking
        if not all(
            isinstance(k, (list, tuple)) for k in (activation, layer_size)
        ) or len(activation) != len(layer_size)-1:
            utils.raise_value_err(
                f"`activation` and `layer_size` inputs passed to {self.name} "
                "block have to be a `list`/`tuple` of length: "
                "len(activation) = len(layer_size)-1"
            )

        ### Layers Initialization ---------------------------------------------
        # Hidden Layers
        names = [f'hidden{i}' for i in range(len(self.layer_size)-2)]
        names = names + ['output']
        self.multi_lays = [
            MultiLayerBlock(
                units=self.layer_size[i+1],
                activation=self.activation[i],
                kernel_initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                weight_norm=self.weight_norm,
                batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                name=names[i]
            ) for i in range(len(self.layer_size)-1)
        ]

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm
        }
        base_config = super(FFBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        # Input layer
        y = x
        # Hidden/Output layers
        for i in range(len(self.layer_size)-1):
            y = self.multi_lays[i](y, training=training)
        return y


# =========================================================================== #
# ResPIBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='ResPIBlock')
class ResPIBlock(FFBlock):
    """An improved residual block architecture for PINNs.

    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed
        neural networks`
        https://arxiv.org/abs/2001.04536"""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        layer_size=None,
        activation=['linear'],
        uv_activation=['sigmoid','sigmoid'],
        kernel_initializer=None,
        regularizer=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='res_pi_block',
        **kwargs
    ):
        super(ResPIBlock, self).__init__(
            layer_size=layer_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            weight_norm=weight_norm,
            name=name,
            **kwargs
        )

        ### Hyperparameters Initialization ------------------------------------
        for i in range(1, len(self.layer_size)-2):
            if self.layer_size[i] != self.layer_size[i+1]:
                utils.raise_value_err(
                    "The number of neurons of each hidden layer must "
                    "be the same in `{}`.".format(self.name)
                )
        self.uv_activation = uv_activation

        ### Layers Initialization ---------------------------------------------
        self.U, self.V = [
            MultiLayerBlock(
                units=self.layer_size[1],
                activation=self.uv_activation[i],
                kernel_initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                weight_norm=self.weight_norm,
                batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                name=name
            ) for i, name in enumerate(('U', 'V'))
        ]

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'uv_activation': self.uv_activation
        }
        base_config = super(ResPIBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        # Input layer
        y = x
        # Transformer layers
        u = self.U(y, training=training)
        v = self.V(y, training=training)
        # Hidden layers
        y = self.multi_lays[0](y, training=training)
        for i in range(1, len(self.layer_size) - 2):
            z = self.multi_lays[i](y, training=training)
            y = (1. - z) * u + z * v
        # Output layer
        y = self.multi_lays[-1](y, training=training)
        return y


# # =========================================================================== #
# # ResPIBlock Layer
# # =========================================================================== #
# @tf.keras.utils.register_keras_serializable(
#     package='PyCOMET', name='CrossEncoderBlock'
# )
# class CrossEncoderBlock(FFBlock):
#     """An improved residual block architecture for PINNs.

#     See:
#         Sifan Wang, Yujun Teng, Paris Perdikaris
#         `Understanding and mitigating gradient pathologies in physics-informed
#         neural networks`
#         https://arxiv.org/abs/2001.04536"""

#     # Initialization
#     ###########################################################################
#     def __init__(
#         self,
#         layer_size=None,
#         activation='linear',
#         kernel_initializer=None,
#         regularizer=None,
#         dropout_rate=0,
#         batch_norm=None,
#         weight_norm=False,
#         linear_out=True,
#         name='cross_encoder_block',
#         **kwargs
#     ):
#         super(CrossEncoderBlock, self).__init__(
#             layer_size=layer_size,
#             activation=activation,
#             kernel_initializer=kernel_initializer,
#             regularizer=regularizer,
#             dropout_rate=dropout_rate,
#             batch_norm=batch_norm,
#             weight_norm=weight_norm,
#             linear_out=linear_out,
#             name=name,
#             **kwargs
#         )

#         ### Hyperparameters Initialization ------------------------------------
#         for i in range(1, len(self.layer_size)-2):
#             if self.layer_size[i] != self.layer_size[i+1]:
#                 utils.raise_value_err(
#                     "The number of neurons of each hidden layer must "
#                     "be the same in `{}`.".format(self.name)
#                 )

#     # Calling
#     ###########################################################################
#     def call(self, x, u, v, training=False):
#         # Hidden layers
#         y = self.multi_lays[0](x, training=training)
#         for i in range(1, len(self.layer_size) - 2):
#             z = self.multi_lays[i](y, training=training)
#             y = (1. - z) * u + z * v
#         # Output layer
#         y = self.multi_lays[-1](y, training=training)
#         return y


# =========================================================================== #
# PreNetBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package='PyCOMET', name='PreNetBlock'
)
class PreNetBlock(FFBlock):
    """See: https://arxiv.org/pdf/2204.12670.pdf
    """

    # Initialization
    ###########################################################################
    def __init__(
        self,
        layer_size=None,
        activation=['linear'],
        kernel_initializer=None,
        regularizer=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        out_xform=None,
        transformation='affine',
        num_coordinates=1,
        name='pre_net_block',
        **kwargs
    ):
        # Initialize math transformation --------------------------------------
        # Number of coordinates of the system to transform (see `x2` in `call`)
        self.n_coords = num_coordinates
        # Available transformations
        self.xform_list = (
            'translation', 'euclidean', 'similarity', 'affine', 'projective'
        )
        # Initialize transformation
        self.xform = transformation
        if self.xform == 'translation':
            self.out_dim = self.n_coords
            self.i, self.j = self.n_coords, 0
        elif self.xform in ('euclidean', 'similarity', 'affine'):
            self.out_dim = self.n_coords * (self.n_coords + 1)
            # if self.transformation == 'similarity' and self.n_coords > 1:
            #     out_dim = out_dim + 1
            self.i, self.j = self.n_coords, self.n_coords
        elif self.xform == 'projective':
            self.out_dim = (self.n_coords + 1) ** 2
            self.i, self.j = self.n_coords+1, self.n_coords
        else:
            utils.raise_value_err(
                f"The types of transformation available are: {self.xform_list}"
            )

        # Outputs manipulation ------------------------------------------------
        self.out_xform = out_xform

        super(PreNetBlock, self).__init__(
            layer_size=list(layer_size)[:-1] + [self.out_dim],
            activation=activation,
            kernel_initializer=kernel_initializer,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            weight_norm=weight_norm,
            name=name,
            **kwargs
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'n_coords': self.n_coords,
            'xform_list': self.xform_list,
            'xform': self.xform
        }
        base_config = super(PreNetBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False):
        # Inputs manipulation -------------------------------------------------
        # Check inputs format
        if not isinstance(x, (list, tuple)):
            utils.raise_value_err(
                f"The inputs of the `{self.name}` net have to be "
                "a `list` or a `tuple`: [x1, x2]"
            )
        # Unpack inputs
        x1, x2 = x

        # Kernel --------------------------------------------------------------
        y = super(PreNetBlock, self).call(x1, training=training)

        # Apply transformation ------------------------------------------------
        y = tf.split(y, [self.i*self.j, self.i], axis=1)
        if self.j > 0:
            y[0] = tf.reshape(y[0], [-1,self.i,self.j])
            z = tf.einsum('nij,nj->ni', y[0], x2)
        else:
            z = x2
        z = z + y[1]

        # Outputs manipulation ------------------------------------------------
        if self.out_xform is not None:
            z = self.out_xform(z)

        return z