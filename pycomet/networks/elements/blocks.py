import tensorflow as tf

from keras import layers as tf_layers
from tensorflow_addons import layers as tfa_layers
from pycomet.networks.elements import layers as pcm_layers
from pycomet.networks.elements import activations
from pycomet.networks.elements import initializers
from pycomet.networks.elements import regularizers
from pycomet import operations as ops
from pycomet import config as cfg
from pycomet import utils


# =========================================================================== #
# MultiLayerBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='MultiLayerBlock'
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
        dropout_kwargs=dict(rate=0.),
        **kwargs
    ):
        super(MultiLayerBlock, self).__init__(
            **kwargs
        )
        # Hyperparameters Initialization --------------------------------------
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
        self.dropout_kwargs = dropout_kwargs

        # Layers Initialization -----------------------------------------------
        self.lays = []
        # Dense
        dense = tf_layers.Dense(
            units=self.units,
            activation=None if self.batch_norm == "before" else self.activation,
            use_bias=False if self.batch_norm == "before" else self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            name=self.name+'_dense'
        )
        # Weight normalization
        if self.weight_norm:
            dense = tfa_layers.WeightNormalization(dense)
        self.lays.append(dense)
        # Batch normalization
        if self.batch_norm is not None:
            self.lays.append(
                tf_layers.BatchNormalization(name=self.name+'_batchNorm')
            )
        # Dropout
        if self.dropout_kwargs['rate'] > 0.:
            self.lays.append(
                pcm_layers.Dropout(**dropout_kwargs, name=self.name+'_dropout')
            )

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        # Dense
        y = self.lays[0](x)
        # Batch normalization
        if self.batch_norm == "before":
            y = self.activation(self.lays[1](y, training=training))
        elif self.batch_norm == "after":
            y = self.lays[1](y, training=training)
        # Dropout
        if self.dropout_kwargs['rate'] > 0.:
            y = self.lays[-1](y, training=training)
        return y

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'weight_norm': self.weight_norm,
            'batch_norm': self.batch_norm,
            'dropout_kwargs': self.dropout_kwargs
        }
        base_config = super(MultiLayerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# FFBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='FFBlock'
)
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
        dropout_kwargs=dict(rate=0.),
        batch_norm=None,
        weight_norm=False,
        **kwargs
    ):
        super(FFBlock, self).__init__(
            **kwargs
        )
        # Hyperparameters Initialization --------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularizer
        self.dropout_kwargs = dropout_kwargs
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm
        # Checking
        if not all(
            [isinstance(k, (list, tuple)) for k in (activation, layer_size)]
        ) or len(activation) != len(layer_size)-1:
            utils.raise_value_err(
                f"`activation` and `layer_size` inputs passed to {self.name} "
                "block have to be a `list`/`tuple` of lengths: "
                "len(activation) = len(layer_size)-1"
            )

        # Layers Initialization -----------------------------------------------
        # Hidden Layers
        names = [f'hidden_{i}' for i in range(len(self.layer_size)-2)]
        names = names + ['output']
        self.multi_lays = [
            MultiLayerBlock(
                units=self.layer_size[i+1],
                activation=self.activation[i],
                kernel_initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                weight_norm=self.weight_norm,
                batch_norm=self.batch_norm,
                dropout_kwargs=self.dropout_kwargs,
                name=names[i]
            ) for i in range(len(self.layer_size)-1)
        ]

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        # Input layer
        y = x
        # Hidden/Output layers
        for i in range(len(self.layer_size)-1):
            y = self.multi_lays[i](y, training=training)
        return y

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_kwargs': self.dropout_kwargs,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm
        }
        base_config = super(FFBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# ResPIBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='ResPIBlock'
)
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
        dropout_kwargs=dict(rate=0.),
        batch_norm=None,
        weight_norm=False,
        **kwargs
    ):
        super(ResPIBlock, self).__init__(
            layer_size=layer_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            regularizer=regularizer,
            dropout_kwargs=dropout_kwargs,
            batch_norm=batch_norm,
            weight_norm=weight_norm,
            **kwargs
        )
        # Hyperparameters Initialization --------------------------------------
        for i in range(1, len(self.layer_size)-2):
            if self.layer_size[i] != self.layer_size[i+1]:
                utils.raise_value_err(
                    "The number of units of each hidden layer "
                    f"must be the same in `{self.name}`."
                )
        self.uv_activation = uv_activation

        # Layers Initialization -----------------------------------------------
        self.U, self.V = [
            MultiLayerBlock(
                units=self.layer_size[1],
                activation=self.uv_activation[i],
                kernel_initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                weight_norm=self.weight_norm,
                batch_norm=self.batch_norm,
                dropout_kwargs=self.dropout_kwargs,
                name=name
            ) for i, name in enumerate(('U', 'V'))
        ]

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        # Input layer
        y = x
        # Encoder layers
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

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'uv_activation': self.uv_activation
        }
        base_config = super(ResPIBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# PreNetBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='PreNetBlock'
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
        dropout_kwargs=dict(rate=0.),
        batch_norm=None,
        weight_norm=False,
        out_xform=None,
        transformation='affine',
        num_coords=1,
        **kwargs
    ):
        # Initialize math transformation --------------------------------------
        # Number of coordinates of the system to transform
        # See `x2` in function `call`
        self.num_coords = num_coords
        # Available transformations
        self.xform_list = (
            'translation', 'euclidean', 'similarity', 'affine', 'projective'
        )
        # Initialize transformation
        self.transformation = transformation
        if self.transformation == 'translation':
            self.outputs_dim = self.num_coords
            self.i, self.j = self.num_coords, 0
        elif self.transformation in ('euclidean', 'similarity', 'affine'):
            self.outputs_dim = self.num_coords * (self.num_coords + 1)
            # if self.transformation == 'similarity' and self.num_coords > 1:
            #     outputs_dim = outputs_dim + 1
            self.i, self.j = self.num_coords, self.num_coords
        elif self.transformation == 'projective':
            self.outputs_dim = (self.num_coords + 1) ** 2
            self.i, self.j = self.num_coords+1, self.num_coords
        else:
            utils.raise_value_err(
                f"The types of transformation available are: {self.xform_list}"
            )

        # Outputs manipulation ------------------------------------------------
        self.out_xform = out_xform

        super(PreNetBlock, self).__init__(
            layer_size=list(layer_size)[:-1] + [self.outputs_dim],
            activation=activation,
            kernel_initializer=kernel_initializer,
            regularizer=regularizer,
            dropout_kwargs=dropout_kwargs,
            batch_norm=batch_norm,
            weight_norm=weight_norm,
            **kwargs
        )

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
        x1, x2 = ops.unpack_list(x, n_elem=2)

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

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'num_coords': self.num_coords,
            'out_xform': self.out_xform,
            'transformation': self.transformation
        }
        base_config = super(PreNetBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
