import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from pycomet.utilities.model import layers
from pycomet.utilities import utils
from pycomet.networks.opnn import OpNN
from pycomet.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='TimeScaleNN')
class TimeScaleNN(Net):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        species,
        reaction,
        temperature_limits,
        log_mass_frac,
        ts_fast_conn,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        opnn_end_scale='bias',
        stacked=True,
        block='FFBlock',
        block_linear_out=True,
        cross_encoder=False,
        cross_encoder_layer_size=None,
        augmented=False,
        name='tsnn',
        *args,
        **kwargs
    ):
        super(TimeScaleNN, self).__init__(
            dim=dim,
            block=block,
            block_linear_out=block_linear_out,
            name=name,
            *args,
            **kwargs
        )

        ### Dimensionality ----------------------------------------------------
        # Inputs
        self.ts_fast_conn = ts_fast_conn

        # Outputs
        if not isinstance(self.outputs_dim, (list,tuple)):
            self.outputs_dim = [self.outputs_dim]

        # OpNNs
        self.opnn_dim = [{
            "inp": self.inputs_dim,
            "out": self.outputs_dim[i]
        } for i in range(len(self.outputs_dim))]

        ### Input Transformation Layers ---------------------------------------
        self.temperature_limits = temperature_limits
        self.temperature_norm_lay = layers.LinearNorm(
            *self.temperature_limits, trainable=False
        )
        self.log_mass_frac = log_mass_frac
        if self.log_mass_frac:
            self.log_mass_frac_lay = layers.Log(offset=0., trainable=False)

        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm
        self.cross_encoder = cross_encoder
        self.cross_encoder_layer_size = cross_encoder_layer_size
        self.augmented = augmented

        ### OpNNs Initialization ----------------------------------------------
        self.opnn_end_scale = opnn_end_scale
        self.stacked = stacked
        self.opnns = [
            OpNN(
                self.opnn_dim[i],
                self.layer_size,
                self.activation,
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
                name='opnn_' + str(i)
            ) for i in range(len(self.outputs_dim))
        ]

        ### Equilibrium Layer -------------------------------------------------
        self.reaction = reaction
        self.species = species
        self.equilibrium = layers.Equilibrium(
            self.outputs_dim, self.species, reaction=self.reaction
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'ts_fast_conn': self.ts_fast_conn,
            'temperature_limits': self.temperature_limits,
            'log_mass_frac': self.log_mass_frac,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularization': self.regularization,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'opnn_end_scale': self.opnn_end_scale,
            'stacked': self.stacked,
            'reaction': self.reaction,
            'species': self.species,
            'cross_encoder': self.cross_encoder,
            'cross_encoder_layer_size': self.cross_encoder_layer_size,
            'augmented': self.augmented
        }
        base_config = super(TimeScaleNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False, mask=None, noneq_factor=False):

        # Inputs manipulation -------------------------------------------------
        # Extract inputs
        ic, t = x

        # Extract physical inputs
        T, rho, Y = tf.split(
            ic, num_or_size_splits=[1,1,self.inputs_dim[0]-2], axis=1
        )

        # Normalize temperature
        T_norm = self.temperature_norm_lay(T)

        # Add mass fractions (micro-bins to macro-bins)
        if self.ts_fast_conn != None:
            Y_ = []
            for indeces in self.ts_fast_conn:
                Y_.append(
                    tf.math.reduce_sum(
                        tf.gather(Y, indeces, axis=1), axis=1, keepdims=True
                    )
                )
            Y = tf_layers.Concatenate()(Y_) if len(Y_) > 1 else Y_[-1]

        # Log-scaled mass fractions
        if self.log_mass_frac:
            Y = self.log_mass_frac_lay(Y)

        # Recompose ICs
        ic = tf_layers.Concatenate()([T_norm, rho, Y])

        # Kernel - OpNNs ------------------------------------------------------
        y = [ g_uy([ic, t], training, mask) for g_uy in self.opnns ]

        # Equilibrium layer ---------------------------------------------------
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        if training is False and noneq_factor is True:
            return y
        else:
            return self.equilibrium([y, T, rho])
