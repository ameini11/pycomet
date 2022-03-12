import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities.model import layers
from prode.utilities import utils
from prode.networks.opnn import OpNN
from prode.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PrODE', name='TimeScaleNN')
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
        if not isinstance(self.out_dim, (list,tuple)):
            self.out_dim = [self.out_dim]

        # OpNNs
        self.opnn_dim = [{
            "inp": self.inp_dim,
            "out": self.out_dim[i]
        } for i in range(len(self.out_dim))]

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
            ) for i in range(len(self.out_dim))
        ]

        ### Equilibrium Layer -------------------------------------------------
        self.reaction = reaction
        self.species = species
        self.equilibrium = layers.Equilibrium(
            self.out_dim, self.species, reaction=self.reaction
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
    def call(
        self,
        x,
        training=False,
        neq_mask=None
    ):

        # Inputs manipulation -------------------------------------------------
        # Extract inputs
        ic, t = x

        # Extract physical inputs
        T, rho, Y = tf.split(
            ic, num_or_size_splits=[1,1,self.inp_dim[0]-2], axis=1
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

        # Recompose inputs
        x = [ic, t]

        # Kernel - Timescales NN ----------------------------------------------
        if training and neq_mask is None:
            y = [ g_uy(x) for g_uy in self.opnns ]
        else:
            y = []
            n = utils.get_cardinality(x)
            neq_mask = tf.split(neq_mask, neq_mask.shape[1], axis=1)
            for i, mask_i in enumerate(neq_mask):
                # Create a dummy `zeros` tensor
                y_i = tf.zeros((n, self.out_dim[i]), dtype=K.floatx())
                # If any values from previuos timescale is greater
                # than the `tol`, evaluates the new values at the
                # new timescale
                if any(mask_i):
                    # Get indices for the actual values to be
                    # evaluated
                    idx = tf.reshape(tf.where(mask_i)[:,0], shape=[-1])
                    # Update the dummy `zeros` tensor with the
                    # actual evaluation of the DeepONet `j`
                    x_i = tf.nest.map_structure(
                        lambda x: tf.gather(x, idx, axis=0), x
                    )
                    y_i = tf.tensor_scatter_nd_update(
                        y_i,
                        tf.expand_dims(idx, 1),
                        self.opnns[i](x_i)
                    )
                # Append output DeepONet `j` for the current
                # timescale `i` 
                y.append(y_i)

        # Equilibrium layer ---------------------------------------------------
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        return self.equilibrium([y, T, rho])
