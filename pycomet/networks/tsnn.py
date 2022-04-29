import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from pycomet.utilities.model import layers
from pycomet.utilities import utils
from pycomet.networks.opnn import OpNN
from pycomet.networks.net import Net
from tensorflow.python.keras import backend as K


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
        stacked_blocks={"branch": True, "trunk": False},
        block='FFBlock',
        block_linear_out=True,
        shifter=None,
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
        self.stacked_blocks = stacked_blocks
        self.shifter = shifter
        self.augmented = augmented

        ### OpNNs Initialization ----------------------------------------------
        self.opnn_end_scale = opnn_end_scale
        self.opnns = [
            OpNN(
                self.opnn_dim[i],
                self.layer_size,
                self.activation,
                self.kernel_initializer,
                regularization=self.regularization,
                end_scale=self.opnn_end_scale,
                stacked_blocks=self.stacked_blocks,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                block=self.block,
                block_linear_out=self.block_linear_out,
                shifter=self.shifter,
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
            'stacked_blocks': self.stacked_blocks,
            'reaction': self.reaction,
            'species': self.species,
            'shifter': self.shifter,
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
        # Inputs manipulation
        x, T, rho = self._manipulate_input(x)
        # Kernel - MMOpNNs
        if training or neq_mask is None:
            y = [opnn(x, training=training) for opnn in self.opnns]
        else:
            y = self._adaptive_call(x, neq_mask)
        # Outputs manipulation
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        # Equilibrium layer
        y = self.equilibrium([y, T, rho])
        return y

    def _manipulate_input(self, x):
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
        return x, T, rho

    def _adaptive_call(self, x, neq_mask):
        y = []
        # Get number of input data
        n = utils.get_cardinality(x)
        # Split `neq_mask` columnwise and loop over each column
        neq_mask = tf.split(neq_mask, neq_mask.shape[1], axis=1)
        for i, mask_i in enumerate(neq_mask):
            # Create a dummy `zeros` tensor
            y_i = tf.zeros((n, self.out_dim[i]), dtype=K.floatx())
            # If any values of `mask_i` is `True`, evaluate the specific
            # data points
            if any(mask_i):
                # Get indices for the actual data to be evaluated
                idx = tf.reshape(tf.where(mask_i)[:,0], shape=[-1])
                # Extract inputs given specific indices
                x_i = tf.nest.map_structure(
                    lambda x: tf.gather(x, idx, axis=0), x
                )
                # Update the dummy `zeros` tensor with the actual
                # evaluation of the DeepONet `i`
                y_i = tf.tensor_scatter_nd_update(
                    y_i,
                    tf.expand_dims(idx, 1),
                    self.opnns[i](x_i, training=training)
                )
            y.append(y_i)
        return y
