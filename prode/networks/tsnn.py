import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.utilities.model import layers
from prode.utilities import utils
from prode.physics.general import const
from prode.networks.opnn import OpNN
from prode.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PrODE', name='TimeScaleNN')
class TimeScaleNN(Net):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        species,
        reaction,
        temp_limits,
        dim,
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
        name='tsnn',
        *args,
        **kwargs
    ):
        super(TimeScaleNN, self).__init__(
            dim=dim,
            block=block,
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

        ### Normalization Layer -----------------------------------------------
        self.temp_limits = temp_limits
        self.temperature_norm = layers.LinearNorm(
            *self.temp_limits, trainable=False
        )

        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

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
                name='opnn_' + str(i)
            ) for i in range(len(self.out_dim))
        ]

        ### Equilibrium Layer -------------------------------------------------
        self.reaction = reaction
        self.species = species
        self.equilibrium = layers.Equilibrium(
            self.out_dim, self.reaction, self.species
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'ts_fast_conn': self.ts_fast_conn,
            'temp_limits': self.temp_limits,
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
            'species': self.species
        }
        base_config = super(TimeScaleNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None, mask=None):

        ic, t = x

        # Inputs manipulation -------------------------------------------------
        # Extract physical inputs
        T, rho, Y = tf.split(
            ic, num_or_size_splits=[1,1,self.inp_dim[0]-2], axis=1
        )

        # Normalize temperature
        T_norm = self.temperature_norm(T)

        # Compose new inputs
        if self.ts_fast_conn != None:
            _Y = []
            for indeces in self.ts_fast_conn:
                _Y.append(
                    tf.math.reduce_sum(
                        tf.gather(Y, indeces, axis=1), axis=1, keepdims=True
                    )
                )
            Y = tf_layers.Concatenate()(_Y) if len(_Y) > 1 else _Y[-1]
        ic = tf_layers.Concatenate()([T_norm, rho, Y])

        # Kernel - OpNNs ------------------------------------------------------
        y = [ g_uy([ic, t], training, mask) for g_uy in self.opnns ]

        # Equilibrium layer ---------------------------------------------------
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        y = self.equilibrium([y, T, rho])

        return y

    # Graph
    ###########################################################################
    def get_graph(self):
        x = [ tf.keras.Input(shape=(self.inp_dim[0],), name="branch_input"),
              tf.keras.Input(shape=(self.inp_dim[1],), name="trunk_input") ]
        y = self.call(x)
        return tf.keras.Model(inputs=x, outputs=y, name=self.name)
