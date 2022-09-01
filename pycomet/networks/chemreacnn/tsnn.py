import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from pycomet.utilities.model import layers
from pycomet.utilities import utils
from pycomet.networks.opnn import OpNN
from pycomet.networks.net import Net
from tensorflow.python.keras import backend as K


@tf.keras.utils.register_keras_serializable(
    package='PyCOMET', name='TimeScaleNN'
)
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
        x, T, rho = self._manipulateinput(x)
        # Kernel - MMOpNNs
        if training or neq_mask is None:
            y = [opnn(x, training=training) for opnn in self.opnns]
        else:
            y = self._adaptive_call(x, neq_mask, training=False)
        # Outputs manipulation
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        # Equilibrium layer
        y = self.equilibrium([y, T, rho])
        return y

    def _manipulateinput(self, x):
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
        # Recompose inputs
        x = [ic, t]
        return x, T, rho

    def _adaptive_call(self, x, neq_mask, training):
        y = []
        # Get number of input data
        n = utils.get_cardinality(x)
        # Split `neq_mask` columnwise and loop over each column
        neq_mask = tf.split(neq_mask, neq_mask.shape[1], axis=1)
        for i, mask_i in enumerate(neq_mask):
            # Create a dummy `zeros` tensor
            y_i = tf.zeros((n, self.outputs_dim[i]), dtype=K.floatx())
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




from pycomet.physics.thermochemistry.specie_thermo import partition_fun
from pycomet.physics.thermochemistry.specie_thermo import specie
# =========================================================================== #
# Equilibrium Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='Equilibrium')
class Equilibrium(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        outputs_dim,
        species,
        reaction=None,
        name='equilibrium',
        *args,
        **kwargs
    ):
        super(Equilibrium, self).__init__(
            name=name,
            *args,
            **kwargs
        )

        self.outputs_dim = outputs_dim
        self.reaction = reaction

        # Species -------------------------------------------------------------
        self.species = {
            sp: specie.Specie(sp, **info) if info != None else specie.Specie(sp) \
                for sp, info in species.items()
        }

        for name, sp_ in self.species.items():
            if sp_.n_at > 1:
                self.molecule = name
            else:
                self.atom = name

        # Partition functions -------------------------------------------------
        if self.reaction is not None:
            self.intPF = {
                name: partition_fun.internal.Internal(specie) \
                    for name, specie in self.species.items()
            }
            if self.reaction == 'diss':
                self.transPF = {
                    name: partition_fun.translational.Translational(specie) \
                        for name, specie in self.species.items()
                }

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'outputs_dim': self.outputs_dim,
            'reaction': self.reaction,
            'species': self.species
        }
        base_config = super(Equilibrium, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, T, rho = inputs
        else:
            utils.raise_value_err(
                "The inputs of the `{}` layer have to be a `list` "
                "or a `tuple`: [x, T]".format(self.name))

        if self.reaction is None:
            y = tf.split(x, num_or_size_splits=self.outputs_dim, axis=1)
            y = [ tf_layers.Softmax(axis=-1)(y_i) for y_i in y ]
        else:
            # Partition functions
            self.updatePF(T)
            if self.reaction == 'diss':
                Q = self.eq_diss(rho)
            elif self.reaction == 'excit':
                Q = [ pf.Q for pf in self.intPF.values() ]
                Q = tf_layers.Concatenate()(Q) if len(Q) > 1 else Q[-1]
            # Simil-Softmax layer
            y = tf_layers.Multiply()([tf.math.exp(x), Q])
            y = tf.split(y, num_or_size_splits=self.outputs_dim, axis=1)
            y = [ y_i / tf.reduce_sum(y_i, axis=-1, keepdims=True) for y_i in y ]
            # y = tf_layers.Multiply()(
            #     [tf.math.exp(-x), Q / tf.reduce_sum(Q, axis=-1, keepdims=True)]
            # )
            # y = tf.split(y, num_or_size_splits=self.outputs_dim, axis=1)

        return y if len(y) > 1 else y[-1]

    def Keq(self):
        '''Diss.-Recomb. equilibrium constants matrix'''
        PFdot = {
            name: self.intPF[name].Q * self.transPF[name].Q \
                for name in self.species.keys()
        }
        return PFdot[self.atom]**2 / PFdot[self.molecule]

    def eq_diss(self, rho):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Calculate equilibrium constant
        backend = self.Keq()

        # Calculate mass fractions
        A = tf.math.pow(rho / self.species[self.atom].m, 2.)
        B = backend * rho / self.species[self.molecule].m
        C = - B

        Y_O = (- B + tf.math.sqrt(B**2. - 4.*A*C)) / (2.*A)
        Y_O2 = 1. - Y_O

        return tf.concat([Y_O2, Y_O], 1)

    def updatePF(self, T):
        # Update partition functions
        for intPF_ in self.intPF.values():
            intPF_.update(T)
        if self.reaction == 'diss':
            for transPF_ in self.transPF.values():
                transPF_.update(T)
