import tensorflow as tf
import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers
from . import layers
from . import steps

from .net import Net


@tf.keras.utils.register_keras_serializable()
class FFNN(Net):
    """Feed-Forward neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='ff_nn'):
        super(FFNN, self).__init__(name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        self.dim                 = self.dim
        self.layer_size          = layer_size
        self.activation          = activations.get(activation)
        self.kernel_initializer  = initializers.get(kernel_initializer)
        self.regularizer         = regularizers.get(regularization)
        self.dropout_rate        = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            raise ValueError(
                "\n[PrODE]: batch_norm == {}. Available choices: "
                "{{None|'before'|'after'}}.".format(
                    batch_norm
                )
            )

        self.batch_norm          = batch_norm
        self.weight_norm         = weight_norm

        ### Layers Initialization ---------------------------------------------
        self.block = layers.FFBlock(
                        self.layer_size,
                        self.activation,
                        self.kernel_initializer,
                        regularization=self.regularizer,
                        dropout_rate=self.dropout_rate,
                        batch_norm=self.batch_norm,
                        weight_norm=self.weight_norm)


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'layer_size':         self.layer_size,
            'activation':         self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularizer':        self.regularizer,
            'dropout_rate':       self.dropout_rate,
            'batch_norm':         self.batch_norm,
            'weight_norm':        self.weight_norm,
            'block':              self.block,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x):

        # Input Transform
        y = x
        if self.inp_trans is not None:
            y = self.input_fn(y)

        # Block Layer
        y = self.block(y)

        # Output Transform
        if self.out_trans is not None:
            y = self.output_fn(y)

        return y


    # Steps
    ###########################################################################
    # Training Step
    def train_step(self, data):
        if self.res_loss_fn is not None:
            return steps.train_step(self, data)
        else:
            return super(FNN, self).train_step(data)


    # Testing Step
    def test_step(self, data):
        if self.res_loss_fn is not None:
            return steps.test_step(self, data)
        else:
            return super(OpNN, self).test_step(data)


    # Graph
    ###########################################################################
    def build_graph(self):
        x= tf.keras.Input(shape=(self.dim,))
        return tf.keras.Model(inputs=x, outputs=self.call(x), name=self.name)
