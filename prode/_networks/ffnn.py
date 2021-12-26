import tensorflow as tf

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PrODE', name='FFNN')
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
        block='FFBlock',
        name='ffnn',
        *args,
        **kwargs
    ):
        super(FFNN, self).__init__(
            dim=dim,
            block=block,
            name=name,
            *args,
            **kwargs
        )
        
        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        if regularization is not None:
            self.regularizer = regularizers.get(**regularization)
        else:
            self.regularizer = regularization

        ### Layers Initialization ---------------------------------------------
        self.hidden_blk = self.block(
            self.layer_size,
            self.activation,
            self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            weight_norm=self.weight_norm
        )

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
            'weight_norm': self.weight_norm,
            'hidden_blk': self.hidden_blk
        }
        base_config = super(FFNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False):

        # Input Transform
        y = x
        if self.inp_trans is not None:
            y = self.inp_trans(y)

        # Block Hidden Layer
        y = self.hidden_blk(y, training=training)

        # Output Transform
        if self.out_trans is not None:
            y = self.out_trans(y)

        return y
