impors tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers

from .net import Net


class OpNN(Net):
    """Deep Operator neural networks."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        G_dim=1,
        layer_size_branch,
        layer_size_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        use_bias=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False):
        super(DeepOpNN, self).__init__()
        
        ### Hyperparameters Initialization ------------------------------------
        self.G_dim = G_dim

        self.layer_size_branch = layer_size_branch
        self.layer_size_trunk  = layer_size_trunk

        if isinstance(activation, dict):
            self.act_branch = activations.get(activation["branch"])
            self.act_trunk  = activations.get(activation["trunk"])
        else:
            self.act_branch = self.act_trunk = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer        = regularizers.get(regularization)
        self.use_bias           = use_bias

        self.dropout_rate       = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            raise ValueError(
                "batch_norm == {}. Available choices: {{None|'before'|'after'}}.".format(
                    batch_norm
                )
            )
        self.batch_norm  = batch_norm
        self.weight_norm = weight_norm

        ### Blocks Initialization ---------------------------------------------
        self.branches   = [ Block(
                                self.layer_size_branch,
                                self.act_branch,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Branch'+str(i)) for i in range(self.G_dim) ]

        self.trunk      =   Block(
                                self.layer_size_trunk,
                                self.act_trunk,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Trunk')

        if self.use_bias:
            self.b_final = lay.Dense(
                                    units=self.G_dim,
                                    activation='linear',
                                    use_bias=True,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer="zeros",
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    name='FinalScale'
                                )


    # Calling
    ###########################################################################
    def call(self, x):

        self.x   = x
        self.x_y = x[:,0:1]
        self.x_u = x[:,1: ]

        # Input layer
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                self.x_u = self.inp_trans["branch"](self.x_u)
                self.x_y = self.inp_trans["trunk"](self.x_y)
            else:
                self.x_u = self.inp_trans(self.x_u)
                self.x_y = self.inp_trans(self.x_y)

        # Branch/-es
        self.G_u = [ self.branches[i](self.x_u) for i in range(self.G_dim) ]

        # Trunk
        self.G_y = self.trunk(self.x_y) for i in range(self.G_dim)

        # Output layer
        self.G_uy = [ lay.Dot(axes=1)([G_u[i], G_y]) for i in range(self.G_dim) ]
        self.G_uy = self.b_final(tf.concat(G_uy, 1))

        if self._out_trans is not None:
            self.G_uy = self.out_trans(self.G_uy)

        self.y = self.G_uy
        
        return self.y


###############################################################################
# Branch/Trunk block
###############################################################################

class Block(tf.keras.layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name=None):
        super(Block, self).__init__(name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        self.name                = name
        self.layer_size          = layer_size
        self.activation          = activations.get(activation)
        self.kernel_initializer  = initializers.get(kernel_initializer)
        self.regularizer         = regularizers.get(regularization)
        self.dropout_rate        = dropout_rate
        self.batch_norm          = batch_norm
        self.weight_norm         = weight_norm

        ### Layers Initialization ---------------------------------------------
        # Hidden Layers
        self.lays = [ self.hidden_block(i+1) for i in range(len(self.layer_size)-2) ]
        # Output Layer
        self.lays.append( self.dense(self.layer_size[-1], name=self.name+'_output') )


    # Calling
    ###########################################################################
    def call(self, x):

        self.x = x

        # Input layer
        y = x
        if self._inp_trans is not None:
            y = self._inp_trans(y)

        # Hidden layers
        for i in range(len(self.layer_size) - 2):
            layer = self.lays[i]
            y = layer[0](y)
            if self.batch_norm == "before":
                y = self.activation( layer[0](y) )
            elif self.batch_norm == "after":
                y = layer[0](y)
            if self.dropout_rate > 0:
                y = layer[-1](y)

        # Output layer
        self.y = self.lays[-1](y)
        if self._out_trans is not None:
            self.y = self._out_trans(self.x, self.y)

        return self.y


    # Layers Definition
    ###########################################################################
    def hidden_block(self, number):

        act_fun = self.activation
        bias    = True
        if self.batch_norm == "before":
            act_fun = None
            bias    = False

        l = [ self.dense(self.layer_size[number], activation=act_fun, \
                         use_bias=bias, name=self.name+'_hid'+str(number)) ]

        if self.batch_norm is not None:
            bn = lay.BatchNormalization(name=self.name+'_normHid'+str(number))
            l.append(bn)

        if self.dropout_rate > 0:
            do = lay.Dropout(self.dropout_rate, noise_shape=None, \
                             training=self.dropout, name=self.name+'_dropHid'+str(number))
            l.append(do)

        return l


    def dense(self, units, activation=None, use_bias=True, name=None):
        l = lay.Dense(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer="zeros",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=name
        )
        if self.weight_norm:
            l = tfa.layers.WeightNormalization(l)

        return l
