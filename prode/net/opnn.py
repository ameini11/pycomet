import tensorflow as tf
import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers
from . import layers

from .net import Net


class OpNN(Net):
    """Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        G_dim,
        layer_size_branch,
        layer_size_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        use_end_scale=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='opnn'):
        super(OpNN, self).__init__(name=name)
        
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
        self.dropout_rate       = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            raise ValueError(
                "batch_norm == {}. Available choices: {{None|'before'|'after'}}.".format(
                    batch_norm
                )
            )

        self.batch_norm    = batch_norm
        self.weight_norm   = weight_norm
        self.use_end_scale = use_end_scale

        ### Blocks Initialization ---------------------------------------------
        self.branches   = [ layers.FNNBlock(
                                self.layer_size_branch,
                                self.act_branch,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Branch'+str(i)) for i in range(self.G_dim) ]

        self.trunk      =   layers.FNNBlock(
                                self.layer_size_trunk,
                                self.act_trunk,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Trunk')

        self.end_scale  =   lay.Dense(
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
                                name='FinalScale') if self.use_end_scale else None


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'G_dim':              self.G_dim,
            'layer_size_branch':  self.layer_size_branch,
            'layer_size_trunk':   self.layer_size_trunk,
            'act_branch':         self.act_branch,
            'act_trunk':          self.act_trunk,
            'kernel_initializer': self.kernel_initializer,
            'regularizer':        self.regularizer,
            'dropout_rate':       self.dropout_rate,
            'batch_norm':         self.batch_norm,
            'weight_norm':        self.weight_norm,
            'use_end_scale':      self.use_end_scale,
            'branches':           self.branches,
            'trunk':              self.trunk,
            'end_scale':          self.end_scale
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x):

        # Inputs
        self.x = x
        if isinstance(self.x, (list, tuple)):
            self.x_u, self.x_y = x[0], x[1]
        else:
            self.x_u, self.x_y = tf.split(x, [x.shape[1]-1, 1], axis=1)

        # Input transformation
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                x_u = lay.Lambda(self.inp_trans["branch"])(self.x_u)
                x_y = lay.Lambda(self.inp_trans["trunk"])(self.x_y)
            else:
                x_u = lay.Lambda(self.inp_trans)(self.x_u)
                x_y = lay.Lambda(self.inp_trans)(self.x_y)

        # Branch/-es
        G_u = [ self.branches[i](x_u) for i in range(self.G_dim) ]

        # Trunk
        G_y = self.trunk(x_y)

        # Output layer
        G_uy = [ lay.Dot(axes=1)([G_u[i], G_y]) for i in range(self.G_dim) ]
        if self.use_end_scale:
            G_uy = self.end_scale(tf.concat(G_uy, 1))

        # Output transformation
        if self.out_trans is not None:
            G_uy = lay.Lambda(self.out_trans)(G_uy)

        # Outputs
        self.y = G_uy
        
        return self.y


    # Training Step
    ##########################################################################
    def train_step(self, data):
        if self.res_loss_fn is not None:
            return steps.train_step(self, data)
        else:
            return super(FNN, self).train_step(data)


    # Testing Step
    ##########################################################################
    def test_step(self, data):
        if self.res_loss_fn is not None:
            return steps.test_step(self, data)
        else:
            return super(FNN, self).test_step(data)
