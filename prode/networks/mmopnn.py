import tensorflow as tf
import tensorflow.keras.layers as lay

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.utilities.model import layers
from prode.utilities import utils
# from prode.networks.opnn import OpNN
from prode.networks.net import Net



@tf.keras.utils.register_keras_serializable(package='PrODE', name='MMOpNN')
class MMOpNN(Net):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        names,
        opnn_dim,
        layer_size,
        activation,
        kernel_initializer,
        opnn_inp_trans=None,
        opnn_out_trans=None,
        regularization=None,
        use_end_bias=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        name='mmopnn',
        *args,
        **kwargs
    ):
        super(MMOpNN, self).__init__(
            dim=dim,
            block=block,
            name=name,
            *args,
            **kwargs
        )

        ### Hyperparameters Initialization ------------------------------------
        if len(names) <= 1:
            utils.raise_value_err(
                "The number of `names` passed into `{}` has to be "
                "grater than 1 (multiple DeepONets).".format(self.name))
        else:
            self.names = names

        self.opnn_dim = opnn_dim
        # self.opnn_dim = {name: {
        #     "inp": self.inp_dim,
        #     "out": self.out_dim[i]
        # } for i, name in enumerate(self.names)}

        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.opnn_inp_trans = opnn_inp_trans
        self.opnn_out_trans = opnn_out_trans
        self.regularization = regularization
        self.use_end_bias = use_end_bias
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### OpNNs Initialization ----------------------------------------------
        # self.opnns = [ OpNN(
        self.opnns = {
            name: OpNNLayer(
                self.opnn_dim[name],
                self.layer_size[name],
                self.activation[name],
                self.kernel_initializer,
                regularization=self.regularization,
                use_end_bias=self.use_end_bias,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                block=self.block,
                name='opnn_'+name
            ) for name in self.names
        }

        # Apply I/O Transformation
        if self.opnn_inp_trans:
            if not isinstance(self.opnn_inp_trans, dict):
                utils.raise_value_err(
                    "`opnn_inp_trans` passed into `{}` has to be a `dict` in "
                    "which a specific input transform function is defined "
                    "for each DeepONet: {'name': fun}".format(self.name)
                )
            for name, trans in self.opnn_inp_trans.items():
                self.opnns[name].inp_trans = trans
        if self.opnn_out_trans:
            if not isinstance(self.opnn_out_trans, dict):
                utils.raise_value_err(
                    "`opnn_out_trans` passed into `{}` has to be a `dict` in "
                    "which a specific output transform function is defined "
                    "for each DeepONet: {'name': fun}".format(self.name)
                )
            for name, trans in self.opnn_out_trans.items():
                self.opnns[name].out_trans = trans

        # for i in self.opnns:
        #     print('+++++++++++')
        #     # print(json.dumps(i.get_config(), sort_keys=False, indent=4))
        #     print(i.get_config())
        #     print('+++++++++++')

        # if self.opnn_inp_trans:
        #     if not isinstance(self.opnn_inp_trans, dict):
        #         utils.raise_value_err(
        #             "`opnn_inp_trans` passed into `{}` has to be a `dict` in "
        #             "which a specific input transform function is defined "
        #             "for each DeepONet: {'name': fun}".format(self.name)
        #         )
        #     for i, name in enumerate(self.names):
        #         self.opnns[i].apply_input_transform(self.opnn_inp_trans[name])
        # if self.opnn_out_trans:
        #     if not isinstance(self.opnn_out_trans, dict):
        #         utils.raise_value_err(
        #             "`opnn_out_trans` passed into `{}` has to be a `dict` in "
        #             "which a specific output transform function is defined "
        #             "for each DeepONet: {'name': fun}".format(self.name)
        #         )
        #     for i, name in enumerate(self.names):
        #         self.opnns[i].apply_output_transform(self.opnn_out_trans[name])


    # Configuration update
    ###########################################################################
    def get_config(self):


        # print('Ciaooooooooooooooooooooooooooooooooooo')

        # opnn_config = {}
        # for opnn in self.opnns:
        #     opnn_config.update({
        #         opnn.name+'_'+k: v for k,v in opnn.get_config().items()
        #     })

        config = super().get_config().copy()
        config.update({
            'names': self.names,
            'opnn_dim': self.opnn_dim,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'opnn_inp_trans': self.opnn_inp_trans,
            'opnn_out_trans': self.opnn_out_trans,
            'regularization': self.regularization,
            'use_end_bias': self.use_end_bias,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'opnns': self.opnns
            # **opnn_config
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x, training=None, mask=None):

        # Inputs
        if isinstance(x, (list, tuple)):
            x_u, x_y = x[0], x[1]
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))

        # Input transformation
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                x_u = self.inp_trans["branch"](x_u)
                x_y = self.inp_trans["trunk"](x_y)
            else:
                x_u = self.inp_trans(x_u)
                x_y = self.inp_trans(x_y)

        # OpNNs
        G_uy = [ g_uy([x_u, x_y]) for g_uy in self.opnns.values() ]

        # Output transformation
        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)

        return G_uy

    # Graph
    ###########################################################################
    def get_graph(self):
        x = [ tf.keras.Input(shape=(self.inp_dim[0],), name="branch_input"),
              tf.keras.Input(shape=(self.inp_dim[1],), name="trunk_input") ]
        # y = {k: v for (k,v) in zip(self.names, self.call(x))}
        y = self.call(x)
        # print(y)
        return tf.keras.Model(inputs=x, outputs=y, name=self.name)



# =========================================================================== #
# OpNN Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='OpNNLayer')
class OpNNLayer(lay.Layer):
    """Deep Operator neural network layer."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        use_end_bias=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        name='opnn_lay',
        *args,
        **kwargs
    ):
        super(OpNNLayer, self).__init__(name=name)

        if not isinstance(dim, dict):
            utils.raise_value_err(
                "`dim` passed into `{}` has to be a `dict` of length 2:\n"
                "  - 'inp': `int` (or list of `int`);\n"
                "  - 'out': `int` (or list of `int`).\n"
                "It represents the dimensions of the feature space.".format(self.name)
            )
        self.inp_dim = dim["inp"]
        self.out_dim = dim["out"]
        if not isinstance(self.inp_dim, (list, tuple)):
            utils.raise_value_err(
                "`dim['inp']` passed into `{}` has to be a `list` or "
                "a `tuple`of length 2: [u_dim, y_dim]".format(self.name))

        self.block = block if callable(block) else utils.get_class(layers, block)

        self.inp_trans = None
        self.out_trans = None

        ### Hyperparameters Initialization ------------------------------------
        if isinstance(layer_size, dict):
            self.layer_size_branch = layer_size["branch"]
            self.layer_size_trunk = layer_size["trunk"]
        else:
            self.layer_size_branch = self.layer_size_trunk = layer_size

        if isinstance(activation, dict):
            self.act_branch = activations.get(activation["branch"])
            self.act_trunk = activations.get(activation["trunk"])
        else:
            self.act_branch = self.act_trunk = activations.get(activation)

        if regularization is not None:
            self.regularizer = regularizers.get(**regularization)
        else:
            self.regularizer = regularization

        if isinstance(batch_norm, dict):
            self.batch_norm_branch = batch_norm["branch"]
            self.batch_norm_trunk  = batch_norm["trunk"]
        else:
            self.batch_norm_branch = self.batch_norm_trunk = batch_norm

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.dropout_rate = dropout_rate
        self.weight_norm = weight_norm
        self.use_end_bias = use_end_bias

        ### Blocks Initialization ---------------------------------------------
        self.branches = [ self.block(
            self.layer_size_branch,
            self.act_branch,
            self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm_branch,
            weight_norm=self.weight_norm,
            name='Branch_'+str(i)
        ) for i in range(self.out_dim) ]

        self.trunk = self.block(
            self.layer_size_trunk,
            self.act_trunk,
            self.kernel_initializer,
            regularizer=self.regularizer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm_trunk,
            weight_norm=self.weight_norm,
            name='Trunk_0'
        )

        self.end_bias = layers.BiasLay(name='EndBias') if self.use_end_bias else None

    @property
    def inp_trans(self):
        return self._inp_trans

    @inp_trans.setter
    def inp_trans(self, transform):
        self._inp_trans = transform

    @property
    def out_trans(self):
        return self._out_trans

    @out_trans.setter
    def out_trans(self, transform):
        self._out_trans = transform


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'inp_dim':              self.inp_dim,
            'out_dim':              self.out_dim,
            'block':                self.block,
            'inp_trans':            self.inp_trans,
            'out_trans':            self.out_trans,
            'layer_size_branch':    self.layer_size_branch,
            'layer_size_trunk':     self.layer_size_trunk,
            'act_branch':           self.act_branch,
            'act_trunk':            self.act_trunk,
            'kernel_initializer':   self.kernel_initializer,
            'regularizer':          self.regularizer,
            'dropout_rate':         self.dropout_rate,
            'batch_norm_branch':    self.batch_norm_branch,
            'batch_norm_trunk':     self.batch_norm_trunk,
            'weight_norm':          self.weight_norm,
            'use_end_bias':         self.use_end_bias,
            'branches':             self.branches,
            'trunk':                self.trunk,
            'end_bias':             self.end_bias
        })
        # print(config['name'])

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x, *args, **kwargs):

        # Inputs
        if isinstance(x, (list, tuple)):
            x_u, x_y = x[0], x[1]
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))

        # Input transformation
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                x_u = self.inp_trans["branch"](x_u)
                x_y = self.inp_trans["trunk"](x_y)
            else:
                x_u = self.inp_trans(x_u)
                x_y = self.inp_trans(x_y)

        # Branch/-es
        G_u = [ branch(x_u) for branch in self.branches ]

        # Trunk
        G_y = self.trunk(x_y)

        # Output layer
        G_uy = [ lay.Dot(axes=1)([G_u_i, G_y]) for G_u_i in G_u ]
        G_uy = lay.Concatenate()(G_uy) if len(G_uy) > 1 else G_uy[-1]
        if self.use_end_bias:
            G_uy = self.end_bias(G_uy)

        # Output transformation
        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)
            
        return G_uy