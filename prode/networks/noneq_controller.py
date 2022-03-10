import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities.model import layers
from prode.utilities import utils
from prode.networks.opnn import OpNN
from prode.networks.net import Net
from tensorflow.python.keras import backend as K


@tf.keras.utils.register_keras_serializable(package='PrODE', name='NoneqController')
class NoneqController(Net):
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
        opnn_end_scale='bias',
        stacked=True,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        block_linear_out=True,
        cross_encoder=False,
        cross_encoder_layer_size=None,
        augmented=False,
        name='noneq_controller',
        *args,
        **kwargs
    ):
        super(NoneqController, self).__init__(
            dim=dim,
            block=block,
            block_linear_out=block_linear_out,
            name=name,
            *args,
            **kwargs
        )

        self.sub_nets = ("branch", "trunk")

        self.names = names
        self.num_mmopnns = len(self.names)

        ### Dimensionality ----------------------------------------------------
        # Outputs
        if not isinstance(self.out_dim, (list,tuple)):
            self.out_dim = [self.out_dim]

        # MMOpNNs
        self.mmopnn_dim = [{
            "inp": self.inp_dim,
            "out": self.out_dim[i]
        } for name in self.names]

        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = utils.value_to_dict(layer_size, self.names)
        self.activation = utils.value_to_dict(activation, self.names)
        self.kernel_initializer = utils.value_to_dict(
            kernel_initializer, self.names
        )
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm
        self.cross_encoder = cross_encoder
        self.cross_encoder_layer_size = cross_encoder_layer_size
        self.augmented = augmented

        ### TimeScaleNNs Initialization ---------------------------------------
        self.opnn_dim = opnn_dim
        self.opnn_end_scale = utils.value_to_dict(opnn_end_scale, self.names)
        self.stacked = stacked

        self.mmopnns = [
            MMOpNNController(
                dim=self.mmopnn_dim[name],
                opnn_dim=self.opnn_dim[name],
                layer_size=self.layer_size[name],
                activation=self.activation[name],
                kernel_initializer=self.kernel_initializer[name],
                opnn_end_scale=self.opnn_end_scale[name],
                stacked=self.stacked,
                regularization=self.regularization,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                block=self.block,
                block_linear_out=self.block_linear_out,
                cross_encoder=self.cross_encoder,
                cross_encoder_layer_size=self.cross_encoder_layer_size,
                augmented=self.augmented,
                name=name
            ) for i, name in enumerate(self.names)
        ]


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'sub_nets': self.sub_nets,
            'names': self.names,
            'opnn_dim': self.opnn_dim,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'opnn_end_scale': self.opnn_end_scale,
            'regularization': self.regularization,
            'stacked': self.stacked,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'cross_encoder': self.cross_encoder,
            'cross_encoder_layer_size': self.cross_encoder_layer_size,
            'augmented': self.augmented
        }
        base_config = super(NoneqController, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False, mask=None, tol=0.1):

        # Inputs manipulation -------------------------------------------------
        if isinstance(x, (list, tuple)):
            x = {k: x[i] for i, k in enumerate(self.sub_nets)}
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))

        # Input transformation
        if self.inp_trans is not None:
            for k in self.sub_nets:
                if isinstance(self.inp_trans, dict):
                    x[k] = self.inp_trans[k](x[k])
                else:
                    x[k] = self.inp_trans(x[k])

        # Input flattening
        x = tf.nest.flatten(x)

        # MMOpNNs
        if training:
            y = [ mmopnn(x) for mmopnn in self.mmopnns ]
        else:
            y = []
            n = utils.get_cardinality(x)
            for i, mmopnn in enumerate(self.mmopnns):
                if i == 0:
                    y.append(mmopnn(x))
                else:
                    if i != self.num_mmopnns-1 and all(y[-1] <= tol):
                        y.append(
                            tf.zeros((n, mmopnn.out_dim), dtype=K.floatx())
                        )
                    else:
                        y_i = []
                        y_iminus1 = tf.split(y[-1], y[-1].shape[1], axis=1)
                        for j, y_j in enumerate(y_iminus1):
                            # Create a dummy `zeros` tensor
                            y_ij = tf.zeros(
                                (n, mmopnn.opnns[j].out_dim),
                                dtype=K.floatx()
                            )
                            # If any values from previuos timescale is greater
                            # than the `tol`, evaluates the new values at the
                            # new timescale
                            if any(y_j > tol):
                                # Get indices for the actual values to be
                                # evaluated
                                idx = tf.reshape(
                                    tf.where(y_j > tol)[:,0], shape=[-1]
                                )
                                # Update the dummy `zeros` tensor with the
                                # actual evaluation of the DeepONet `j`
                                y_ij = tf.scatter_nd(
                                    idx,
                                    mmopnn.opnns[j](tf.gather(x, idx, axis=0)),
                                    y_ij
                                )
                            # Append output DeepONet `j` for the current
                            # timescale `i` 
                            y_i.append(y_ij)
                        # Append output timescale `i` for the final output
                        y.append(
                            tf_layers.Concatenate()(y_i) \
                                if len(y_i) > 1 else y_i[-1]
                        )
        return y



@tf.keras.utils.register_keras_serializable(package='PrODE', name='MMOpNNControllerler')
class MMOpNNController(Net):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim=None,
        names=None,
        opnn_dim=None,
        layer_size=None,
        activation=None,
        kernel_initializer=None,
        opnn_end_scale='bias',
        stacked=True,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        block_linear_out=True,
        cross_encoder=False,
        cross_encoder_layer_size=None,
        augmented=False,
        name='mmopnn_controller',
        *args,
        **kwargs
    ):
        super(MMOpNNController, self).__init__(
            dim=dim,
            block=block,
            block_linear_out=block_linear_out,
            name=name,
            *args,
            **kwargs
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
        if names == None:
            self.names = [f'output_{o+1}' for o in range(len(self.out_dim))]
        else:
            if len(names) <= 1:
                utils.raise_value_err(
                    "The number of `names` passed into `{}` has to be "
                    "grater than 1 (multiple DeepONets).".format(self.name))
            else:
                self.names = names
        self.opnn_dim = opnn_dim
        self.opnn_end_scale = opnn_end_scale
        self.stacked = stacked
        self.cross_encoder = cross_encoder
        self.cross_encoder_layer_size = cross_encoder_layer_size
        self.augmented = augmented

        self.opnns = [
            OpNN(
                self.opnn_dim[name],
                self.layer_size[name],
                self.activation[name],
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
                name='opnn_'+name
            ) for name in self.names
        ]

        # Apply I/O Transformation
        if self.opnn_inp_trans:
            if not isinstance(self.opnn_inp_trans, dict):
                utils.raise_value_err(
                    "`opnn_inp_trans` passed into `{}` has to be a `dict` in "
                    "which a specific input transform function is defined "
                    "for each DeepONet: {'name': fun}".format(self.name)
                )
            for i, (name, fun) in self.opnn_inp_trans.items():
                self.opnns[i].inp_trans = fun
        if self.opnn_out_trans:
            if not isinstance(self.opnn_out_trans, dict):
                utils.raise_value_err(
                    "`opnn_out_trans` passed into `{}` has to be a `dict` in "
                    "which a specific output transform function is defined "
                    "for each DeepONet: {'name': fun}".format(self.name)
                )
            for i, (name, fun) in self.opnn_out_trans.items():
                self.opnns[i].out_trans = fun

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'names': self.names,
            'opnn_dim': self.opnn_dim,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'opnn_end_scale': self.opnn_end_scale,
            'regularization': self.regularization,
            'stacked': self.stacked,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'cross_encoder': self.cross_encoder,
            'cross_encoder_layer_size': self.cross_encoder_layer_size,
            'augmented': self.augmented
        }
        base_config = super(MMOpNNController, self).get_config()
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
        mask=None,
        opnn_index=None
    ):
        # OpNNs
        G_uy = [ g_uy(x) for g_uy in self.opnns ]

        # Concatenate if single output or if final scaling is applied
        if not isinstance(self.out_dim, (list,tuple)):
            G_uy = tf_layers.Concatenate()(G_uy) if len(G_uy) > 1 else G_uy[-1]

        return G_uy
