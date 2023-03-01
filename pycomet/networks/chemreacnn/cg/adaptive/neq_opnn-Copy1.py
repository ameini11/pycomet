import copy
import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities import utils
from prode.networks.opnn import OpNN
from prode.networks.net import Net
from tensorflow.python.keras import backend as K


class NeqOpNN(BaseNet):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        force_siso=False,
        *args,
        **kwargs
    ):
        # Layers names
        super(NeqOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            force_siso=force_siso,
            *args,
            **kwargs
        )
        # Layers --------------------------------------------------------------
        # Default layers
        self.default_lays_propty.update({
            "input": {
                "branch": {
                    "index": 0,
                    "xform": None
                },
                "trunk": {
                    "index": 1,
                    "xform": None
                }
            },
            "output": {
                "xform": None
            }
        })
        # Layers names
        if lays_name == None:
            self.lays_name = self.lays_name + ("mmopnns",)
        else:
            self.lays_name = lays_name
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=("mmopnns",),
            default=self.default_lays_propty
        )
        # Initializing
        self.active_lays = self.init_layers()
        self.init_preinput_lay()

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                lays[identifier] = _init_input_lay(self, identifier, propty)
            elif identifier == 'mmopnns':
                lays[identifier] = self.init_mmopnns_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_inout_lay(propty)
        return lays

    def init_mmopnns_lay(self, propty):
        if not isinstance(propty, dict):
            utils.raise_value_err(
                "The properties for 'mass' layer has to be provided "
                "as a dictionary of (ts_opnn_name, ts_opnn_args) pairs."
            )
        lays = {}
        for i, (name, ipropty) in enumerate(propty.items()):
            lays[name] = TimeScaleOpNN(
                dim=ipropty["dim"],
                lays_propty=ipropty["lays_propty"],
                fastest_to_current_ts=self.fastest_to_current_ts[i]
            )
        return lays

    
    

    
    

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
        opnn_end_scale='bias',
        stacked_blocks={"branch": True, "trunk": False},
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        block_linear_out=True,
        shifter=None,
        augmented=False,
        name='neqcrtlnn',
        *args,
        **kwargs
    ):
        super(NeqCtrlNN, self).__init__(
            dim=dim,
            block=block,
            block_linear_out=block_linear_out,
            name=name,
            *args,
            **kwargs
        )

        self.sub_nets = ("branch", "trunk")

        self.names = tuple(tf.nest.flatten(names))
        self.num_mmopnns = len(self.names)

        ### Dimensionality ----------------------------------------------------
        # MMOpNNs
        self.mmopnn_dim = {
            name: {
                "inp": self.inp_dim,
                "out": tf.nest.flatten(self.out_dim)[i]
            } for i, name in enumerate(self.names)
        }

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
        self.shifter = utils.value_to_dict(shifter, self.names)
        self.augmented = augmented

        ### MMOpNNCtrl Initialization -----------------------------------------
        self.opnn_dim = utils.value_to_dict(opnn_dim, self.names)
        self.opnn_inp_trans = utils.value_to_dict(opnn_inp_trans, self.names)
        self.opnn_out_trans = utils.value_to_dict(opnn_out_trans, self.names)
        self.opnn_end_scale = utils.value_to_dict(opnn_end_scale, self.names)
        self.stacked_blocks = stacked_blocks

        self.mmopnns = {
            name: MMOpNNCtrl(
                dim=self.mmopnn_dim[name],
                opnn_dim=self.opnn_dim[name],
                layer_size=self.layer_size[name],
                activation=self.activation[name],
                kernel_initializer=self.kernel_initializer[name],
                opnn_inp_trans=self.opnn_inp_trans[name],
                opnn_out_trans=self.opnn_out_trans[name],
                opnn_end_scale=self.opnn_end_scale[name],
                stacked_blocks=self.stacked_blocks,
                regularization=self.regularization,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                block=self.block,
                block_linear_out=self.block_linear_out,
                shifter=self.shifter[name],
                augmented=self.augmented,
                name=name
            ) for i, name in enumerate(self.names)
        }

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
            'opnn_inp_trans': self.opnn_inp_trans,
            'opnn_out_trans': self.opnn_out_trans,
            'opnn_end_scale': self.opnn_end_scale,
            'regularization': self.regularization,
            'stacked_blocks': self.stacked_blocks,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'shifter': self.shifter,
            'augmented': self.augmented
        }
        base_config = super(NeqCtrlNN, self).get_config()
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
        tol=None
    ):
        # Inputs manipulation
        x = self._manipulate_input(x)
        # Kernel - MMOpNNs
        if training or tol is None:
            y = [self.mmopnns[n](x, training=training) for n in self.names]
        else:
            y = self._adaptive_call(x, tol)
        return y

    def _manipulate_input(self, x):
        if isinstance(x, (list, tuple)):
            x = {k: x[i] for i, k in enumerate(self.sub_nets)}
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))
        if self.inp_trans is not None:
            for k in self.sub_nets:
                if isinstance(self.inp_trans, dict):
                    x[k] = self.inp_trans[k](x[k])
                else:
                    x[k] = self.inp_trans(x[k])
        # Inputs flattening
        x = tf.nest.flatten(x)
        return x

    def _adaptive_call(self, x, tol):
        y = []
        n = utils.get_cardinality(x)
        # Looping over each Timescale
        for i, (name, mmopnn) in enumerate(self.mmopnns.items()):
            if i == 0:
                # First Timescale
                y.append(mmopnn(x))
            else:
                # Second to highest Timescales
                if i != self.num_mmopnns-1 and all(y[-1] <= tol):
                    y.append(
                        tf.zeros((n, mmopnn.out_dim), dtype=K.floatx())
                    )
                else:
                    y_i = []
                    # Get split values from previous Timescale
                    y_im1 = tf.split(y[-1], y[-1].shape[1], axis=1)
                    # Looping over each OpNN in the current i-th Timescale
                    for j, y_j in enumerate(y_im1):
                        # Get j-th OpNN name
                        name = mmopnn.names[j]
                        # Create a dummy `zeros` tensor with the j-th OpNN
                        # dimensions
                        y_ij = tf.zeros(
                            (n, mmopnn.opnns[name].out_dim),
                            dtype=K.floatx()
                        )
                        # If any values from previuos timescale is greater
                        # than the `tol`, evaluates the current timescale
                        if any(y_j > tol):
                            # Get indices for the actual values to be
                            # evaluated
                            idx = tf.reshape(
                                tf.where(y_j > tol)[:,0], shape=[-1]
                            )
                            # Update the dummy `zeros` tensor with the
                            # actual evaluation of the j-th OpNN
                            x_j = tf.nest.map_structure(
                                lambda x: tf.gather(x, idx, axis=0), x
                            )
                            y_ij = tf.tensor_scatter_nd_update(
                                y_ij,
                                tf.expand_dims(idx, 1),
                                mmopnn.opnns[name](x_j)
                            )
                        # Append outputs of the j-th OpNN for the current
                        # i-th Timescale
                        y_i.append(y_ij)
                    # Append output of the current i-th Timescale
                    # for the final output
                    y.append(
                        tf_layers.Concatenate()(y_i) \
                            if len(y_i) > 1 else y_i[-1]
                    )
        return y



@tf.keras.utils.register_keras_serializable(package='PrODE', name='MMOpNNCtrl')
class MMOpNNCtrl(Net):
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
        opnn_inp_trans=None,
        opnn_out_trans=None,
        opnn_end_scale='bias',
        stacked_blocks={"branch": True, "trunk": False},
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        block_linear_out=True,
        shifter=None,
        augmented=False,
        name='mmopnn_controller',
        *args,
        **kwargs
    ):
        super(MMOpNNCtrl, self).__init__(
            dim=dim,
            block=block,
            block_linear_out=block_linear_out,
            name=name,
            *args,
            **kwargs
        )

        ### Dimensionality ----------------------------------------------------
        # Flatten output dimension
        opnn_dim["out"] = tf.nest.flatten(opnn_dim["out"])

        # Names
        if names == None:
            self.names = tuple([
                f'opnn_{i+1}' for i in range(len(opnn_dim["out"]))
            ])
        else:
            if len(names) <= 1:
                utils.raise_value_err(
                    "The number of `names` passed into `{}` has to be "
                    "grater than 1 (multiple DeepONets).".format(self.name))
            else:
                self.names = names

        # OpNNs
        self.opnn_dim = {
            name: {
                "inp": opnn_dim["inp"],
                "out": opnn_dim["out"][i]
            } for i, name in enumerate(self.names)
        }

        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### OpNNs Initialization ----------------------------------------------
        self.opnn_inp_trans = opnn_inp_trans
        self.opnn_out_trans = opnn_out_trans
        self.opnn_end_scale = opnn_end_scale
        self.stacked_blocks = stacked_blocks
        self.shifter = shifter
        self.augmented = augmented

        self.opnns = {
            name: OpNN(
                self.opnn_dim[name],
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
                name=name
            ) for name in self.names
        }

        # Apply I/O Transformation
        for i, (name, opnn) in enumerate(self.opnns.items()):
            if self.opnn_inp_trans is not None:
                if isinstance(self.opnn_inp_trans, (list,tuple)):
                    opnn.inp_trans = self.opnn_inp_trans[i]
                else:
                    opnn.inp_trans = self.opnn_inp_trans
            if self.opnn_out_trans is not None:
                if isinstance(self.opnn_out_trans, (list,tuple)):
                    opnn.out_trans = self.opnn_out_trans[i]
                else:
                    opnn.out_trans = self.opnn_out_trans

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'names': self.names,
            'opnn_dim': self.opnn_dim,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'opnn_inp_trans': self.opnn_inp_trans,
            'opnn_out_trans': self.opnn_out_trans,
            'opnn_end_scale': self.opnn_end_scale,
            'regularization': self.regularization,
            'stacked_blocks': self.stacked_blocks,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'shifter': self.shifter,
            'augmented': self.augmented
        }
        base_config = super(MMOpNNCtrl, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(
        self,
        x,
        training=False
    ):
        # OpNNs
        y = [self.opnns[n](x, training=training) for n in self.names]
        # Concatenate if single output or if final scaling is applied
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        return y
