import tensorflow as tf

from pycomet import utils
from pycomet.networks.basenet import BaseNet
from pycomet.networks.opnn import _init_input_lay, _manipulate_input


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
                "The properties for 'mmopnns' layer has to be provided "
                "as a dictionary of (mmopnn_name, mmopnn_args) pairs."
            )
        lays = {}
        for i, (name, ipropty) in enumerate(propty.items()):
            lays[name] = TSMMOpNN(
                dim=ipropty["dim"],
                lays_propty=ipropty["lays_propty"]
            )
        return lays
    
    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=False,
        tol=None
    ):
        # Inputs manipulation
        x = _manipulate_input(self, x)
        # Kernel - MMOpNNs
        y = []
        if training or (tol is None):
            for mmopnn in self.active_lays["mmopnns"].values():
                y.append(mmopnn(x, training=training))
        else:
            lays = self.active_lays["mmopnns"]
            nb_mmopnns = len(lays)
            for i, mmopnn in enumerate(lays.values()):
                if i == 0:
                    # First Timescale
                    y_i = mmopnn(x)
                else:
                    # Second to highest Timescales
                    if tf.math.reduce_any(y[-1] > tol):
                        y_i = mmopnn([x, y[-1]], tol=tol)
                    else:
                        y_i = tf.zeros(
                            (n, mmopnn.output_dim[-1]), dtype=K.floatx()
                        )
                y.append(y_i)
        return y


# =========================================================================== #
# TSMMOpNN
# =========================================================================== #
class TSMMOpNN(MMOpNN):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        *args,
        **kwargs
    ):
        super(TSMMOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )

    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=None,
        tol=None
    ):
        if training or (tol is None):
            return super(TSMMOpNN, self).call_net(x, training)
        else:
            return self.adaptive_call_net(x, tol)

    def adaptive_call_net(
        self,
        x,
        tol=None
    ):
        # Extract inputs: input + output previous timescale
        y, ym1 = x
        # Inputs manipulation
        y = _manipulate_input(self, y)
        # Global prenet
        y = _eval_prenet(self, y, training)
        # Kernel - OpNNs
        y = {name: tf.nest.map_structure(tf.identity, y) \
             for name in self.active_lays['opnns'].keys()}
        y = self._adaptive_call([y, ym1], tol)
        # Outputs manipulation
        y = self._manipulate_inout(
            inputs=dict(x=x, y=y), identifier='output', training=training
        )
        return y

    def _adaptive_call(
        self,
        x,
        tol=None
    ):
        y = []
        # Extract inputs: input + output previous timescale
        x, ym1 = x
        # Get number of input data
        n = utils.get_cardinality(tf.nest.flatten(x))
        # Get split values from previous Timescale (i-1)
        ym1 = tf.split(ym1, ym1.shape[1], axis=1)
        # Looping over each OpNN in the current Timescale
        for i, (name, opnn) in enumerate(self.active_lays['opnns'].items()):
            # Create a dummy `zeros` tensor with the i-th OpNN dimensions
            y_i = tf.zeros((n, opnn.output_dim[-1]), dtype=K.floatx())
            # If any values from previuos timescale is greater
            # than the `tol`, evaluates the current timescale
            if tf.math.reduce_any(ym1[i] > tol):
                # Get indices for the actual values to be evaluated
                idx = tf.reshape(tf.where(ym1[i] > tol)[:,0], shape=[-1])
                # Update the dummy `zeros` tensor with the
                # actual evaluation of the j-th OpNN
                x_i = tf.nest.map_structure(
                    lambda z: tf.gather(z, idx, axis=0), x
                )
                y_i = tf.tensor_scatter_nd_update(
                    y_i, tf.expand_dims(idx, 1), opnn(x_i)
                )
            # Append outputs of the j-th OpNN for the current i-th Timescale
            y.append(y_i)
        return y
