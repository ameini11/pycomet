import tensorflow as tf

from tensorflow.keras import backend
from pycomet import utils
from pycomet.networks import MMOpNN
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
from pycomet.networks.chemreacnn.cg.adaptive import TSFastestToCurrent


class TimeScaleOpNN(BaseNet):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_propty,
        fastest_to_current_ts,
        *args,
        **kwargs
    ):
        # Layers names
        lays_name = ("input", "mmopnn", "equilibrium")
        super(TimeScaleOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )
        # Layers --------------------------------------------------------------
        # Checking
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=("mmopnn",)
        )
        # Initializing
        self.active_lays = self.init_layers()
        # Timescales ----------------------------------------------------------
        # Fastest-to-current timescale connectivity
        self.fastest_to_current_ts = fastest_to_current_ts

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == "input":
                lays[identifier] = self.init_input_lay(propty)
            elif identifier == "mmopnn":
                lays[identifier] = MMOpNN(
                    dim=propty["dim"],
                    lays_propty=propty["lays_propty"]
                )
            elif identifier == "equilibrium":
                propty["output_dim"] = lays["mmopnn"].output_dim
                lays[identifier] = self.init_inout_lay(propty)
        return lays

    def init_input_lay(self, propty):
        lays = {}
        for k in ("xform_temp", "xform_mass", "split_ic"):
            if k.startswith("xform"):
                _propty = propty[k] if k in propty else {}
                lays[k] = self.init_inout_lay(propty[k])
            else:
                _propty = propty[k] if k in propty else dict(
                    num_or_size_splits=[1,1,self.input_dim[0]-2],
                    axis=1
                )
                lays[k] = nn_elem.Split(**propty[k])
        lays["ts_connect"] = FastestToCurrentTS(self.fastest_to_current_ts)
        return lays

    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=None,
        neq_mask=None,
        eq_lay=None
    ):
        # Inputs manipulation
        x, T, rho = self._manipulate_input(x, training)
        # Kernel - MMOpNNs
        if training or (neq_mask is None):
            y = self.active_lays["mmopnn"](x, training=training)
        else:
            y = self._adaptive_call_net(x, neq_mask)
        # Equilibrium layer
        if eq_lay is not None:
            y = self.active_lays["equilibrium"](
                inputs=[y, T, rho], training=training
            )
        # Outputs manipulation
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        return y

    def _manipulate_input(self, x, training):
        # Extract inputs
        T, [ic, t] = x
        # Extract initial physical inputs
        T0, rho, Y0 = self.active_lays["input"]["split_ic"](ic)
        if T is None:
            T = T0
        # Add mass fractions: micro-bins (fastest ts) to macro-bins (current ts)
        Y0 = self.active_lays["input"]["ts_connect"](Y0)
        # Xform mass fractions
        Y0 = self._manipulate_inout(
            inputs=dict(x=Y0, y=None),
            lay=self.active_lays["input"]["xform_mass"],
            training=training
        )
        # Xform temperature
        T0 = self._manipulate_inout(
            inputs=dict(x=T0, y=None),
            lay=self.active_lays["input"]["xform_temp"],
            training=training
        )
        # Recompose ICs
        ic = tf_layers.Concatenate()([T0, rho, Y0])
        # Recompose inputs
        x = [ic, t]
        return x, T, rho

    def _adaptive_call_net(self, x, neq_mask):
        y = []
        # Get number of input data
        n = utils.get_cardinality(x)
        # Split `neq_mask` columnwise and loop over each column
        neq_mask = nn_elem.Split(neq_mask.shape[1], axis=1)(neq_mask)
        opnns = self.active_lays["mmopnn"].active_lays["opnns"]
        for i, (name, opnn) in enumerate(opnns.items()):
            # Create a dummy `zeros` tensor
            y_i = tf.zeros((n, self.ouput_dim[i]), dtype=backend.floatx())
            # If any values of `mask_i` is `True`, evaluate the specific
            # data points
            if tf.math.reduce_any(neq_mask[i]):
                # Get indices for the actual data to be evaluated
                idx = tf.reshape(tf.where(neq_mask[i])[:,0], shape=[-1])
                # Extract inputs given specific indices
                x_i = tf.nest.map_structure(
                    lambda x: tf.gather(x, idx, axis=0), x
                )
                # Update the dummy `zeros` tensor with the actual
                # evaluation of the DeepONet `i`
                y_i = tf.tensor_scatter_nd_update(
                    y_i,
                    tf.expand_dims(idx, 1),
                    opnn(x_i)
                )
            y.append(y_i)
        return y
