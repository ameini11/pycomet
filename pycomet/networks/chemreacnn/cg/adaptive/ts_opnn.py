import tensorflow as tf

from tensorflow.keras import backend
from keras import layers as tf_layers
from pycomet import utils
from pycomet.networks.mmopnn import MMOpNN
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
from pycomet.networks.flexopnn import _eval_prenet
from pycomet.networks.opnn import _manipulate_input
from pycomet.networks.chemreacnn.cg.adaptive.layers import FastestToCurrentTS


class TimeScaleOpNN(BaseNet):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        fastest_to_current_ts=None,
        *args,
        **kwargs
    ):
        # Layers names
        super(TimeScaleOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )
        # Timescales ----------------------------------------------------------
        # Fastest-to-current timescale connectivity
        self.fastest_to_current_ts = fastest_to_current_ts
        # Layers --------------------------------------------------------------
        # Default layers
        self.default_lays_propty.update({
            "input": {
                "split_ic": {},
                "multilays": {}
            }
        })
        # Layers names
        if lays_name == None:
            self.lays_name = self.lays_name + ("mmopnn", "output")
        else:
            self.lays_name = lays_name
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=("mmopnn", "output"),
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
            if identifier == "input":
                lays[identifier] = self.init_input_lay(propty)
            elif identifier == "mmopnn":
                lays[identifier] = TSMMOpNN(
                    dim=propty["dim"],
                    lays_propty=propty["lays_propty"]
                )
            elif identifier == "output":
                lays[identifier] = self.init_inout_lay(propty)
        return lays

    def init_input_lay(self, propty):
        lays = {}
        for k in ("split_ic", "multilays"):
            if k == "multilays":
                if (k in propty) and (len(propty[k]) != 0):
                    lays[k] = self.get_lay(propty[k])
            elif k == "split_ic":
                if (k in propty) and (len(propty[k]) != 0):
                    _propty = propty[k]
                else:
                    _propty = dict(
                        num_or_size_splits=[1,self.input_dim[0]-2,1],
                        axis=1
                    )
                lays[k] = nn_elem.Split(**_propty)
        lays["ts_connect"] = FastestToCurrentTS(self.fastest_to_current_ts)
        return lays

    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=None,
        mask=None
    ):
        # Inputs manipulation
        x, T = self._manipulate_input(x, training)
        # Kernel - MMOpNNs
        y = self.active_lays["mmopnn"](
            x, training=training, mask=mask
        )
        # Equilibrium layer
        y = self._manipulate_inout(
            inputs=dict(y=[y, T]), identifier="output", training=training
        )
        # Outputs manipulation
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        return y

    def _manipulate_input(self, x, training):
        # Extract inputs
        ic, t, T = x
        # Extract initial physical inputs
        rho, Y0, T0 = self.active_lays["input"]["split_ic"](ic)
        if T is None:
            T = T0
        # Add mass fractions: micro-bins (fastest ts) to macro-bins (current ts)
        Y0 = self.active_lays["input"]["ts_connect"](Y0)
        # Recompose ICs
        ic = tf_layers.Concatenate()([rho, Y0, T0])
        # Transform ICs
        inp_lay = self.active_lays["input"]
        if "multilays" in inp_lay.keys():
            ic = inp_lay["multilays"](ic, training=training)
        # Recompose inputs
        return [ic, t], T


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
        mask=None
    ):
        if training or (mask is None):
            return super(TSMMOpNN, self).call_net(x, training)
        else:
            return self.adaptive_call_net(x, mask)

    def adaptive_call_net(
        self,
        x,
        mask=None
    ):
        y = x
        # Inputs manipulation
        y = _manipulate_input(self, y)
        # Global Prenet
        y = _eval_prenet(self, y)
        # Kernel - OpNNs
        y = self._adaptive_call(y, mask)
        # Outputs manipulation
        y = self._manipulate_inout(
            inputs=dict(x=x, y=y), identifier='output'
        )
        return y

    def _adaptive_call(
        self,
        x,
        mask
    ):
        y = []
        # Split neq. mask columnwise and loop over each column
        mask = nn_elem.Split(mask.shape[1], axis=1)(mask)
        # Default zeros
        zeros = tf.zeros_like(mask[-1], dtype=backend.floatx())
        for i, opnn in enumerate(self.active_lays['opnns'].values()):
            # Create a dummy `zeros` tensor
            y_i = tf.repeat(zeros, repeats=[opnn.output_dim[-1]], axis=1)
            # If any values of `mask_i` is `True`, evaluate the specific
            # data points
            mask_i = tf.squeeze(mask[i], axis=1)
            # Extract inputs given specific indices
            x_i = tf.nest.map_structure(
                lambda z: tf.boolean_mask(z, mask_i, axis=0), x
            )
            # Update the dummy `zeros` tensor with the
            # actual evaluation of the j-th OpNN
            y_i = tf.tensor_scatter_nd_update(y_i, tf.where(mask_i), opnn(x_i))
            # Append outputs of the j-th OpNN for the current i-th Timescale
            y.append(y_i)
        return y
