import tensorflow as tf

from tensorflow.keras import backend
from keras import layers as tf_layers
from pycomet import utils
from pycomet.networks.mmopnn import MMOpNN
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
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
                "xform_log": {},
                "xform_linscale": {},
                "prenet": {}
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
                lays[identifier] = MMOpNN(
                    dim=propty["dim"],
                    lays_propty=propty["lays_propty"]
                )
            elif identifier == "output":
                lays[identifier] = self.init_inout_lay(propty)
        return lays

    def init_input_lay(self, propty):
        lays = {}
        for k in ("split_ic", "xform_log", "xform_linscale", "prenet"):
            if k.startswith("xform"):
                if (k in propty) and (len(propty[k]) != 0):
                    _propty = {"xform": propty[k]}
                else:
                    _propty = {}
                lays[k] = self.init_inout_lay(_propty)
            elif k == "split_ic":
                if (k in propty) and (len(propty[k]) != 0):
                    _propty = propty[k]
                else:
                    _propty = dict(
                        num_or_size_splits=[1,self.input_dim[0]-2,1],
                        axis=1
                    )
                lays[k] = nn_elem.Split(**_propty)
            elif k == "prenet":
                if (k in propty) and (len(propty[k]) != 0):
                    lays[k] = self.get_lay(
                        propty={
                            'lay_id': 'PreNetBlock',
                            'kwargs': propty[k]['kwargs']
                        }
                    )
        lays["ts_connect"] = FastestToCurrentTS(self.fastest_to_current_ts)
        return lays

    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=None,
        neq_mask=None
    ):
        # Inputs manipulation
        x, T = self._manipulate_input(x, training)
        # Kernel - MMOpNNs
        if training or (neq_mask is None):
            y = self.active_lays["mmopnn"](x, training=training)
        else:
            y = self._adaptive_call_net(x, neq_mask)
        # Equilibrium layer
        y = self._manipulate_inout(
            inputs=dict(y=[y, T]), identifier="output", training=training
        )
        # Outputs manipulation
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        return y

    def _manipulate_input(self, x, training):
        # Extract inputs
        [ic, t], T = x
        # Extract initial physical inputs
        rho, Y0, T0 = self.active_lays["input"]["split_ic"](ic)
        if T is None:
            T = T0
        # Add mass fractions: micro-bins (fastest ts) to macro-bins (current ts)
        Y0 = self.active_lays["input"]["ts_connect"](Y0)
        # Recompose ICs
        ic = tf_layers.Concatenate()([rho, Y0, T0])
        # Transform ICs
        ic = self._manipulate_inout(
            inputs=dict(y=ic),
            lay=self.active_lays["input"]["xform_log"],
            training=training
        )
        ic = self._manipulate_inout(
            inputs=dict(y=ic),
            lay=self.active_lays["input"]["xform_linscale"],
            training=training
        )
        # Prenet for automatic time shifting
        if "prenet" in self.active_lays["input"].keys():
            t = self.active_lays["input"]["prenet"](
                x=[ic, t], training=training
            )
        # Recompose inputs
        return [ic, t], T

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
