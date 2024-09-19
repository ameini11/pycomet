import tensorflow as tf

from keras import layers as tf_layers
from tensorflow.python.keras import backend
from pycomet import utils
from pycomet.networks.mmopnn import MMOpNN
from pycomet.networks.basenet import BaseNet
from pycomet.networks.flexopnn import _eval_prenet
from pycomet.networks.opnn import _init_input_lay, _manipulate_input
from pycomet.networks.chem.cg.adaptive.tools.groups_connect import GroupsConnect


class NeqOpNN(BaseNet):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        groups_graph,
        lays_name=None,
        lays_propty=None,
        force_siso=False,
        inference=None,
        *args,
        **kwargs
    ):
        # Layers names
        super(NeqOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            force_siso=force_siso,
            inference=inference,
            *args,
            **kwargs
        )
        # Timescales connectivity ---------------------------------------------
        self.grp_conn = GroupsConnect(groups_graph)
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
                lays[identifier] = self.init_mmopnns_lay(propty)
        return lays

    def init_mmopnns_lay(self, propty):
        if not isinstance(propty, dict):
            utils.raise_value_err(
                "The properties for 'mmopnns' layer has to be provided "
                "as a dictionary of (mmopnn_name, mmopnn_args) pairs."
            )
        lays = {}
        for i, (name, ipropty) in enumerate(propty.items()):
            lays[name] = NeqMMOpNN(
                dim=ipropty["dim"],
                lays_propty=ipropty["lays_propty"]
            )
        return lays
    
    # Calling
    ###########################################################################
    def forward(
        self,
        x,
        training=None
    ):
        y = x
        # Inputs manipulation
        y = _manipulate_input(self, y)
        # Kernel - MMOpNNs
        y_ = []
        for mmopnn in self.active_lays["mmopnns"].values():
            y_.append(mmopnn(y, training=training))
        return y_
    
    def forward_infer(
        self,
        x,
        x_extra,
        *args,
        **kwargs
    ):
        # Extract inputs
        y, tol = x, x_extra[-1]
        # Inputs manipulation
        y = _manipulate_input(self, y)
        # Kernel - MMOpNNs
        y_ = []
        for i, mmopnn in enumerate(self.active_lays["mmopnns"].values()):
            if i == 0:
                # First Timescale ---------------------------------------------
                y_i = mmopnn(x=y)
            else:
                # Second to highest Timescales --------------------------------
                # Previous Timescale
                ym1 = y_[i-1]
                # Tot. nb. of micro bins belonging to the previous timescale
                ratio = self.grp_conn.fastest_to_current_ts[f'TS{i-1}']
                y_i = mmopnn(x=[y, ym1], ratio=ratio, tol=tol)
            y_.append(y_i)
        # Outputs manipulation
        y = tf.nest.map_structure(
            lambda x: tf.cast(x > 0, dtype=backend.floatx()), y_
        )
        return y

# =========================================================================== #
# TSMMOpNN
# =========================================================================== #
class NeqMMOpNN(MMOpNN):

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
        super(NeqMMOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )

    # Calling
    ###########################################################################
    def forward(
        self,
        x,
        training=None,
        ratio=None,
        tol=None
    ):
        if training or (tol is None):
            y = super(NeqMMOpNN, self).forward(x, training)
        else:
            y = self.adaptive_forward(x, ratio, tol)
        y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
        return y

    def adaptive_forward(
        self,
        x,
        ratio,
        tol
    ):
        # Extract inputs: input + output previous timescale
        x, ym1 = x
        # Inputs manipulation
        y = _manipulate_input(self, x)
        # Global prenet
        y = _eval_prenet(self, y)
        # Kernel - OpNNs
        y = self._adaptive_call(y, ym1, ratio, tol)
        # Outputs manipulation
        y = self._manipulate_inout(
            inputs=dict(x=x, y=y), identifier='output'
        )
        return y

    def _adaptive_call(
        self,
        x,
        ym1,
        ratio,
        tol
    ):
        y = []
        # Get split values from previous Timescale (i-1)
        ym1 = tf.split(ym1, ym1.shape[1], axis=1)
        # Default zeros
        zeros = tf.zeros_like(ym1[-1], dtype=backend.floatx())
        # Looping over each OpNN in the current Timescale
        for i, (name, opnn) in enumerate(self.active_lays['opnns'].items()):
            # Create a dummy `zeros` tensor with the i-th OpNN dimensions
            y_i = tf.repeat(zeros, repeats=[opnn.output_dim[-1]], axis=1)
            # If any values from previuos timescale is greater
            # than the `tol`, evaluates the current timescale
            mask = tf.squeeze(ym1[i] > tol * len(ratio[i]), axis=1)
            # Update the dummy `zeros` tensor with the
            # actual evaluation of the j-th OpNN
            x_i = tf.nest.map_structure(
                lambda z: tf.boolean_mask(z, mask, axis=0), x
            )
            y_i = tf.tensor_scatter_nd_update(y_i, tf.where(mask), opnn(x_i))
            # Append outputs of the j-th OpNN for the current i-th Timescale
            y.append(y_i)
        return y
