import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks import elements as nn_elem
from pycomet import config as cfg

# Hypernet (Version v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo.specie_thermo


# =========================================================================== #
# Equilibrium Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Equilibrium(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        output_dim,
        use_pf_int=False,
        specie_thermo=None,
        inputs_indices={'x': 0, 'T': 1, 'rho': 2}
        **kwargs
    ):
        super(Equilibrium, self).__init__(**kwargs)
        # Output dimensions
        self.output_dim = output_dim
        # Partition functions
        self.use_pf_int = use_pf_int
        if self.use_pf_int:
            # Specie thermodynamics
            self.sp_th = {}
            for index, s in specie_thermo.items():
                if "config" in s:
                    args = s
                else:
                    args = {**s, **dict(index=index, process=process)}
                sp_th = sp_th_mod.SpecieThermo(**args)
                self.sp_th[sp_th.specie.name] = sp_th
        # Inputs unpacking order
        self.inputs_keys = ('x', 'T', 'rho')
        self.inputs_indices = inputs_indices
        err_msg = f"For the `{self.name}` layer, 'inputs_indices' " \
            "has to be a `dict` with keys ('x', 'T', 'rho') and " \
            "values the corresponding indices for unpacking."
        if not isinstance(self.inputs_indices, dict):
            utils.raise_value_err(err_msg)
        else:
            for k in self.inputs_keys:
                if k not in self.inputs_keys:
                    utils.raise_value_err(err_msg)

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)):
            x, T, rho = [
                inputs[self.inputs_indices[k]] for k in self.inputs_keys
            ]
        else:
            utils.raise_value_err(
                f"The inputs of the `{self.name}` layer have "
                "to be a `list` or a `tuple`: [x, T, rho]"
            )
        if self.use_pf_int:
            # Partition functions
            Q = []
            for st in self.sp_th.values():
                st.pf.int.update(T)
                Q.append(st.pf.int.Q)
            Q = tf_layers.Concatenate()(Q) if len(Q) > 1 else Q[-1]
            # Simil-Softmax layer
            y = tf_layers.Multiply()([tf.math.exp(x), Q])
            y = nn_elem.Split(self.output_dim, axis=1)(y)
            y = [y_i / tf.reduce_sum(y_i, axis=-1, keepdims=True) for y_i in y]
        else:
            y = nn_elem.Split(self.output_dim, axis=1)(x)
            y = [tf_layers.Softmax(axis=-1)(y_i) for y_i in y]
        return y if len(y) > 1 else y[-1]

    # Configuration
    ###########################################################################
    def get_config(self):
        if self.use_pf_int:
            specie_thermo_cfg = {
                st.specie.index: {"config": st.get_config()} \
                    for st in self.sp_th.values()
            }
        else:
            specie_thermo_cfg = None
        config = {
            "output_dim": self.output_dim,
            "use_pf_int": self.use_pf_int,
            "specie_thermo": specie_thermo_cfg,
            "inputs_indices": self.inputs_indices
        }
        base_config = super(Equilibrium, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# FastestToCurrentTS Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class FastestToCurrentTS(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        fastest_to_current_ts,
        **kwargs
    ):
        super(FastestToCurrentTS, self).__init__(**kwargs)
        # Fastest-to-current timescale connectivity
        self.fastest_to_current_ts = fastest_to_current_ts

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        y = inputs
        if self.fastest_to_current_ts is not None:
            y_ = []
            for indeces in self.fastest_to_current_ts:
                y_.append(
                    tf.math.reduce_sum(
                        tf.gather(y, indeces, axis=1), axis=1, keepdims=1
                    )
                )
            y = tf_layers.Concatenate()(y_) if len(y_) > 1 else y_[-1]
        return y

    # Configuration
    ###########################################################################
    def get_config(self):
        config = {
            "fastest_to_current_ts": self.fastest_to_current_ts
        }
        base_config = super(FastestToCurrentTS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# CurrentToNextTS Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class CurrentToNextTS(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        current_to_next_ts,
        **kwargs
    ):
        super(CurrentToNextTS, self).__init__(**kwargs)
        # Current-to-next timescale connectivity
        self.current_to_next_ts = current_to_next_ts

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        y = inputs
        for ts, indeces in self.current_to_next_ts.items():
            if indeces is not None:
                # Split the current slower timescale
                y_slow = nn_elem.Split(y[ts].shape[1], axis=1)(y[ts])
                # Gather the next faster timescale by indeces
                y_fast = [tf.gather(y[ts+1], idx, axis=1) for idx in indeces]
                if len(y_slow) != len(y_fast):
                    y_slow = y_slow[-len(y_fast):]
                # Multiply slower and faster timescales
                y_mul = [tf_layers.Multiply()([ys, yf]) \
                    for ys, yf in zip(y_slow, y_fast)]
                y[ts+1] = tf_layers.Concatenate()(y_mul) \
                    if len(y_mul) > 1 else y_mul[-1]
        return y

    # Configuration
    ###########################################################################
    def get_config(self):
        config = {
            "current_to_next_ts": self.current_to_next_ts
        }
        base_config = super(CurrentToNextTS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
