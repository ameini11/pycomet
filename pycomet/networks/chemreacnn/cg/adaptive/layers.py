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
        output_dim=2,
        use_pf_int=False,
        specie_thermo=None,
        process=None,
        **kwargs
    ):
        super(Equilibrium, self).__init__(**kwargs)
        # Output dimensions
        self.output_dim = tf.nest.flatten(output_dim)
        # Partition functions
        self.use_pf_int = use_pf_int
        if self.use_pf_int:
            # Specie thermodynamics
            if "config" in specie_thermo:
                args = specie_thermo
            else:
                args = {**specie_thermo, **dict(index=0, process=process)}
            self.sp_th = sp_th_mod.SpecieThermo(**args)

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)):
            x, T = inputs
        else:
            utils.raise_value_err(
                f"The inputs of the `{self.name}` layer have "
                "to be a `list` or a `tuple`: [x, T]"
            )
        y = tf.nest.flatten(x)
        if self.use_pf_int:
            # Partition functions
            Q = self.sp_th.pf.int.spline['Q'].apply(points=T)
            Q = nn_elem.Split(self.output_dim, axis=-1)(Q)
            # Simil-Softmax layer
            y = [tf_layers.Multiply()([tf.math.exp(yi), Qi]) \
                for yi, Qi in zip(y, Q)]
            y = [yi / tf.math.reduce_sum(yi, axis=-1, keepdims=1) for yi in y]
        else:
            y = [tf_layers.Softmax(axis=-1)(yi) for yi in y]
        return y

    # Configuration
    ###########################################################################
    def get_config(self):
        if self.use_pf_int:
            specie_thermo_cfg = {"config": self.sp_th.get_config()}
        else:
            specie_thermo_cfg = None
        config = {
            "output_dim": self.output_dim,
            "use_pf_int": self.use_pf_int,
            "specie_thermo": specie_thermo_cfg
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
