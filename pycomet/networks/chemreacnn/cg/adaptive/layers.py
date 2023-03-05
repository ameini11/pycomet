import numpy as np
import tensorflow as tf

from keras import layers as tf_layers
from pycomet import config as cfg
from pycomet.networks import elements as nn_elem

# Hypernet (v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo.specie_thermo


# =========================================================================== #
# CGEquilibrium Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class CGEquilibrium(tf_layers.Layer):

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
        super(CGEquilibrium, self).__init__(**kwargs)
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
            y = [tf_layers.Multiply()([tf.math.exp(-yi), Qi]) \
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
        base_config = super(CGEquilibrium, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
# =========================================================================== #
# CGThermoEncoder Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class CGThermoEncoder(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        tensor_toencode=0,
        nb_bins=1,
        indices_toencode=None,
        encoded_tensors_order=None,
        **kwargs
    ):
        super(CGThermoEncoder, self).__init__(**kwargs)
        # Tensor to encode
        self.tensor_toencode = tensor_toencode
        # Tensor's indices to encode
        self.nb_bins = int(nb_bins)
        self.indices_toencode = indices_toencode
        if self.indices_toencode == None:
            indices = np.arange(self.nb_bins)
            self.indices_toencode = {
                'rho_a': [0], 'rho_m': 1+indices, 'Th': [indices[-1]+2]
            }
        self.indices_toencode = {
            k: list(indices) for k, indices in self.indices_toencode.items()
        }
        # Encoded tensor's order
        self.encoded_tensors_order = encoded_tensors_order
        if self.encoded_tensors_order == None:
            self.encoded_tensors_order = {0: 'rho', 1: 'Y_m', 2: 'Th'}

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        # Extracting inputs quantities to encode
        x = {k: tf.gather(inputs[self.tensor_toencode], index, axis=1) \
            for k, index in self.indices_toencode.items()}
        # Encoded quantities
        y = {}
        # Encode mass quantities
        y['rho'] = tf.math.reduce_sum(x['rho_m'], axis=1, keepdims=1)
        y['rho'] = y['rho'] + x['rho_a']
        y['Y_m'] = x['rho_m'] / y['rho']
        # Encode temperature quantities
        y['Th'] = x['Th']
        # Compose and return outputs
        outputs = []
        outputs_i = tf.concat(
            [y[self.encoded_tensors_order[i]] for i in range(len(y))], 1
        )
        for i in range(len(inputs)):
            if i == self.tensor_toencode:
                outputs.append(outputs_i)
            else:
                outputs.append(inputs[i])
        return outputs

    # Configuration
    ###########################################################################
    def get_config(self):
        config = {
            "tensor_toencode": self.tensor_toencode,
            "nb_bins": self.nb_bins,
            "indices_toencode": self.indices_toencode,
            "encoded_tensors_order": self.encoded_tensors_order
        }
        base_config = super().get_config()
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
        y = tf.nest.map_structure(tf.identity, inputs)
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
