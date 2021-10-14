import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from prode.utilities import utils
from prode.utilities import const


@tf.keras.utils.register_keras_serializable(package='PrODE', name='Equilibrium')
class Equilibrium(tf_layers.Subtract):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        database,
        grouping,
        name='equilibrium',
        **kwargs
    ):
        super(Equilibrium, self).__init__(name=name, **kwargs)

        # Retrieving Molecular data -------------------------------------------
        self.grouping = grouping
        self.levels_file = database + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = database + '/grouping/O3_UMN/O2/LevelsMap_' \
            + self.grouping + '.csv'
        self.g, self.E, self.num_lev = self.read_levels()
        self.lev_to_bin, self.num_bins = self.read_groups()

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'grouping': self.grouping,
            'levels_file': self.levels_file,
            'binMap_file': self.binMap_file,
            'g': self.g,
            'E': self.E,
            'num_lev': self.num_lev,
            'lev_to_bin': self.lev_to_bin,
            'num_bins': self.num_bins
        }
        base_config = super(Equilibrium, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, T = inputs
        else:
            utils.raise_value_err(
                "The inputs of the `{}` layer have to be a `list` "
                "or a `tuple`: [x, T]".format(self.name))
        e = self.energy_bins(T) * tf.math.reciprocal(T) / const.UKB
        return super(Equilibrium, self).call([x, e])

    def energy_bins(self, T):
        '''Compute bins partition function.'''
        T_rec = tf.math.reciprocal(tf.transpose(T))
        q = tf.math.multiply(
            self.g, tf.math.exp(
                - tf.tensordot(self.E, T_rec, 1) / const.UKB
            )
        )
        E_bins = []
        for bin_i in range(self.num_bins):
            mask = self.lev_to_bin == bin_i
            Q_bin_i = tf.math.reduce_sum(q[mask], axis=0, keepdims=True)
            E_bin_i = self.E[mask] * q[mask] * tf.math.reciprocal(Q_bin_i)
            E_bins.append(
                tf.math.reduce_sum(E_bin_i, axis=0, keepdims=True)
            )
        return tf.transpose(tf.concat(E_bins, 0))

    # Util Functions
    ###########################################################################
    def read_levels(self):
        '''Retrieve all the ro-vib levels.'''
        data = pd.read_csv(
            self.levels_file, header=None, skiprows=15, delim_whitespace=True
        )
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn = np.array(data[0])                     # Vibrational Q.N.
        jqn = np.array(data[1])                     # Rotational  Q.N.
        g = 1.5*(2.0*jqn+1.0)                       # Degeneracy
        E = np.array(data[2]) * const.EH_to_J       # Energy in J
        E = E - E[0]
        num = len(vqn)
        g, E = list(map(lambda x: tf.expand_dims(tf.convert_to_tensor(x, \
            dtype=backend.floatx()), 1), [g, E]))
        return g, E, num

    def read_groups(self):
        '''Retrieve mapping of each rot-vib level to the respective bin.'''
        data = pd.read_csv(self.binMap_file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx = np.array(data[0])                                                                           
        lev_to_bin = np.array(data[1]) - 1    # Level-to-Bin
        num = max(lev_to_bin) + 1
        return lev_to_bin, num
